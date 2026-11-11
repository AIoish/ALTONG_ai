"""용도: 실제 입력을 읽고 작성한 관련도 제안을 정확한 입력 지문에 연결해 검수 사본으로 저장한다.
생성일: 2026-11-08
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import re

from filtering_training.common.artifact_io import created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import input_text
from filtering_training.preparation.curate_relevance import key, write_lines
from filtering_training.preparation.prepare_real_relevance import OUTPUT


def merge_decisions(records, decisions):
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate source IDs')
    by_id = {}
    for decision in decisions:
        if decision['id'] in by_id:
            raise ValueError('duplicate decision ID')
        score = decision['proposed_score']
        if score is not None and (type(score) is not int or score not in (1, 2, 3)):
            raise ValueError('expected relevance 1-3 or unresolved null')
        if not decision.get('reason', '').strip() or decision.get('reviewer') not in ('assistant', 'user'):
            raise ValueError('every decision needs a reason and reviewer attribution')
        by_id[decision['id']] = decision
    if set(by_id) != {r['id'] for r in records}:
        raise ValueError('decisions must cover every source ID exactly once')
    result = []
    for row in records:
        decision = by_id[row['id']]
        signature = key(json.loads(input_text(row['state'], 'relevance')))
        if signature != decision['input_sha256']:
            raise ValueError('reviewed model input changed: ' + row['id'])
        if row['score'] is not None or row['review_status'] != 'needs_relevance_label':
            raise ValueError('expected original unlabeled import')
        labeled = deepcopy(row)
        labeled['score'] = decision['proposed_score']
        labeled['review_status'] = ('needs_context_review' if labeled['score'] is None else
                                   'ai_proposed_unreviewed' if decision['reviewer'] == 'assistant' else 'user_labeled')
        labeled['review'] = deepcopy(decision)
        labeled['training_eligible'] = False
        result.append(labeled)
    return result


def apply_review(folder=OUTPUT, *, decisions_path, batch='assistant-pass-01'):
    folder, decisions_path = Path(folder).resolve(), Path(decisions_path).resolve()
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', batch):
        raise ValueError('invalid batch name')
    output = folder / 'review' / batch
    if output.exists():
        raise ValueError('prior review must be preserved; choose a new batch')
    parent = read(folder / 'manifest.json')
    protected = {str(folder / name): info['sha256'] for name, info in parent['files'].items()}
    protected.update({parent['source']: parent['source_sha256'], str(folder / 'manifest.json'): digest(folder / 'manifest.json'),
                      str(decisions_path): digest(decisions_path)})
    if any(digest(Path(path)) != expected for path, expected in protected.items()):
        raise ValueError('import source changed')
    records = load_lines(folder / 'relevance/unlabeled.jsonl')
    decisions = load_lines(decisions_path)
    rows = merge_decisions(records, decisions)
    output.mkdir(parents=True)
    write_lines(output / 'proposed.jsonl', rows)
    write_lines(output / 'decisions.jsonl', decisions)
    write_lines(output / 'unresolved.jsonl', [r for r in rows if r['score'] is None])
    columns = ('source_number', 'id', 'provisional_group', 'notification_app', 'title', 'body',
               'current_app', 'current_title', 'recent_windows', 'proposed_relevance_1_to_3', 'reason',
               'confidence', 'reviewed_relevance_1_to_3', 'reviewer_note')
    with (output / 'review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            n, c = row['state']['notification'], row['state']['context']
            writer.writerow({'source_number': row['source_number'], 'id': row['id'], 'provisional_group': row['group_id'],
                             'notification_app': n['app_name'], 'title': n['title'], 'body': n['body'],
                             'current_app': c['active_process'], 'current_title': c['window_title'],
                             'recent_windows': json.dumps(c['recent_windows'], ensure_ascii=False),
                             'proposed_relevance_1_to_3': '' if row['score'] is None else row['score'],
                             'reason': row['review']['reason'], 'confidence': row['review'].get('confidence', ''),
                             'reviewed_relevance_1_to_3': '', 'reviewer_note': ''})
    if any(digest(Path(path)) != expected for path, expected in protected.items()):
        raise ValueError('source changed during annotation')
    manifest = {'purpose': 'Proposed real-data relevance labels; no stored predictions used as gold',
                'created_date': created_date(), 'status': 'proposed_labels_pending_context_and_group_review',
                'rows': len(rows), 'proposed_score_counts': dict(Counter(r['score'] for r in rows if r['score'] is not None)),
                'held_rows': sum(r['score'] is None for r in rows),
                'held_reasons': dict(Counter(r['review'].get('reason_code') for r in rows if r['score'] is None)),
                'reviewer_counts': dict(Counter(r['review']['reviewer'] for r in rows)),
                'training_ready': False, 'splits_assigned': False, 'inputs_unchanged': True,
                'source_files_unchanged': True, 'protected_hashes': protected,
                'files': {name: {'sha256': digest(output / name)} for name in (
                    'proposed.jsonl', 'decisions.jsonl', 'unresolved.jsonl', 'review.csv')},
                'limitations': ['Assistant proposals are not independently verified ground truth.',
                                'Keep simulated/tool-test rows and uncertain referents separate.',
                                'Provisional groups and data usage must be settled before any train/evaluation split.']}
    write(output / 'manifest.json', manifest)
    write(folder / 'review/current-review.json', {'purpose': 'Latest real-data annotation review',
          'created_date': created_date(), 'batch': batch, 'manifest': str((output / 'manifest.json').relative_to(folder)),
          'review_csv': str((output / 'review.csv').relative_to(folder)), 'training_ready': False})
    print(json.dumps({name: manifest[name] for name in ('rows', 'proposed_score_counts', 'held_rows', 'held_reasons')}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--batch', default='assistant-pass-01')
    args = parser.parse_args()
    apply_review(args.folder, decisions_path=args.decisions, batch=args.batch)
