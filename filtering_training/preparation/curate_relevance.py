"""용도: v13 관련도에서 같은 의미·맥락의 반복 인사말 변형만 줄이고 새 검수 버전을 만든다.
생성일: 2026-11-08
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unicodedata

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_data import DEFAULT_DATASET, load_windows_prepared
from filtering_training.common.encoder_input import INPUT_SCHEMA, RECENT_WINDOWS_POLICY, input_text

OUTPUT = ROOT / 'filtering_training/outputs/v14_relevance_curation_01'
SCORE_MAP = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
# Exact wrappers used in the old synthetic urgent-scenario expansion. No fuzzy deletion.
PREFIXES = ('확인 부탁드립니다. ', '현장 전달: ', '담당자 연락입니다. ',
            '상황 안내입니다. ', '담당 확인 요청: ', '중요 연락입니다. ')
SUFFIXES = (' 안내를 놓치지 않도록 확인 부탁드립니다.',
            ' 지금 담당 연락을 확인해주세요.', ' 현장 안내가 전달됐습니다.')


def key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def core_body(body):
    for prefix in PREFIXES:
        if body.startswith(prefix):
            return body[len(prefix):]
    for suffix in SUFFIXES:
        if body.endswith(suffix):
            return body[:-len(suffix)]
    return body


def body_group(body):
    # Prefix-only rewrites must not become independent evaluation examples.
    normalized = ''.join(unicodedata.normalize('NFKC', core_body(body)).split()).casefold()
    return 'body_' + key(normalized)[:20]


def profile(records):
    app = defaultdict(Counter)
    history = defaultdict(Counter)
    for row in records:
        app[row['state']['notification']['app_name']][row['score']] += 1
        history[len(row['state']['context']['recent_windows'])][row['score']] += 1
    return {'rows': len(records), 'scores': dict(Counter(r['score'] for r in records)),
            'body_groups': len({r['group_id'] for r in records}),
            'body_at_most_10': sum(len(r['state']['notification']['body'].strip()) <= 10 for r in records),
            'notification_app_by_score': {k: dict(v) for k, v in sorted(app.items())},
            'history_length_by_score': {str(k): dict(v) for k, v in sorted(history.items())}}


def curate(rows, *, keep_variants=2):
    if type(keep_variants) is not int or keep_variants < 1:
        raise ValueError('keep_variants must be a positive integer')
    groups = defaultdict(list)
    for line, row in enumerate(rows, 1):
        score5 = int(row['expected']['relevance'])
        if score5 not in SCORE_MAP:
            raise ValueError('invalid source relevance label')
        payload = json.loads(input_text(row['state'], 'relevance'))
        body = payload['notification']['body']
        payload['notification']['body'] = core_body(body)
        # Every model-visible field except the exact wrapper must match.
        signature = key({'payload': payload, 'score5': score5})
        record = {'id': f'v13rel_{line:05d}', 'group_id': body_group(body),
                  'split': 'train_only', 'state': deepcopy(row['state']),
                  'score': SCORE_MAP[score5], 'source_score5': score5,
                  'review_status': 'inherited_label_not_rereviewed',
                  'provenance': {'source_file': 'relevance/train.jsonl', 'source_line': line}}
        groups[signature].append(record)
    retained, excluded, repeated = [], [], []
    for signature, group in groups.items():
        ranked = sorted(group, key=lambda r: (len(r['state']['notification']['body']), r['id']))
        chosen = ranked[:keep_variants]
        retained.extend(chosen)
        if len(group) > 1:
            repeated.append({'signature': signature, 'rows': len(group),
                             'kept_ids': [r['id'] for r in chosen],
                             'all_ids': [r['id'] for r in group]})
        excluded.extend({'id': r['id'], 'group_id': r['group_id'],
                         'reason': 'same_core_body_same_model_input_same_label_wrapper_repetition',
                         'retained_ids': [x['id'] for x in chosen],
                         'source_line': r['provenance']['source_line']}
                        for r in ranked[keep_variants:])
    retained.sort(key=lambda r: r['provenance']['source_line'])
    excluded.sort(key=lambda r: r['source_line'])
    return retained, excluded, repeated


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8'))


def prepare(output=OUTPUT, source=DEFAULT_DATASET, keep_variants=2):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists():
        raise ValueError('Use a new output folder; existing data is preserved')
    sources, protected, _ = load_windows_prepared(source)
    source_manifest = read(source / 'manifest.json')
    protected.update(source_manifest.get('protected_sources', {}))
    for name, info in source_manifest['files'].items():
        protected[str(source / name)] = info['sha256']
    if any(digest(Path(p)) != h for p, h in protected.items()):
        raise ValueError('protected source hash changed')
    rows, excluded, repeats = curate(sources['relevance'], keep_variants=keep_variants)
    output.mkdir(parents=True)
    write_lines(output / 'relevance/retained.jsonl', rows)
    write_lines(output / 'review/excluded-repetitions.jsonl', excluded)
    write(output / 'review/repeated-groups.json', repeats)
    assert len(rows) + len(excluded) == len(sources['relevance'])
    assert load_lines(output / 'relevance/retained.jsonl') == rows
    if any(digest(Path(p)) != h for p, h in protected.items()):
        raise ValueError('source changed during curation')
    manifest = {'purpose': 'Relevance-only curation draft; source labels retained, not approved for training',
                'created_date': created_date(), 'status': 'review_pending', 'input_schema': INPUT_SCHEMA,
                'recent_windows_policy': RECENT_WINDOWS_POLICY, 'source_dir': str(source),
                'source_rows': len(sources['relevance']), 'retained_rows': len(rows),
                'excluded_rows': len(excluded), 'keep_wrapper_variants': keep_variants,
                'source_labels_changed': False, 'urgency_changed': False, 'source_files_unchanged': True,
                'protected_hashes': protected, 'profile': profile(rows),
                'files': {name: {'sha256': digest(output / name)} for name in (
                    'relevance/retained.jsonl', 'review/excluded-repetitions.jsonl', 'review/repeated-groups.json')},
                'limitations': ['No approximate template/topic matching was used to delete rows.',
                                'App/history-count bias is not fixed merely by removing wrappers.',
                                'Existing trained examples are train-only, never a new independent test.',
                                'Inherited labels still require targeted semantic review.']}
    write(output / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'source_rows', 'retained_rows', 'excluded_rows')}, ensure_ascii=False))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--keep-wrapper-variants', type=int, default=2)
    args = parser.parse_args()
    prepare(args.output_dir, args.source_dir, args.keep_wrapper_variants)
