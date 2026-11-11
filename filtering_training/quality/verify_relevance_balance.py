"""용도: 관련도 구성 개선의 모든 입력 변경·근거 보존·표현 범위와 메타데이터 지름길을 독립 점검한다.
생성일: 2026-11-08
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from tokenizers import Tokenizer

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import input_text
from filtering_training.preparation.balance_relevance import OUTPUT
from filtering_training.preparation.curate_relevance import key
from filtering_training.review.review_relevance_dataset import Reviewer


def probe(records, fields):
    features = []
    for row in records:
        n = row['state']['notification']
        item = {name: n.get(name, '') for name in fields if name != 'history_count'}
        if 'history_count' in fields:
            item['history_count'] = str(len(row['state']['context']['recent_windows']))
        features.append(item)
    encoded = DictVectorizer().fit_transform(features)
    labels = [r['score'] for r in records]
    predictions = cross_val_predict(DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42),
        encoded, labels, groups=[r['group_id'] for r in records], cv=GroupKFold(5))
    return {'features': list(fields), 'accuracy': float(accuracy_score(labels, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, predictions))}


def metrics(records):
    counts = Counter(r['score'] for r in records)
    return {'rows': len(records), 'body_groups': len({r['group_id'] for r in records}),
            'majority_accuracy': max(counts.values()) / len(records),
            'probes': [probe(records, fields) for fields in (
                ('app_name',), ('history_count',), ('app_name', 'history_count'),
                ('app_name', 'history_count', 'sender', 'title', 'timestamp'))]}


def verify(folder=OUTPUT):
    folder = Path(folder).resolve()
    destination = folder / 'review/verification.json'
    if destination.exists():
        raise ValueError('existing verification is preserved; use a new dataset version')
    manifest = read(folder / 'manifest.json')
    protected = dict(manifest['protected_hashes'])
    protected[str(folder / 'manifest.json')] = digest(folder / 'manifest.json')
    protected.update({str(folder / name): info['sha256'] for name, info in manifest['files'].items()})
    assert all(digest(Path(path)) == h for path, h in protected.items())
    source_rows = load_lines(manifest['source'])
    sources = {r['id']: r for r in source_rows}
    rows = load_lines(folder / 'relevance/dataset.jsonl')
    excluded = load_lines(folder / 'review/excluded.jsonl')
    lineage = load_lines(folder / 'review/lineage.jsonl')
    assert len(rows) == len(lineage)
    ids = {r['augmentation']['source_id'] for r in rows}
    assert len(ids) == len(rows)
    assert not ids.intersection(r['id'] for r in excluded)
    assert ids | {r['id'] for r in excluded} == set(sources)
    assert len(rows) + len(excluded) == len(source_rows)
    reviewer = Reviewer()
    payloads = set()
    for row, entry in zip(rows, lineage):
        original = sources[entry['source_id']]
        assert row['augmentation'] == entry
        assert row['score'] == original['score']
        assert row['group_id'] == original['group_id']
        assert row['split'] == 'train_only'
        assert not row['training_eligible']
        # Only app and history may be authored; notification content/current state must stay identical.
        state = deepcopy(row['state'])
        state['notification']['app_name'] = original['state']['notification']['app_name']
        state['context']['recent_windows'] = original['state']['context']['recent_windows']
        assert state == original['state']
        if entry['fixed_case_preserved']:
            assert row['state'] == original['state']
        evidence = entry['history_evidence']
        if evidence:
            required = [original['state']['context']['recent_windows'][i]
                        for i in evidence['required_original_positions']]
            positions = [row['state']['context']['recent_windows'].index(w) for w in required]
            assert positions == sorted(positions)
        test_row = deepcopy(row)
        test_row['id'], test_row['review'] = original['id'], original['review']
        assert reviewer.decide(test_row)[0] == row['score']
        payload = json.loads(input_text(row['state'], 'relevance'))
        assert payload['recent_windows'] == row['state']['context']['recent_windows']
        assert len(payload['recent_windows']) <= 3
        signature = key(payload)
        assert signature == entry['output_input_sha256']
        assert signature not in payloads
        payloads.add(signature)
    before_tasks = set().union(*(reviewer.targets(r['state']['context']['window_title']) for r in source_rows))
    after_tasks = set().union(*(reviewer.targets(r['state']['context']['window_title']) for r in rows))
    before_templates = {reviewer.matching_template(r['state']['notification']['body']) for r in source_rows}
    after_templates = {reviewer.matching_template(r['state']['notification']['body']) for r in rows}
    assert before_tasks == after_tasks
    assert before_templates == after_templates
    # A donor window never changes the notification body, so body-level evaluation overlap cannot increase.
    assert {r['state']['notification']['body'] for r in rows}.issubset(
        {r['state']['notification']['body'] for r in source_rows})
    tokenizer = Tokenizer.from_file(str(ROOT / 'filtering_training/outputs/models/klue-roberta-base/tokenizer.json'))
    lengths = [len(e.ids) for e in tokenizer.encode_batch([input_text(r['state'], 'relevance') for r in rows])]
    assert max(lengths) <= 512
    same_subset = [sources[r['augmentation']['source_id']] for r in rows]
    report = {'purpose': 'Input-preservation and metadata-shortcut audit; not a KLUE model evaluation',
              'created_date': created_date(), 'protected_sources_unchanged': True,
              'rows': len(rows), 'excluded_rows': len(excluded), 'task_coverage': len(after_tasks),
              'max_tokens': max(lengths), 'full_source': metrics(source_rows),
              'same_selected_rows_before_transform': metrics(same_subset), 'after_transform': metrics(rows),
              'method': 'Five-fold GroupKFold by stored body group, depth-four decision tree, min leaf ten',
              'limitations': ['Same-subset comparison holds examples and group folds fixed; it measures only selected metadata predictability.',
                              'Low app/history-count predictability does not prove absence of semantic or other synthetic shortcuts.',
                              'No real-user dataset or independent final test was added.'], 'training_ready': False}
    assert all(digest(Path(path)) == h for path, h in protected.items())
    write(destination, report)
    write_examples(folder / 'review/examples.md', rows, sources)
    print(json.dumps({'rows': len(rows), 'max_tokens': max(lengths),
                      'before': report['same_selected_rows_before_transform'], 'after': report['after_transform']},
                     ensure_ascii=False, indent=2))
    return report


def write_examples(path, rows, sources):
    chosen = []
    conditions = (
        lambda r: r['augmentation']['fixed_case_preserved'],
        lambda r: r['augmentation']['app_changed'] and r['score'] == 1,
        lambda r: r['augmentation']['app_changed'] and r['score'] == 2,
        lambda r: r['augmentation']['app_changed'] and r['score'] == 3,
        lambda r: r['augmentation']['history_evidence'] and r['augmentation']['history_evidence']['required_original_positions'],
        lambda r: r['augmentation']['history_evidence'] and r['augmentation']['history_evidence']['new_count'] > r['augmentation']['history_evidence']['original_count'],
        lambda r: r['augmentation']['history_changed'] and not r['state']['context']['recent_windows'],
    )
    for condition in conditions:
        candidate = next((r for r in rows if r not in chosen and condition(r)), None)
        if candidate:
            chosen.append(candidate)
    lines = ['<!-- 용도: 관련도 합성 구성 개선의 실제 변경 사례와 보존 근거를 확인한다.',
             f'생성일: {created_date()} -->', '', '# 구성 개선 사례', '',
             '실제 수집된 과거 창을 복원한 기록이 아니라 명시적으로 만든 합성 변형이다.', '']
    for row in chosen:
        original = sources[row['augmentation']['source_id']]
        a, b = original['state'], row['state']
        lines.extend([f"## {row['id']} · {row['score']}점 유지", '',
                      f"- 본문: {b['notification']['body']}",
                      f"- 현재 창: {b['context']['active_process']} / {b['context']['window_title'] or '(없음)'}",
                      f"- 알림 앱: {a['notification']['app_name']} → {b['notification']['app_name']}"])
        for title, state in (('기존 최근 창', a), ('합성 최근 창', b)):
            lines.append('- ' + title + ': ' + (' → '.join(w['window_title'] for w in state['context']['recent_windows']) or '(없음)'))
        lines.extend([f"- 필수 원본 창 위치: {(row['augmentation']['history_evidence'] or {}).get('required_original_positions', '고정 사례')}", ''])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    verify(parser.parse_args().folder)
