"""용도: 현재 창을 제외한 직전 창 최대3개로 학습·검증·평가를 통일한 새 데이터 버전을 만든다.
생성일: 2026-11-04
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_data import DEFAULT_DATASET
from filtering_training.common.encoder_input import (
    INPUT_SCHEMA, WINDOWS_INPUT_SCHEMA, LEGACY_INPUT_SCHEMA, RECENT_WINDOWS_POLICY,
    input_text, windows_only_state,
)
from filtering_training.common.validation_data import EVALUATION_SOURCES, SOURCE, VALIDATION_POOL, evaluation_sets

SOURCE_DATASET = ROOT / 'filtering_training/outputs/v12_windows_only_relevance_01'
TRAINING_FILES = ('relevance/train.jsonl', 'mapped-training/initial.jsonl', 'mapped-training/safety.jsonl')


def transform_record(row):
    task = row.get('task', 'relevance')
    if task not in ('urgency', 'relevance'):
        raise ValueError('unknown training task')
    result = deepcopy(row)
    if task == 'relevance':
        result['state'] = windows_only_state(row['state'])
    if 'input_text' in row:
        if row['input_text'] != input_text(row['state'], task, schema=WINDOWS_INPUT_SCHEMA):
            raise ValueError('source input_text does not match the v12 contract')
        result['input_text'] = input_text(result['state'], task)
    return result


def prepare(output=DEFAULT_DATASET, source_dir=SOURCE_DATASET):
    output, source_dir = Path(output).resolve(), Path(source_dir).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; existing data is preserved')
    parent_manifest = read(source_dir / 'manifest.json')
    if parent_manifest.get('removed_field') != 'recent_processes':
        raise ValueError('expected v12 source with recent_processes removed')
    protected = {str(source_dir / 'manifest.json'): digest(source_dir / 'manifest.json')}
    converted, info = {}, {}
    for name in TRAINING_FILES:
        source = source_dir / name
        expected = parent_manifest['files'][name]
        if digest(source) != expected['sha256']:
            raise ValueError(f'source dataset hash changed: {name}')
        rows = load_lines(source)
        if len(rows) != expected['rows']:
            raise ValueError(f'source dataset row count changed: {name}')
        result = [transform_record(row) for row in rows]
        for before, after in zip(rows, result):
            # Reinsert the entire original context/input_text and require every other field to match.
            restored = deepcopy(after)
            if 'context' in before['state']:
                restored['state']['context'] = before['state']['context']
            if 'input_text' in before:
                restored['input_text'] = before['input_text']
            if restored != before:
                raise ValueError('unexpected change to labels, notifications or training metadata')
            if before.get('task') == 'urgency' and before != after:
                raise ValueError('urgency rows must stay unchanged')
        converted[name] = result
        protected[str(source)] = expected['sha256']
        changed = sum(before != after for before, after in zip(rows, result))
        info[name] = {'source': str(source.relative_to(ROOT)), 'source_sha256': expected['sha256'],
                      'rows': len(rows), 'changed_rows': changed,
                      'task_counts': dict(Counter(row.get('task', 'relevance') for row in rows)),
                      'labels_ids_order_and_other_fields_preserved': True}

    # Approval checks run while reading the original validation sources.
    frozen_evals = evaluation_sets(schema=LEGACY_INPUT_SCHEMA)
    for name, rows in frozen_evals.items():
        filename = f'evaluation/{name}.jsonl'
        converted[filename] = [windows_only_state(row) for row in rows]
        info[filename] = {'rows': len(rows), 'changed_rows': sum(a != b for a, b in zip(rows, converted[filename])),
                          'source': 'approved validation sources' if name == 'validation' else str(EVALUATION_SOURCES[name].relative_to(ROOT)),
                          'labels_ids_order_and_other_fields_preserved': True}
    donor_paths = [SOURCE / 'dataset.jsonl', SOURCE / 'prepared/manifest.json',
                   VALIDATION_POOL / 'manifest.json', VALIDATION_POOL / 'approval.json', VALIDATION_POOL / 'dataset.jsonl',
                   *EVALUATION_SOURCES.values()]
    subset = read(VALIDATION_POOL / 'manifest.json')['evaluation_set']
    for seed in subset['seed_sources']:
        seed_folder = Path(seed['folder'])
        if not seed_folder.is_absolute():
            seed_folder = ROOT / seed_folder
        donor_paths.extend([seed_folder / 'dataset.jsonl', seed_folder / 'approval.json'])
    protected.update({str(path): digest(path) for path in donor_paths})
    output.mkdir(parents=True)
    for name, rows in converted.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8'))
        if load_lines(target) != rows:
            raise ValueError(f'saved rows changed: {name}')
        info[name]['sha256'] = digest(target)
    if any(digest(Path(path)) != value for path, value in protected.items()):
        raise ValueError('source files changed during conversion')
    manifest = {'purpose': 'App-aligned previous-window training and evaluation inputs; current window excluded',
                'created_date': created_date(), 'status': 'prepared_not_trained', 'input_schema': INPUT_SCHEMA,
                'recent_windows_policy': RECENT_WINDOWS_POLICY, 'removed_field': 'recent_processes',
                'label_mapping_changed': False, 'urgency_rows_unchanged': True, 'source_files_unchanged': True,
                'source_dataset': str(source_dir.relative_to(ROOT)), 'protected_sources': protected, 'files': info,
                'limitations': ['No missing fourth historical window was invented.',
                                'Frozen source labels are retained; quality improvement requires training and evaluation.',
                                'Current app integration and model deployment were not changed.']}
    write(output / 'manifest.json', manifest)
    print(json.dumps({'output': str(output), 'input_schema': INPUT_SCHEMA,
                      'files': {name: {'rows': value['rows'], 'changed_rows': value['changed_rows']} for name, value in info.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DATASET)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    prepare(args.output_dir, args.source_dir)
