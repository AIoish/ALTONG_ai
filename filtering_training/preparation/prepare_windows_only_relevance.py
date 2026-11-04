"""용도: 관련도의 recent_processes만 제거한 새 학습 데이터 버전을 만든다.
생성일: 2026-11-04
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / 'filtering_training/outputs/runs/2026-11-03-klue-joint-u5-r3-02'
NATIVE = ROOT / 'filtering_training/outputs/v7_relevance_ladder_01/prepared_laya_v2/relevance/train.jsonl'
OUTPUT = ROOT / 'filtering_training/outputs/v12_windows_only_relevance_01'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(row):
    result = deepcopy(row)
    relevance = row.get('task') == 'relevance' or set(row.get('expected', {})) == {'relevance'}
    if not relevance:
        return result
    # The redundant field can occur both in state and in serialized model input.
    del result['state']['context']['recent_processes']
    if 'input_text' in result:
        payload = json.loads(result['input_text'])
        del payload['recent_processes']
        result['input_text'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    return result


def verify(before, after):
    """Restore only the removed keys and require all other values to match."""
    relevance = before.get('task') == 'relevance' or set(before.get('expected', {})) == {'relevance'}
    restored = deepcopy(after)
    if relevance:
        assert 'recent_processes' not in after['state']['context']
        restored['state']['context']['recent_processes'] = before['state']['context']['recent_processes']
        if 'input_text' in before:
            original_payload = json.loads(before['input_text'])
            payload = json.loads(after['input_text'])
            assert 'recent_processes' not in payload
            payload['recent_processes'] = original_payload['recent_processes']
            assert payload == original_payload
            restored['input_text'] = before['input_text']
    assert restored == before, 'Unexpected change beyond recent_processes'


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Choose a new output directory; existing data is preserved')
    sources = {
        'relevance/train.jsonl': NATIVE,
        'mapped-training/initial.jsonl': SOURCE_RUN / 'mapped-training/initial.jsonl',
        'mapped-training/safety.jsonl': SOURCE_RUN / 'mapped-training/safety.jsonl',
    }
    source_hashes = {name: digest(path) for name, path in sources.items()}
    prepared = {}
    reports = {}
    for name, path in sources.items():
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        converted = [transform(row) for row in rows]
        for before, after in zip(rows, converted):
            verify(before, after)
        counts = Counter(row.get('task', 'relevance') for row in rows)
        assert counts['relevance'] > 0
        prepared[name] = converted
        reports[name] = {
            'source': str(path.relative_to(ROOT)),
            'source_sha256': source_hashes[name],
            'rows': len(rows),
            'task_counts': dict(counts),
            'changed_rows': counts['relevance'],
            'original_labels_order_and_other_fields_preserved': True,
        }
    # An interrupted conversion can resume only when all existing outputs match.
    serialized = {name: ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
                  for name, rows in prepared.items()}
    if output.exists():
        for target in output.rglob('*'):
            if target.is_file():
                name = target.relative_to(output).as_posix()
                if name not in serialized or target.read_text(encoding='utf-8') != serialized[name]:
                    raise ValueError(f'Preserve unexpected existing output: {target}')
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in prepared.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(serialized[name].encode('utf-8'))
        # Verify the persisted JSONL too, not only the in-memory conversion.
        saved = [json.loads(line) for line in target.read_text(encoding='utf-8').splitlines()]
        assert saved == rows
        reports[name]['sha256'] = digest(target)
    assert all(digest(path) == source_hashes[name] for name, path in sources.items())
    manifest = {
        'purpose': 'Dataset-only removal of redundant recent_processes from relevance state and input_text',
        'created_date': datetime.now(timezone(timedelta(hours=9), 'Asia/Seoul')).date().isoformat(),
        'status': 'prepared_not_trained',
        'removed_field': 'recent_processes',
        'label_mapping_changed': False,
        'urgency_rows_unchanged': True,
        'source_files_unchanged': True,
        'files': reports,
        'limitations': [
            'Training and application input builders still use the old schema; integration is pending.',
            'Validation and evaluation data have not been migrated in this dataset-only step.',
            'Native relevance/train.jsonl retains original five-level questions and labels; mapped-training retains the existing three-level mapping.',
        ],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'files': reports, 'status': manifest['status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    prepare(parser.parse_args().output_dir.resolve())
