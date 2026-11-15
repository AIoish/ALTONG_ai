"""용도: 기존 분할·입력을 보존해 관련도 라벨 정책을 이관하고 저장 예측을 새 정답으로 재평가한다.
생성일: 2026-11-15
"""

import argparse
from collections import Counter
from pathlib import Path
import shutil

from filtering_training.common.artifact_io import created_date, digest, load_lines, read, write
from filtering_training.common.relevance_policy import POLICY, apply_policy
from filtering_training.modeling.relevance_only import evaluate
from filtering_training.preparation.curate_relevance import write_lines
from filtering_training.preparation.prepare_relevance_experiment import audit_rows, input_key


def migrate(source, output):
    from filtering_training.quality.verify_relevance_experiment import verify
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('preserve existing migration; choose a new output')
    verify(source, write_report=False)
    manifest = read(source / 'manifest.json')
    protected = dict(manifest['protected_sources'])
    for path in [source / 'manifest.json', *[source / n for n in manifest['files']]]:
        protected[str(path)] = digest(path)
    old = [r for split in ('train', 'validation') for r in load_lines(source / f'relevance/{split}.jsonl')]
    rows = apply_policy(old)
    output.mkdir(parents=True)
    (output / 'relevance').mkdir()
    shutil.copytree(source / 'review', output / 'review', ignore=shutil.ignore_patterns('readiness-verification.json'))
    changes = [dict(id=a['id'], split=a['split'], input_sha256=input_key(a),
                    previous_score=a['score'], score=b['score'], policy=POLICY)
               for a, b in zip(old, rows) if a['score'] != b['score']]
    for split in ('train', 'validation'):
        write_lines(output / f'relevance/{split}.jsonl', [r for r in rows if r['split'] == split])
    write_lines(output / 'review/policy-changes.jsonl', changes)
    write(output / 'review/consistency.json', audit_rows(rows))
    write(output / 'review/policy.json', {'version': POLICY, 'created_date': created_date(),
          'authority': 'User confirmed: targetless reactions score 1 even with approval title.',
          'model_input_normalization': False, 'inference_score_override': False,
          'supersedes': ['project-room targetless score 2', 'approval-title yes/no score 2'],
          'changed_rows': len(changes), 'changed_by_split': dict(Counter(c['split'] for c in changes)),
          'limitations': ['Finite explicit reaction vocabulary; meaningful short messages and uncertain expressions are not blanket relabeled.']})
    new = dict(manifest, created_date=created_date(), label_policy=POLICY,
               policy_migration={'source': str(source), 'version': POLICY},
               protected_sources=protected, frozen_reference=str(source), frozen_reference_rows=len(old))
    new['files'] = {str(p.relative_to(output)): {'sha256': digest(p)} for p in output.rglob('*') if p.is_file()}
    new['splits'] = {s: dict(manifest['splits'][s], scores=dict(Counter(r['score'] for r in rows if r['split'] == s))) for s in ('train', 'validation')}
    write(output / 'manifest.json', new)
    verify(output)
    return new


def rescore(dataset, run, output):
    from filtering_training.quality.verify_relevance_experiment import verify
    dataset, run, output = Path(dataset), Path(run), Path(output)
    if output.exists():
        raise ValueError('preserve prior evaluation')
    verify(dataset, write_report=False)
    config = read(run / 'run_config.json')
    if config['status'] != 'completed' or config['output_classes'] != {'relevance': 3}:
        raise ValueError('expected completed relevance-only run')
    verify(Path(config['dataset_dir']), write_report=False)
    rows = load_lines(dataset / 'relevance/validation.jsonl')
    originals = {r['id']: r for r in load_lines(Path(config['dataset_dir']) / 'relevance/validation.jsonl')}
    cached = load_lines(run / 'validation.predictions.jsonl')
    predictions = {r['id']: r for r in cached}
    if len(predictions) != len(cached) or set(predictions) != {r['id'] for r in rows}:
        raise ValueError('cached prediction IDs mismatch')
    for row in rows:
        old = originals.get(row['id'])
        if old is None or input_key(row) != input_key(old) or predictions[row['id']]['gold'] != old['score']:
            raise ValueError('cached predictions not tied to unchanged original inputs')
    if digest(run / 'model.pt') != config['checkpoint_sha256']:
        raise ValueError('checkpoint hash mismatch')
    report = evaluate(rows, [predictions[r['id']]['predicted'] for r in rows])
    write(output, {'purpose': 'Rescore unchanged cached predictions against new policy; no training or fresh inference',
          'created_date': created_date(), 'policy': read(dataset / 'manifest.json')['label_policy'],
          'dataset_manifest_sha256': digest(dataset / 'manifest.json'), 'predictions_sha256': digest(run / 'validation.predictions.jsonl'),
          'run_config_sha256': digest(run / 'run_config.json'), 'checkpoint_sha256': config['checkpoint_sha256'],
          'previous_report': read(run / 'validation.json'), 'new_report': report})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', nargs='*', type=Path)
    args = parser.parse_args()
    migrate(args.source, args.output)
    for run in args.run or []:
        report = rescore(args.output, run, args.output / ('evaluation-' + run.name + '.json'))
        print(run.name, report['exact_correct'], '/', report['rows'])
