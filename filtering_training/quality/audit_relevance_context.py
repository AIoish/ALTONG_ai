"""용도: 고정 관련도 검증 입력의 알림 앱 변경·최근 창 제거에 따른 예측 민감도를 비교한다.
생성일: 2026-11-13
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from filtering_training.common.artifact_io import created_date, digest, load_lines, write
from filtering_training.common.encoder_input import encode_checked, input_text
from filtering_training.modeling.relevance_only import evaluate, load_run, predictions
from filtering_training.preparation.curate_relevance import key, write_lines
from filtering_training.quality.audit_relevance_reactions import APPS


def make_variants(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty rows with unique IDs required')
    variants = []
    for row in rows:
        state = row['state']
        original = input_text(state, 'relevance')
        candidates = [('original', deepcopy(state))]
        for app in APPS:
            changed = deepcopy(state)
            changed['notification']['app_name'] = app
            if input_text(changed, 'relevance') != original:
                candidates.append(('app:' + app, changed))
        changed = deepcopy(state)
        changed['context']['recent_windows'] = []
        if input_text(changed, 'relevance') != original:
            candidates.append(('without_recent', changed))
        for kind, changed in candidates:
            variants.append({'source_id': row['id'], 'variant': kind, 'state': changed,
                             'input_sha256': key(json.loads(input_text(changed, 'relevance')))})
    return variants


def summarize(rows, variants, scores):
    if len(variants) != len(scores) or any(type(s) is not int or s not in (1, 2, 3) for s in scores):
        raise ValueError('one score in 1-3 per variant required')
    grouped = defaultdict(dict)
    for variant, score in zip(variants, scores):
        group = grouped[variant['source_id']]
        if variant['variant'] in group:
            raise ValueError('duplicate variant')
        group[variant['variant']] = score
    if set(grouped) != {r['id'] for r in rows} or any('original' not in g for g in grouped.values()):
        raise ValueError('complete original coverage required')
    details = []
    for row in rows:
        group = grouped[row['id']]
        original = group['original']
        apps = [original] + [s for k, s in group.items() if k.startswith('app:')]
        details.append({'id': row['id'], 'origin': row['experiment']['origin'],
                        'case_kind': row.get('case_kind'), 'original_reference': row['score'],
                        'original_correct': original == row['score'], 'scores': group,
                        'app_changed_score': len(set(apps)) > 1,
                        'history_changed_score': 'without_recent' in group and group['without_recent'] != original})
    report = {'rows': len(rows), 'variants': len(variants),
              'original_validation': evaluate(rows, [grouped[r['id']]['original'] for r in rows]),
              'app_changed_rows': sum(d['app_changed_score'] for d in details),
              'history_eligible_rows': sum('without_recent' in d['scores'] for d in details),
              'history_changed_rows': sum(d['history_changed_score'] for d in details),
              'interpretation': 'Only original inputs are scored against labels. Changed scores indicate sensitivity, not automatically errors or internal attention weights.'}
    report['by_case_kind'] = {}
    for kind in sorted({d['case_kind'] for d in details if d['case_kind']}):
        selected = [d for d in details if d['case_kind'] == kind]
        report['by_case_kind'][kind] = {'rows': len(selected),
            'original_correct': sum(d['original_correct'] for d in selected),
            'app_changed_rows': sum(d['app_changed_score'] for d in selected),
            'history_changed_rows': sum(d['history_changed_score'] for d in selected)}
    return report, details


def audit(rows_path, runs, output, *, device='cuda'):
    rows_path, output = Path(rows_path), Path(output)
    if output.exists():
        raise ValueError('preserve earlier audit; choose a new folder')
    source_hash = digest(rows_path)
    rows = load_lines(rows_path)
    if any(r.get('split') != 'validation' for r in rows):
        raise ValueError('this diagnostic requires validation rows')
    variants = make_variants(rows)
    output.mkdir(parents=True)
    write_lines(output / 'variants.jsonl', variants)
    reports, models = {}, {}
    for number, run in enumerate(runs, 1):
        model, tokenizer, config = load_run(run, device=device)
        encoded, lengths = encode_checked(tokenizer, [input_text(v['state'], 'relevance') for v in variants])
        scores, probabilities = predictions(model, tokenizer, encoded, device=device, batch_size=8)
        report, details = summarize(rows, variants, scores)
        name = f'model-{number}'
        report['maximum_tokens'] = max(lengths)
        write(output / f'{name}.summary.json', report)
        write_lines(output / f'{name}.cases.jsonl', details)
        write_lines(output / f'{name}.predictions.jsonl', [dict(v, predicted_score=s, probabilities_1_2_3=p)
                    for v, s, p in zip(variants, scores, probabilities)])
        reports[name] = report
        models[name] = {'run': str(Path(run).resolve()), 'checkpoint_sha256': config['checkpoint_sha256']}
        print(json.dumps({'model': name, 'original_correct': sum(d['original_correct'] for d in details),
                          'app_changed_rows': report['app_changed_rows'], 'history_changed_rows': report['history_changed_rows']}, ensure_ascii=False), flush=True)
        del model
    if digest(rows_path) != source_hash:
        raise ValueError('validation source changed during audit')
    write(output / 'manifest.json', {'purpose': 'Fixed-validation sensitivity diagnostic; no transformed labels or training',
        'created_date': created_date(), 'source_sha256': source_hash, 'models': models,
        'rows': len(rows), 'variants': len(variants), 'source': str(rows_path.resolve()),
        'code_sha256': digest(__file__), 'files': {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        'limitations': ['Development validation, not an independent final test.',
                        'Removing history can remove necessary evidence and change the correct answer.',
                        'App sensitivity is diagnostic; changed inputs have no propagated gold labels.']})
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--run', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    audit(args.rows, args.run, args.output_dir, device=args.device)
