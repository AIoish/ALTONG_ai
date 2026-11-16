"""용도: 관련도 개발 지표와 사용자 확정 모델 채택 조건을 분리하고 평가 자료 사용 이력을 기록한다.
생성일: 2026-11-15
"""

import argparse
from collections import Counter
from pathlib import Path

from filtering_training.common.artifact_io import created_date, digest, load_lines, read, write
from filtering_training.common.relevance_policy import POLICY
from filtering_training.modeling.relevance_only import evaluate
from filtering_training.preparation.prepare_relevance_experiment import input_key

CONTRACT = {'version': 'relevance-adoption-v1', 'priority': 'exact_score_accuracy',
            'maximum_accuracy_drop': .005, 'maximum_extra_relevant_misses': 0,
            'maximum_extra_false_relevant': 0, 'authority': 'user_confirmed_2026-11-15',
            'checkpoint_order': ['exact_accuracy', 'macro_f1', 'lower_mae']}


def metrics(rows, scores):
    if not rows or len(rows) != len(scores) or any(type(s) is not int or s not in (1, 2, 3) for s in scores):
        raise ValueError('complete relevance 1-3 predictions required')
    report = evaluate(rows, scores)
    cm = report['confusion_matrix']
    report.update(confusion_1_2=cm[0][1] + cm[1][0], relevant_misses=cm[2][0] + cm[2][1],
                  false_relevant=cm[0][2] + cm[1][2])
    predicted_relevant = sum(cm[i][2] for i in range(3))
    actual_relevant = sum(cm[2])
    report.update(relevant_precision=cm[2][2]/predicted_relevant if predicted_relevant else None,
                  relevant_recall=cm[2][2]/actual_relevant if actual_relevant else None)
    return report


def decide(baseline, candidate, contract=CONTRACT):
    if baseline['rows'] != candidate['rows'] or baseline['rows'] < 1:
        raise ValueError('same nonempty evaluation required')
    violations = []
    if baseline['exact_accuracy'] - candidate['exact_accuracy'] > contract['maximum_accuracy_drop'] + 1e-12:
        violations.append('accuracy_drop_exceeds_limit')
    for name, limit in [('relevant_misses', 'maximum_extra_relevant_misses'), ('false_relevant', 'maximum_extra_false_relevant')]:
        if candidate[name] - baseline[name] > contract[limit]:
            violations.append(name + '_increased')
    return {'development_gate': 'pass' if not violations else 'reject', 'violations': violations,
            'deployment_approved': False, 'final_real_evaluation': 'missing',
            'interpretation': 'Passing development constraints permits further evaluation; it does not establish real-use performance.'}


def development_report(rows, scores):
    """Report the same predictions by origin and current/history conflict, never training rows."""
    report = metrics(rows, scores)
    from filtering_training.preparation.curate_relevance import key
    report['evaluation_sha256'] = key([(r['id'], r['state'], r['score']) for r in rows])
    slices = {}
    for origin in sorted({r['experiment']['origin'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['experiment']['origin'] == origin]
        slices['origin:' + origin] = metrics([rows[i] for i in indices], [scores[i] for i in indices])
    indices = [i for i, r in enumerate(rows) if r.get('case_kind') == 'current_history_conflict']
    if indices:
        slices['case:current_history_conflict'] = metrics([rows[i] for i in indices], [scores[i] for i in indices])
    report['development_slices'] = slices
    return report


def development_decision(baseline, candidate):
    if baseline['evaluation_sha256'] != candidate['evaluation_sha256']:
        raise ValueError('same evaluation inputs and labels required')
    decision = decide(baseline, candidate)
    before, after = baseline['development_slices'], candidate['development_slices']
    if before.keys() != after.keys():
        raise ValueError('same development slices required')
    decision['slices'] = {name: decide(before[name], after[name]) for name in before}
    decision['slice_policy'] = 'Apply the confirmed accuracy/miss/false-relevant limits to each origin and conflict slice.'
    if any(d['development_gate'] != 'pass' for d in decision['slices'].values()):
        decision['development_gate'] = 'reject'
        decision['violations'].append('development_slice_regressed')
    return decision


def compare(dataset, baseline_run, candidate_run, output):
    from filtering_training.preparation.migrate_relevance_policy import rescore
    output = Path(output)
    if output.exists():
        raise ValueError('preserve prior comparison')
    output.mkdir(parents=True)
    rows = load_lines(Path(dataset) / 'relevance/validation.jsonl')
    reports = []
    for name, run in [('baseline', baseline_run), ('candidate', candidate_run)]:
        # Reuse input/cache/checkpoint verification; do not silently rescore mismatched inputs.
        rescore(dataset, run, output / (name + '-rescore.json'))
        predictions = {p['id']: p['predicted'] for p in load_lines(Path(run) / 'validation.predictions.jsonl')}
        reports.append(metrics(rows, [predictions[r['id']] for r in rows]))
    result = {'purpose': 'Development adoption gate, not deployment approval or final evaluation',
              'created_date': created_date(), 'contract': CONTRACT, 'baseline': reports[0], 'candidate': reports[1],
              'decision': decide(*reports), 'dataset_manifest_sha256': digest(Path(dataset) / 'manifest.json')}
    write(output / 'comparison.json', result)
    return result


def compare_real(rows_path, baseline_run, candidate_run, output, *, device='cpu'):
    from filtering_training.common.encoder_input import encode_checked, input_text
    from filtering_training.modeling.relevance_only import load_run, predictions
    rows_path, output = Path(rows_path), Path(output)
    if output.exists():
        raise ValueError('preserve existing real comparison')
    manifest_path = rows_path.parent / 'manifest.json'
    manifest = read(manifest_path)
    if manifest.get('policy') != POLICY:
        raise ValueError('expected current relevance evaluation policy')
    if manifest['files'].get(rows_path.name) != digest(rows_path):
        raise ValueError('reviewed evaluation file changed')
    rows = load_lines(rows_path)
    if not rows or any(r['split'] != 'real_development' or
                       r.get('evaluation_review', {}).get('human_reviewed') is not True or
                       r['evaluation_review']['input_sha256'] != input_key(r) for r in rows):
        raise ValueError('only explicitly reviewed real development rows may be used for model selection')
    output.mkdir(parents=True)
    reports, checkpoints = [], {}
    for name, run in [('baseline', baseline_run), ('candidate', candidate_run)]:
        model, tokenizer, config = load_run(run, device=device)
        encoded, _ = encode_checked(tokenizer, [input_text(r['state'], 'relevance') for r in rows])
        scores, _ = predictions(model, tokenizer, encoded, device=device, batch_size=4)
        reports.append(metrics(rows, scores))
        write(output / (name + '-predictions.json'), [{'id':r['id'], 'predicted':s} for r,s in zip(rows,scores)])
        checkpoints[name] = config['checkpoint_sha256']
        del model
    if digest(rows_path) != manifest['files'][rows_path.name]:
        raise ValueError('review data changed during evaluation')
    result = {'purpose': 'Human-reviewed real development evaluation, not final test', 'created_date': created_date(),
              'contract': CONTRACT, 'baseline': reports[0], 'candidate': reports[1], 'decision': decide(*reports),
              'checkpoints':checkpoints, 'rows_sha256':digest(rows_path), 'manifest_sha256':digest(manifest_path)}
    write(output / 'comparison.json', result)
    return result


def inventory(dataset, real_capture, output, *, additional_known=()):
    output = Path(output)
    if output.exists():
        raise ValueError('preserve previous inventory')
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for split in ('train', 'validation'):
        path = Path(dataset) / f'relevance/{split}.jsonl'
        rows = load_lines(path)
        entries.append({'path': str(path.resolve()), 'sha256': digest(path), 'rows': len(rows),
            'role': 'training' if split == 'train' else 'development_validation',
            'origins': dict(Counter(r['experiment']['origin'] for r in rows)),
            'label_sources': dict(Counter(r['experiment']['label_source'] for r in rows)),
            'final_eligible': False, 'reason': 'Already trained on or repeatedly inspected for development.'})
    if real_capture:
        path = Path(real_capture)
        entries.append({'path': str(path.resolve()), 'sha256': digest(path), 'rows': len(load_lines(path)),
                        'role': 'previously_inspected_capture', 'final_eligible': False,
                        'reason': 'Existing export was inspected; captured rows already supplied training candidates.'})
    for path in map(Path, additional_known):
        entries.append({'path': str(path.resolve()), 'sha256': digest(path), 'rows': len(load_lines(path)),
                        'role': 'previously_inspected_candidates', 'final_eligible': False,
                        'reason': 'Additional authored/training-side candidates already used for development.'})
    # Template overlap is a diagnostic, not proof of exact-row leakage.
    from filtering_training.quality.audit_relevance_diversity import authored_templates
    from filtering_training.preparation.curate_relevance import core_body
    templates = authored_templates()
    matches = {}
    for split in ('train', 'validation'):
        matches[split] = [next((t for t, pattern in templates if pattern.fullmatch(core_body(r['state']['notification']['body']))), None)
                          for r in load_lines(Path(dataset) / f'relevance/{split}.jsonl')]
    seen = set(matches['train']) - {None}
    result = {'purpose': 'Evaluation source usage inventory; no new real labels or final-test claims',
              'created_date': created_date(), 'entries': entries, 'contract': CONTRACT,
              'validation_template_matches_seen_in_training': sum(t in seen for t in matches['validation']),
              'real_development_evaluation': 'missing_new_capture', 'real_final_evaluation': 'missing_new_capture',
              'diagnostics': ['authored context/reaction probes include training-side captures; not independent final tests'],
              'template_overlap_is_not_exact_row_leakage': True}
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--real-rows', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--candidate-run', type=Path)
    parser.add_argument('--real-capture', type=Path)
    parser.add_argument('--additional-known', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if args.real_rows and args.baseline_run and args.candidate_run:
        print(compare_real(args.real_rows, args.baseline_run, args.candidate_run, args.output, device=args.device)['decision'])
    elif args.dataset and args.baseline_run and args.candidate_run:
        print(compare(args.dataset, args.baseline_run, args.candidate_run, args.output)['decision'])
    elif args.dataset and not args.baseline_run and not args.candidate_run:
        inventory(args.dataset, args.real_capture, args.output, additional_known=args.additional_known)
    else:
        parser.error('both runs are required for comparison')
