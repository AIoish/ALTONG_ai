"""용도: 생성형 모델·프롬프트와 독립적으로 5단계 점수와 통과 정책을 평가한다.
생성일: 2026-11-04
"""

from collections import defaultdict
from src.filtering.policy import should_pass


def score_metrics(samples, predictions):
    if not samples or len(samples) != len(predictions):
        raise ValueError('predictions must cover all samples')
    fields = ('urgency_score', 'relevance_score')
    for prediction in predictions:
        if prediction is not None and (not isinstance(prediction, dict) or set(prediction) != set(fields)
                or any(type(value) is not int or not 1 <= value <= 5 for value in prediction.values())):
            raise ValueError('scores must be integers from 1 to 5 with exactly two fields')
    valid = [(sample, pred) for sample, pred in zip(samples, predictions) if pred is not None]
    urgent = blocked = failed = invalid_urgent = false_pass = correct = gold_block = 0
    groups = defaultdict(list)
    by_context = defaultdict(lambda: {'urgent': 0, 'urgent_failures': 0})
    for sample, pred in zip(samples, predictions):
        expected = should_pass(sample.label.urgency_score, sample.label.relevance_score)
        decision = should_pass(**pred) if pred is not None else None
        correct += decision == expected
        gold_block += not expected
        false_pass += not expected and decision is True
        is_urgent = sample.label.urgency_score >= 4
        urgent += is_urgent
        blocked += is_urgent and decision is False
        failed += is_urgent and decision is not True
        invalid_urgent += is_urgent and pred is None
        role = sample.notification.id.rsplit('_', 1)[-1]
        by_context[role]['urgent'] += is_urgent
        by_context[role]['urgent_failures'] += is_urgent and decision is not True
        groups[sample.notification.id.rsplit('_', 1)[0]].append(pred)
    invariance = {}
    for group, items in groups.items():
        values = [pred['urgency_score'] for pred in items if pred is not None]
        invariance[group] = {'count': len(items), 'valid': len(values),
                             'range': max(values) - min(values) if len(values) == len(items) else None}
    metrics = {'count': len(samples), 'valid_outputs': len(valid),
               'policy_correct': correct, 'policy_accuracy': correct / len(samples),
               'urgent_count': urgent, 'urgent_false_blocks': blocked,
               'urgent_invalid': invalid_urgent, 'urgent_failures': failed,
               'gold_block_count': gold_block, 'false_passes': false_pass,
               'by_context': dict(by_context), 'urgency_invariance': invariance,
               'fully_invariant_groups': sum(v['count'] > 1 and v['range'] == 0 for v in invariance.values())}
    for field in fields:
        errors = [abs(getattr(sample.label, field) - pred[field]) for sample, pred in valid]
        metrics[field] = {'exact_accuracy': sum(error == 0 for error in errors) / len(samples),
                          'within_one_accuracy': sum(error <= 1 for error in errors) / len(samples),
                          'mae_on_valid': sum(errors) / len(errors) if errors else None}
    return metrics
