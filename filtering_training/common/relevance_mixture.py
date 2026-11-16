"""용도: 동결 기준 자료의 학습 비중을 보존하고 신규 대비의 총비중을 시나리오별로 배분한다.
생성일: 2026-11-15
"""

from collections import Counter, defaultdict
import math

from filtering_training.preparation.curate_relevance import key


def mixture_weights(rows, reference, supplement_share, *, base_shares=None, scenario_field='experiment_group'):
    if not math.isfinite(supplement_share) or not 0 <= supplement_share < 1:
        raise ValueError('supplement share must be finite in [0, 1)')
    by_id = {r['id']: r for r in reference}
    current = {r['id']: r for r in rows}
    if not rows or not reference or len(by_id) != len(reference) or len(current) != len(rows) or not by_id.keys() <= current.keys():
        raise ValueError('unique IDs and all frozen reference rows required')
    def source(row):
        return row['experiment']['origin']
    def origin(row):
        return row['experiment'].get('loss_origin', source(row))
    counts = Counter(origin(r) for r in reference)
    # Freeze the legacy source-balanced per-row mass; reviewed remains separately reported.
    buckets = defaultdict(list)
    bucket_mass = defaultdict(float)
    source_mass = defaultdict(float)
    for old in reference:
        row = current[old['id']]
        if row['state'] != old['state'] or row['score'] != old['score'] or row['experiment_group'] != old['experiment_group'] or source(row) != source(old):
            raise ValueError('frozen reference input, score, group or source changed')
        mass = 1 / (len(counts) * counts[origin(old)])
        bucket = (source(old), old['experiment_group'])
        bucket_mass[bucket] += mass
        source_mass[source(old)] += mass
        buckets[bucket].append(old['id'])
    if base_shares is not None:
        if set(base_shares) != set(source_mass) or any(not math.isfinite(v) or v <= 0 for v in base_shares.values()) or not math.isclose(sum(base_shares.values()), 1):
            raise ValueError('base shares must explicitly cover every source, be positive and sum to one')
        for bucket in bucket_mass:
            bucket_mass[bucket] *= base_shares[bucket[0]] / source_mass[bucket[0]]
    supplements = defaultdict(list)
    for row in rows:
        if row['id'] in by_id:
            continue
        teacher = source(row) == 'teacher_distilled' and row['experiment'].get('train_only') is True
        if (row.get('training_only') is not True and not teacher) or row['split'] != 'train':
            raise ValueError('additional candidates must be explicitly training-only')
        scenario = row.get(scenario_field)
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError('explicit supplemental scenario required')
        supplements[scenario].append(row['id'])
    if supplement_share and not supplements:
        raise ValueError('nonzero supplement share requires supplemental scenarios')
    weights, report_groups = {}, defaultdict(lambda: {'rows': 0, 'scenarios': 0, 'mass': 0.})
    for bucket, ids in buckets.items():
        mass = (1-supplement_share) * bucket_mass[bucket]
        for identifier in ids:
            weights[identifier] = len(rows) * mass / len(ids)
        report_groups[bucket[0]]['rows'] += len(ids)
        report_groups[bucket[0]]['scenarios'] += 1
        report_groups[bucket[0]]['mass'] += mass
    for ids in supplements.values():
        mass = supplement_share / len(supplements)
        for identifier in ids:
            weights[identifier] = len(rows) * mass / len(ids)
        report_groups['supplement']['rows'] += len(ids)
        report_groups['supplement']['scenarios'] += 1
        report_groups['supplement']['mass'] += mass
    result = [weights[r['id']] for r in rows]
    if not math.isclose(sum(result)/len(rows), 1, abs_tol=1e-10):
        raise ValueError('mixture mean must equal one')
    report = {'version': 'frozen-reference-mixture-v1', 'supplement_share': supplement_share,
              'supplement_scenario_field': scenario_field,
              'groups': dict(report_groups), 'zero_weight_rows': sum(w == 0 for w in result),
              'minimum_weight': min(result), 'maximum_weight': max(result), 'mean_weight': sum(result)/len(rows),
              'row_weight_sha256': key(list(zip([r['id'] for r in rows], result))),
              'weight_semantics': 'Loss-weight mass, not measured gradient contribution; zero-share rows forward only.'}
    return result, report
