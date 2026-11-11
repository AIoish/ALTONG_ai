"""용도: 검수된 관련도 수정안에서 대비 묶음을 보존해 반복량을 줄이고 합성 앱·최근 창 구성을 다양화한다.
생성일: 2026-11-08
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import re

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import INPUT_SCHEMA, RECENT_WINDOWS_POLICY, input_text
from filtering_training.preparation.curate_relevance import OUTPUT as CURATION, core_body, key, profile, write_lines
from filtering_training.review.review_relevance_dataset import GENERATOR, POLICY, Reviewer, normalize

OUTPUT = ROOT / 'filtering_training/outputs/v15_relevance_balanced_01'
APPS = ('Slack', 'Teams', 'KakaoTalk.exe')
APP_SPECIFIC = re.compile(r'slack|teams|kakao|슬랙|팀즈|카카오|카톡|스레드|멘션|이메일|메일함', re.IGNORECASE)
CASE_METHODS = {'individual_case_read', 'user_case_confirmation'}


def stable(seed, *parts):
    return key([seed, *parts])


def metadata(row, reviewer):
    n = row['state']['notification']
    template = reviewer.matching_template(core_body(n['body']))
    rule = row['review']['rule']
    family = rule + ':' + key(template)[:12] if rule.endswith('_templates') else rule
    domain = next((t['domain'] for t in reviewer.tasks
                   if normalize(t['domain']) in normalize(n.get('title', '') + n.get('sender', ''))), None)
    if domain is None:
        tasks = reviewer.targets(row['state']['context']['window_title'])
        domain = next(iter(sorted({reviewer.tasks[i]['domain'] for i in tasks})), 'manual_or_urgent')
    return {'family': family, 'domain': domain, 'template': template,
            'tasks': reviewer.targets(row['state']['context']['window_title'])}


def select_groups(records, reviewer, *, cap=2, seed=42):
    if type(cap) is not int or cap < 1:
        raise ValueError('cap must be a positive integer')
    groups = defaultdict(list)
    for row in records:
        groups[row['group_id']].append(row)
    info = {gid: metadata(rows[0], reviewer) for gid, rows in groups.items()}
    buckets = defaultdict(list)
    selected, reasons = set(), defaultdict(list)
    for gid, rows in groups.items():
        if any(r['review']['method'] != 'template_substitution_and_context_check' for r in rows):
            selected.add(gid)
            reasons[gid].append('preserve_individually_reviewed_or_rare_scenario')
        else:
            buckets[(info[gid]['family'], info[gid]['domain'])].append(gid)
    task_counts, phrase_counts = Counter(), Counter()

    def retain(gid, reason):
        selected.add(gid)
        reasons[gid].append(reason)
        task_counts.update(set().union(*(reviewer.targets(r['state']['context']['window_title']) for r in groups[gid])))
        if info[gid]['template']:
            phrase_counts[info[gid]['template']] += 1

    for gid in sorted(selected):
        retain(gid, 'coverage_seed')
    for bucket, candidates in sorted(buckets.items(), key=lambda item: stable(seed, item[0])):
        remaining = set(candidates)
        for _ in range(min(cap, len(remaining))):
            def rank(gid):
                tasks = info[gid]['tasks']
                return (sum(task_counts[t] for t in tasks) / max(len(tasks), 1),
                        phrase_counts[info[gid]['template']], stable(seed, gid))
            chosen = min(remaining, key=rank)
            retain(chosen, 'family_domain_cap')
            remaining.remove(chosen)
    # Caps are subordinate to preserving rare wording, task and label/context contrasts.
    all_templates = {m['template'] for m in info.values() if m['template']}
    for template in sorted(all_templates):
        if not any(info[g]['template'] == template for g in selected):
            retain(min((g for g in groups if info[g]['template'] == template), key=lambda g: stable(seed, g)),
                   'rare_wording_coverage')
    all_tasks = set().union(*(m['tasks'] for m in info.values()))
    covered_tasks = set().union(*(info[g]['tasks'] for g in selected))
    for task in sorted(all_tasks - covered_tasks):
        retain(min((g for g in groups if task in info[g]['tasks']), key=lambda g: stable(seed, g)), 'task_coverage')

    def contrasts(row):
        c = row['state']['context']
        return (row['review']['rule'], row['score'], bool(c['window_title']), bool(c['recent_windows']))
    all_contrasts = {contrasts(r) for r in records}
    kept_contrasts = {contrasts(r) for g in selected for r in groups[g]}
    for contrast in sorted(all_contrasts - kept_contrasts):
        gid = min((g for g, rows in groups.items() if any(contrasts(r) == contrast for r in rows)),
                  key=lambda g: stable(seed, g))
        retain(gid, 'rare_label_context_coverage')
    kept = [r for r in records if r['group_id'] in selected]
    excluded = [{'id': r['id'], 'group_id': r['group_id'], 'score': r['score'],
                 'family': info[r['group_id']]['family'], 'domain': info[r['group_id']]['domain'],
                 'reason': 'overrepresented_authored_family_domain_group',
                 'source_input_sha256': key(json.loads(input_text(r['state'], 'relevance')))}
                for r in records if r['group_id'] not in selected]
    report = {'source_rows': len(records), 'selected_rows': len(kept), 'excluded_rows': len(excluded),
              'source_body_groups': len(groups), 'selected_body_groups': len(selected),
              'task_coverage': len(all_tasks), 'wording_coverage': len(all_templates),
              'label_context_coverage': len(all_contrasts), 'cap_per_family_domain': cap,
              'retention_reasons': dict(Counter(reason for values in reasons.values() for reason in values)),
              'note': 'Whole body groups stay together; coverage exceptions may exceed the nominal cap.'}
    return kept, excluded, report


def fixed_case(row):
    return row['review']['method'] in CASE_METHODS or row.get('provenance', {}).get('real_user_data', False)


def minimum_history(row, reviewer):
    history = row['state']['context']['recent_windows']
    for size in range(len(history) + 1):
        for positions in combinations(range(len(history)), size):
            candidate = deepcopy(row)
            candidate['state']['context']['recent_windows'] = [deepcopy(history[i]) for i in positions]
            if reviewer.decide(candidate)[0] == row['score']:
                return tuple(positions)
    raise ValueError('source label does not reproduce under current review policy: ' + row['id'])


def window_catalogue(records, reviewer):
    catalogue = {}
    for row in records:
        c = row['state']['context']
        for window in c['recent_windows'] + [{'app_name': c['active_process'], 'window_title': c['window_title']}]:
            tasks = reviewer.targets(window['window_title'])
            if window['app_name'] and tasks:
                domains = frozenset(reviewer.tasks[i]['domain'] for i in tasks)
                catalogue[(window['app_name'], window['window_title'])] = domains
    return catalogue


def make_history(row, positions, count, catalogue, reviewer, seed):
    old = row['state']['context']['recent_windows']
    if count < len(positions):
        return None
    chosen = set(positions)
    for index in range(len(old)):
        if len(chosen) == count:
            break
        chosen.add(index)
    history = [deepcopy(old[i]) for i in sorted(chosen)]
    n, c = row['state']['notification'], row['state']['context']
    tasks = reviewer.targets(n['body']) | reviewer.targets(c['window_title'])
    excluded_domains = {reviewer.tasks[i]['domain'] for i in tasks}
    excluded_domains.update(t['domain'] for t in reviewer.tasks
                            if normalize(t['domain']) in normalize(n.get('title', '') + n.get('sender', '')))
    seen = {(w['app_name'], w['window_title']) for w in history}
    current = (c['active_process'], c['window_title'])
    for pair, domains in sorted(catalogue.items(), key=lambda item: stable(seed, row['id'], item[0])):
        if len(history) == count:
            break
        if pair != current and pair not in seen and not domains.intersection(excluded_domains):
            history.append({'app_name': pair[0], 'window_title': pair[1]})
            seen.add(pair)
    if len(history) != count:
        return None
    candidate = deepcopy(row)
    candidate['state']['context']['recent_windows'] = history
    return history if reviewer.decide(candidate)[0] == row['score'] else None


def diversify_history(records, reviewer, seed):
    result = {r['id']: deepcopy(r) for r in records}
    catalogue = window_catalogue(records, reviewer)
    counts, requirements = defaultdict(Counter), {}
    for row in records:
        if fixed_case(row):
            counts[row['score']][len(row['state']['context']['recent_windows'])] += 1
        else:
            requirements[row['id']] = minimum_history(row, reviewer)
    ordered = sorted((r for r in records if not fixed_case(r)),
                     key=lambda r: (-len(requirements[r['id']]), stable(seed, r['id'])))
    changes = {}
    for row in ordered:
        minimum = requirements[row['id']]
        options = sorted(range(len(minimum), 4), key=lambda count: (counts[row['score']][count], stable(seed, row['id'], count)))
        for count in options:
            history = make_history(row, minimum, count, catalogue, reviewer, seed)
            if history is not None:
                result[row['id']]['state']['context']['recent_windows'] = history
                counts[row['score']][count] += 1
                changes[row['id']] = {'required_original_positions': list(minimum),
                                      'original_count': len(row['state']['context']['recent_windows']),
                                      'new_count': count,
                                      'synthetic_context': history != row['state']['context']['recent_windows']}
                break
        else:
            raise ValueError('no label-preserving history configuration: ' + row['id'])
    return [result[r['id']] for r in records], changes


def diversify_apps(records, seed):
    result = deepcopy(records)
    counts, eligible = defaultdict(Counter), []
    for row in result:
        n = row['state']['notification']
        if fixed_case(row) or n['app_name'] not in APPS or APP_SPECIFIC.search(' '.join(n.get(k, '') for k in ('sender', 'title', 'body'))):
            counts[row['score']][n['app_name']] += 1
        else:
            eligible.append(row)
    for row in sorted(eligible, key=lambda r: stable(seed, r['id'], 'app')):
        app = min(APPS, key=lambda app: (counts[row['score']][app], stable(seed, row['id'], app)))
        row['state']['notification']['app_name'] = app
        counts[row['score']][app] += 1
    return result


def rebalance(records, reviewer, *, cap=2, seed=42):
    selected, excluded, selection = select_groups(records, reviewer, cap=cap, seed=seed)
    histories, history_changes = diversify_history(selected, reviewer, seed)
    transformed = diversify_apps(histories, seed)
    originals = {r['id']: r for r in selected}
    output, lineage, seen = [], [], {}
    for row in transformed:
        original = originals[row['id']]
        if reviewer.decide(row)[0] != original['score']:
            raise ValueError('transformation changed relevance: ' + row['id'])
        if row['state']['context']['window_title'] != original['state']['context']['window_title']:
            raise ValueError('current window must not change')
        payload = json.loads(input_text(row['state'], 'relevance'))
        signature = key(payload)
        if signature in seen:
            previous = seen[signature]
            if previous['score'] != row['score'] or previous['group_id'] != row['group_id']:
                raise ValueError('conflicting or cross-group duplicate after balancing: ' + row['id'])
            excluded.append({'id': row['id'], 'group_id': row['group_id'], 'score': row['score'],
                             'reason': 'identical_after_label_preserving_rebalance',
                             'retained_source_id': previous['source_id'],
                             'source_input_sha256': key(json.loads(input_text(original['state'], 'relevance')))})
            continue
        seen[signature] = {'score': row['score'], 'group_id': row['group_id'], 'source_id': row['id']}
        app_changed = row['state']['notification']['app_name'] != original['state']['notification']['app_name']
        history_changed = row['state']['context']['recent_windows'] != original['state']['context']['recent_windows']
        entry = {'id': 'balanced_' + row['id'], 'source_id': row['id'], 'group_id': row['group_id'],
                 'score': row['score'], 'source_input_sha256': key(json.loads(input_text(original['state'], 'relevance'))),
                 'output_input_sha256': signature, 'app_changed': app_changed, 'history_changed': history_changed,
                 'history_evidence': history_changes.get(row['id']), 'fixed_case_preserved': fixed_case(row),
                 'meaning_check': 'Same confirmed-policy score before/after; required source window subset retained',
                 'real_capture': False}
        row['id'] = entry['id']
        row['parent_review'] = row.pop('review')
        row['review_status'] = 'synthetic_rebalanced_candidate' if app_changed or history_changed else 'reviewed_source_preserved'
        row['augmentation'] = entry
        row['training_eligible'] = False
        output.append(row)
        lineage.append(entry)
    selection['selected_rows_before_rebalance'] = selection['selected_rows']
    selection['exact_duplicates_after_rebalance'] = selection['selected_rows'] - len(output)
    selection['selected_rows'], selection['excluded_rows'] = len(output), len(excluded)
    return output, excluded, lineage, selection


def prepare(output=OUTPUT, curation=CURATION, *, cap=2, seed=42):
    output, curation = Path(output).resolve(), Path(curation).resolve()
    if output.exists():
        raise ValueError('choose a new output folder; prior data is preserved')
    pointer = read(curation / 'review/current-review.json')
    source = curation / pointer['proposed']
    if digest(source) != pointer['proposed_sha256']:
        raise ValueError('review source hash changed')
    parent = read(curation / pointer['manifest'])
    if parent['coverage'].get('needs_user_review', 0):
        raise ValueError('finish unresolved source review first')
    protected = {**parent['protected_hashes'], str(source): digest(source),
                 str(curation / pointer['manifest']): digest(curation / pointer['manifest']),
                 str(curation / 'review/current-review.json'): digest(curation / 'review/current-review.json'),
                 str(Path(__file__).resolve()): digest(__file__)}
    if any(digest(Path(path)) != h for path, h in protected.items()):
        raise ValueError('protected source changed')
    records = load_lines(source)
    reviewer = Reviewer()
    rows, excluded, lineage, selection = rebalance(records, reviewer, cap=cap, seed=seed)
    output.mkdir(parents=True)
    write_lines(output / 'relevance/dataset.jsonl', rows)
    write_lines(output / 'review/excluded.jsonl', excluded)
    write_lines(output / 'review/lineage.jsonl', lineage)
    write(output / 'review/selection.json', selection)
    write(output / 'review/policy-snapshot.json', reviewer.policy)
    if any(digest(Path(path)) != h for path, h in protected.items()):
        raise ValueError('source changed during balancing')
    manifest = {'purpose': 'Relevance-only synthetic distribution/wording improvement after full review',
                'created_date': created_date(), 'status': 'synthetic_review_pending',
                'input_schema': INPUT_SCHEMA, 'recent_windows_policy': RECENT_WINDOWS_POLICY,
                'source': str(source), 'seed': seed, 'selection': selection,
                'before_profile': profile(records), 'after_profile': profile(rows),
                'app_changed_rows': sum(r['app_changed'] for r in lineage),
                'history_changed_rows': sum(r['history_changed'] for r in lineage),
                'fixed_cases_preserved': sum(r['fixed_case_preserved'] for r in lineage),
                'training_ready': False, 'real_user_data_added': False, 'urgency_changed': False,
                'source_files_unchanged': True, 'protected_hashes': protected,
                'files': {name: {'sha256': digest(output / name)} for name in (
                    'relevance/dataset.jsonl', 'review/excluded.jsonl', 'review/lineage.jsonl',
                    'review/selection.json', 'review/policy-snapshot.json')},
                'limitations': ['Variants are authored synthetic examples, not recovered historical window captures.',
                                'Semantic preservation is checked by the reviewed rule policy, not independent human certification.',
                                'Only source-training groups are used; do not call a split of these examples an independent final test.',
                                'Other synthetic artifacts and performance on real inputs need separate evaluation.']}
    write(output / 'manifest.json', manifest)
    print(json.dumps({name: manifest[name] for name in ('status', 'selection', 'app_changed_rows', 'history_changed_rows',
                                                       'fixed_cases_preserved', 'after_profile')}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--curation-dir', type=Path, default=CURATION)
    parser.add_argument('--cap', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.output_dir, args.curation_dir, cap=args.cap, seed=args.seed)
