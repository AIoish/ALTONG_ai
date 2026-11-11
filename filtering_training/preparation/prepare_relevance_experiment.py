"""용도: 관련도 후보의 정답 충돌을 검사하고 문장·변형 묶음을 보존하는 개발용 분할을 만든다.
생성일: 2026-11-11
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import unicodedata

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import INPUT_SCHEMA, input_text
from filtering_training.preparation.curate_relevance import context_only_reaction, core_body, key, write_lines


OUTPUT = ROOT / 'filtering_training/outputs/v17_relevance_experiment_01'


def normalized(value):
    return ''.join(unicodedata.normalize('NFKC', value).split()).casefold()


def input_key(row):
    return key(json.loads(input_text(row['state'], 'relevance')))


class Groups:
    def __init__(self, size):
        self.parents = list(range(size))

    def root(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parents[max(a, b)] = min(a, b)


def group_rows(rows, *, similarity=0.92):
    """Link provenance, exact/core bodies, model inputs and high character similarity.

    Similarity is only a leakage precaution, never a reason to change a label.
    """
    if not 0 < similarity <= 1:
        raise ValueError('similarity must be in (0, 1]')
    groups, tokens, bodies = Groups(len(rows)), {}, defaultdict(list)
    for i, row in enumerate(rows):
        for token in [('input', input_key(row)), *[('lineage', t) for t in row['experiment']['group_hints']]]:
            if token in tokens:
                groups.join(i, tokens[token])
            else:
                tokens[token] = i
        body = normalized(core_body(row['state']['notification']['body']))
        if body and not context_only_reaction(row['state']['notification']['body']):
            bodies[body].append(i)
    for indices in bodies.values():
        for i in indices[1:]:
            groups.join(indices[0], i)
    near_edges = []
    long_bodies = sorted(b for b in bodies if len(b) >= 12)
    if len(long_bodies) > 1:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        matrix = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=False).fit_transform(long_bodies)
        neighbors = NearestNeighbors(metric='cosine', algorithm='brute', radius=1 - similarity + 1e-10).fit(matrix)
        # Query in batches to avoid a dense all-pairs matrix.
        for start in range(0, len(long_bodies), 128):
            distances, indices = neighbors.radius_neighbors(matrix[start:start + 128])
            for offset, (ds, js) in enumerate(zip(distances, indices)):
                a = start + offset
                for distance, b in zip(ds, js):
                    b = int(b)
                    if a >= b:
                        continue
                    ia, ib = bodies[long_bodies[a]][0], bodies[long_bodies[b]][0]
                    groups.join(ia, ib)
                    near_edges.append({'left': rows[ia]['id'], 'right': rows[ib]['id'],
                                       'cosine': round(1 - float(distance), 8)})
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[groups.root(i)].append(row)
    result = {}
    for members in components.values():
        group = 'experiment_' + key(sorted(r['id'] for r in members))[:20]
        for row in members:
            result[row['id']] = group
    return result, near_edges


def audit_rows(rows):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate source ID')
    exact, semantic = defaultdict(list), defaultdict(list)
    for row in rows:
        if type(row['score']) is not int or row['score'] not in (1, 2, 3):
            raise ValueError('expected relevance 1-3')
        payload = json.loads(input_text(row['state'], 'relevance'))
        if 'recent_processes' in row['state']['context']:
            raise ValueError('obsolete recent_processes present')
        exact[key(payload)].append(row)
        # Flag repeated content with identical visible windows, ignoring sender/time/app.
        content = deepcopy(row['state'])
        n, c = content['notification'], content['context']
        semantic[key([normalized(n['title']), normalized(n['body']), c['active_process'],
                      c['window_title'], c['recent_windows']])].append(row)
    conflicts = [{'ids': [r['id'] for r in group], 'scores': sorted({r['score'] for r in group})}
                 for group in exact.values() if len({r['score'] for r in group}) > 1]
    warnings = [{'ids': [r['id'] for r in group], 'scores': sorted({r['score'] for r in group})}
                for group in semantic.values() if len({r['score'] for r in group}) > 1]
    return {'exact_input_label_conflicts': conflicts, 'content_window_score_warnings': warnings,
            'exact_duplicate_excess_rows': sum(len(g) - 1 for g in exact.values()),
            'rows': len(rows), 'scores': dict(Counter(r['score'] for r in rows))}


def assign_splits(rows, group_ids, *, fraction=0.2, seed=42, reference=None):
    if not 0 < fraction < 0.5:
        raise ValueError('validation fraction must be between 0 and 0.5')
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_ids[row['id']]].append(row)
    forced = {g for g, members in grouped.items() if any(r['experiment']['train_only'] for r in members)}
    fixed = {g: 'train' for g in forced}
    for g, members in grouped.items():
        inherited = {(reference or {})[r['id']] for r in members if r['id'] in (reference or {})}
        if g in fixed:
            inherited.add(fixed[g])
        if len(inherited) > 1 or inherited - {'train', 'validation'}:
            raise ValueError('new grouping bridges frozen train and validation')
        if inherited:
            fixed[g] = inherited.pop()
    candidates = sorted(set(grouped) - set(fixed), key=lambda g: key([seed, g]))
    # Select whole groups, optimizing class counts against targets on eligible rows.
    totals = Counter(r['score'] for g in candidates for r in grouped[g])
    target = {s: totals[s] * fraction for s in (1, 2, 3)}
    counts, selected = Counter(), set()
    def loss(values):
        return sum(((values[s] - target[s]) / max(target[s], 1)) ** 2 for s in (1, 2, 3))
    while True:
        options = [(loss(counts + Counter(r['score'] for r in grouped[g])), g)
                   for g in candidates if g not in selected]
        if not options:
            break
        improvement, best = min(options)
        if improvement >= loss(counts):
            break
        selected.add(best)
        counts.update(r['score'] for r in grouped[best])
    result = []
    for original in rows:
        row = deepcopy(original)
        group = group_ids[row['id']]
        row['experiment_group'] = group
        row['split'] = fixed.get(group, 'validation' if group in selected else 'train')
        row['training_eligible'] = False
        result.append(row)
    for split in ('train', 'validation'):
        if {r['score'] for r in result if r['split'] == split} != {1, 2, 3}:
            raise ValueError('unable to create whole-group splits with every class')
    return result


def verify_splits(rows, source_rows, group_ids, near_edges):
    source = {r['id']: r for r in source_rows}
    if len(rows) != len(source) or {r['id'] for r in rows} != set(source):
        raise ValueError('split lost or duplicated rows')
    signatures, bodies, groups = defaultdict(set), defaultdict(set), defaultdict(set)
    by_id = {r['id']: r for r in rows}
    for row in rows:
        if row['score'] != source[row['id']]['score'] or row['state'] != source[row['id']]['state']:
            raise ValueError('source score/input changed')
        split = row['split']
        if split not in ('train', 'validation') or (row['experiment']['train_only'] and split != 'train'):
            raise ValueError('invalid split or real-data holdout contamination')
        if row['experiment_group'] != group_ids[row['id']]:
            raise ValueError('incorrect group')
        groups[row['experiment_group']].add(split)
        signatures[input_key(row)].add(split)
        body = normalized(core_body(row['state']['notification']['body']))
        if body and not context_only_reaction(row['state']['notification']['body']):
            bodies[body].add(split)
    if any(len(s) > 1 for mapping in (groups, signatures, bodies) for s in mapping.values()):
        raise ValueError('group/input/body leakage across splits')
    if any(by_id[e['left']]['split'] != by_id[e['right']]['split'] for e in near_edges):
        raise ValueError('near-duplicate leakage')
    return {'all_rows_accounted_for': True, 'scores_and_states_unchanged': True,
            'group_overlap': 0, 'exact_input_overlap': 0, 'normalized_core_body_overlap': 0,
            'detected_near_duplicate_overlap': 0,
            'reaction_grouping': 'Context-only reactions use full-input and scenario groups, not a global character group.',
            'warning': 'Lexical grouping cannot establish complete semantic independence.'}


def reviewed_examples(path):
    path = Path(path).resolve()
    manifest_path = path.parent / 'manifest.json'
    manifest = read(manifest_path)
    protected = {str(manifest_path): digest(manifest_path)}
    for name, expected in manifest['files'].items():
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent) or digest(target) != expected:
            raise ValueError('review artifact hash mismatch')
        protected[str(target)] = expected
    if str(path) not in protected:
        raise ValueError('review data is not protected by its manifest')
    selected, controls = [], set()
    for probe in load_lines(path):
        exact = probe['label_status'] == 'user_confirmed_input'
        control = (probe['label_status'] == 'user_confirmed_example' and
                   probe['context_id'] == 'approval_target' and probe['body'] in ('네', '아니'))
        if not exact and not control:
            continue
        score = probe['reference_score']
        if type(score) is not int or score not in (1, 2, 3) or input_key(probe) != probe['input_sha256']:
            raise ValueError('review score/input fingerprint mismatch')
        if exact:
            decision = probe.get('review', {})
            if (decision.get('reviewer') != 'user' or decision.get('human_reviewed') is not True or
                    decision.get('input_sha256') != probe['input_sha256'] or decision.get('proposed_score') != score):
                raise ValueError('exact user review attribution missing')
        else:
            # One representative of each confirmed approval example, not all app variants.
            control_key = (probe['context_id'], probe['body'])
            if control_key in controls:
                continue
            controls.add(control_key)
            if score != 2:
                raise ValueError('approval context must retain user-confirmed score 2')
        identifier = 'reviewed_' + probe['id']
        selected.append({'id': identifier, 'group_id': 'reviewed_' + probe['context_id'],
            'split': 'unassigned', 'state': deepcopy(probe['state']), 'score': score,
            'review_status': 'user_confirmed_input' if exact else 'confirmed_rule_applied_control',
            'review': deepcopy(probe.get('review', {})), 'training_eligible': False,
            'case_kind': 'user_confirmed_reaction' if exact else 'approval_acknowledgment_control',
            'provenance': {'origin': 'reviewed_diagnostic', 'real_user_data': probe['scope'] == 'training_side_capture',
                           'probe_id': probe['id'], 'label_status': probe['label_status']},
            'experiment': {'origin': 'reviewed', 'loss_origin': 'pilot', 'source_file': str(path),
                'source_input_sha256': probe['input_sha256'], 'group_hints': ['reviewed:context:' + probe['context_id']],
                'train_only': True, 'label_source': 'user' if exact else 'user_rule_applied',
                'label_status': 'reviewed_training_example_not_independent_gold'}})
    if not selected:
        raise ValueError('no confirmed review rows')
    return selected, protected


def load_sources(balanced, pilot, real, *, reviewed=None, protected=None):
    sources, excluded = [], []
    for name, path in [('balanced', balanced), ('pilot', pilot), ('real', real)]:
        for original in load_lines(path):
            row = deepcopy(original)
            if row['score'] is None:
                excluded.append({'id': row['id'], 'origin': name, 'reason': row['review']['reason_code']})
                continue
            hints = [name + ':group:' + row['group_id']]
            if name == 'pilot':
                hints += ['pilot:scenario:' + row['scenario']]
            if name == 'real':
                # One project/capture export is not independent validation material.
                hints += ['real:whole-capture']
            row['experiment'] = {'origin': name, 'source_file': str(path),
                                 'source_input_sha256': input_key(row), 'group_hints': hints,
                                 'train_only': name == 'real',
                                 'label_source': 'user' if row.get('review', {}).get('human_reviewed') else 'assistant',
                                 'label_status': 'candidate_not_independent_gold'}
            sources.append(row)
    if reviewed is not None:
        extra, hashes = reviewed_examples(reviewed)
        sources.extend(extra)
        if protected is not None:
            protected.update(hashes)
    return sources, excluded


def prepare(balanced, pilot, real, output=OUTPUT, *, seed=42, fraction=0.2, similarity=0.92, reference=None, reviewed=None):
    output = Path(output)
    if output.exists():
        raise ValueError('preserve existing experiment; choose a new output folder')
    protected = {str(Path(p).resolve()): digest(p) for p in (balanced, pilot, real)}
    sources, excluded = load_sources(balanced, pilot, real, reviewed=reviewed, protected=protected)
    reference_splits = {}
    if reference is not None:
        by_id = {r['id']: r for r in sources}
        for split in ('train', 'validation'):
            path = Path(reference) / f'relevance/{split}.jsonl'
            protected[str(path.resolve())] = digest(path)
            for old in load_lines(path):
                if old['id'] in reference_splits:
                    raise ValueError('duplicate frozen reference ID')
                current = by_id.get(old['id'])
                if current is None or current['score'] != old['score'] or current['state'] != old['state']:
                    raise ValueError('frozen reference row changed or missing')
                reference_splits[old['id']] = split
    audit = audit_rows(sources)
    if audit['exact_input_label_conflicts'] or audit['content_window_score_warnings']:
        raise ValueError('label conflicts require review before splitting: ' + json.dumps(audit, ensure_ascii=False))
    groups, near = group_rows(sources, similarity=similarity)
    rows = assign_splits(sources, groups, fraction=fraction, seed=seed, reference=reference_splits)
    verification = verify_splits(rows, sources, groups, near)
    output.mkdir(parents=True)
    (output / 'relevance').mkdir()
    (output / 'review').mkdir()
    for split in ('train', 'validation'):
        write_lines(output / f'relevance/{split}.jsonl', [r for r in rows if r['split'] == split])
    write_lines(output / 'review/excluded.jsonl', excluded)
    write_lines(output / 'review/near-duplicate-links.jsonl', near)
    write(output / 'review/consistency.json', audit)
    write(output / 'review/verification.json', verification)
    summary = {split: {'rows': sum(r['split'] == split for r in rows),
                      'scores': dict(Counter(r['score'] for r in rows if r['split'] == split)),
                      'origins': dict(Counter(r['experiment']['origin'] for r in rows if r['split'] == split)),
                      'groups': len({r['experiment_group'] for r in rows if r['split'] == split})}
               for split in ('train', 'validation')}
    if any(digest(path) != expected for path, expected in protected.items()):
        raise ValueError('source changed during preparation')
    manifest = {'purpose': 'Relevance-only experimental candidate split, not approved training or final test',
                'created_date': created_date(), 'status': 'candidate_split_pending_semantic_review',
                'input_schema': INPUT_SCHEMA, 'seed': seed, 'target_validation_fraction': fraction,
                'near_duplicate_cosine_threshold': similarity, 'near_duplicate_edges': len(near),
                'source_rows': len(sources) + len(excluded), 'candidate_rows': len(rows),
                'excluded_rows': len(excluded), 'splits': summary,
                'frozen_reference': str(Path(reference).resolve()) if reference else None,
                'frozen_reference_rows': len(reference_splits),
                'reviewed_source': str(Path(reviewed).resolve()) if reviewed else None,
                'reviewed_rows': sum(r['experiment']['origin'] == 'reviewed' for r in rows),
                'human_labeled_real_rows': sum(r['experiment']['origin'] == 'real' and r['experiment']['label_source'] == 'user' for r in rows),
                'training_ready': False, 'final_test_created': False, 'urgency_changed': False,
                'protected_sources': protected,
                'files': {str(p.relative_to(output)): {'sha256': digest(p)} for p in output.rglob('*') if p.is_file()},
                'limitations': ['Assistant labels remain candidates, not user-certified gold.',
                                'Existing inspected training sources cannot become an unseen final test.',
                                'Real capture stays on training side; no real-use accuracy claim from this split.',
                                'Exact/core-body and lexical near-duplicate grouping does not catch all paraphrases.',
                                'Evaluate separate unseen captures and projects before deployment.']}
    write(output / 'manifest.json', manifest)
    print(json.dumps({'splits': summary, 'excluded_rows': len(excluded), 'near_edges': len(near)}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--balanced', type=Path, default=ROOT / 'filtering_training/outputs/v15_relevance_balanced_01/relevance/dataset.jsonl')
    parser.add_argument('--pilot', type=Path, default=ROOT / 'filtering_training/outputs/v14_relevance_curation_01/new/policy-confirmed-v2/candidates.jsonl')
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reviewed-probes', type=Path)
    args = parser.parse_args()
    prepare(args.balanced, args.pilot, args.real, args.output_dir, seed=args.seed,
            reference=args.reference, reviewed=args.reviewed_probes)
