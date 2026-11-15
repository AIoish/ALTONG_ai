"""용도: 새 실제 알림의 세션별 평가 역할을 고정하고 기존 검수 도구로 예측 비공개 카드를 만든다.
생성일: 2026-11-15
"""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from filtering_training.common.artifact_io import created_date, digest, load_lines, read, write
from filtering_training.common.relevance_policy import POLICY, targetless
from filtering_training.preparation.curate_relevance import key, write_lines
from filtering_training.preparation.prepare_relevance_experiment import group_rows, input_key
from filtering_training.review.prepare_real_review_packets import answer_decisions, build_packet, write_packet

ROLES = ('real_development', 'real_final')


def draft_plan(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('preserve existing plan')
    rows = load_lines(source)
    if not rows or any(not r.get('group_id') for r in rows):
        raise ValueError('convert CSV with prepare_real_relevance first; review provisional session groups')
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, {'purpose': 'Review session boundaries and assign evaluation roles before labeling',
          'created_date': created_date(), 'source_sha256': digest(source),
          'new_capture_confirmed': False, 'natural_capture_confirmed': False,
          'sessions': {g: None for g in sorted({r['group_id'] for r in rows})}})


def partition(rows, plan, known):
    assignments = plan.get('assignments')
    if assignments is None:
        assignments = {r['id']: {'session_id': r.get('group_id'), 'role': plan.get('sessions', {}).get(r.get('group_id'))} for r in rows}
    if not rows or len({r['id'] for r in rows}) != len(rows) or set(assignments) != {r['id'] for r in rows}:
        raise ValueError('explicit session and role assignment required for every unique row')
    if plan.get('new_capture_confirmed') is not True or plan.get('natural_capture_confirmed') is not True:
        raise ValueError('new natural capture must be explicitly declared, not inferred from an old export')
    sessions = defaultdict(set)
    fresh = []
    for original in rows:
        spec = assignments[original['id']]
        if spec['role'] not in ROLES or not isinstance(spec['session_id'], str) or not spec['session_id'].strip():
            raise ValueError('valid role and nonempty session ID required')
        if original.get('provenance', {}).get('real_user_data') is not True:
            raise ValueError('real capture provenance required')
        sessions[spec['session_id']].add(spec['role'])
        row = deepcopy(original)
        row.update(score=None, split=spec['role'], evaluation_session=spec['session_id'], case_kind=spec.get('case_kind'))
        row['experiment'] = {'origin': 'real_evaluation', 'group_hints': ['new-session:' + spec['session_id']]}
        row['review'] = {'input_sha256': input_key(row)}
        fresh.append(row)
    if any(len(roles) > 1 for roles in sessions.values()):
        raise ValueError('one session cannot cross development and final')
    past = []
    for i, original in enumerate(known):
        row = deepcopy(original)
        row['id'] = f'known-{i}'
        row['experiment'] = {'group_hints': ['past:' + str(original.get('experiment_group', original.get('group_id', i)))]}
        past.append(row)
    combined = past + fresh
    groups, near = group_rows(combined, reaction_predicate=targetless)
    occupied = defaultdict(set)
    for row in past:
        occupied[groups[row['id']]].add('previously_used')
    for row in fresh:
        occupied[groups[row['id']]].add(row['split'])
        row['experiment_group'] = groups[row['id']]
    if any(len(roles) > 1 for roles in occupied.values()):
        raise ValueError('input/body/near-duplicate/session group crosses roles or previously inspected data')
    return fresh, near


def prepare(source, plan_path, inventory_path, output):
    source, plan_path, inventory_path, output = map(Path, (source, plan_path, inventory_path, output))
    if output.exists():
        raise ValueError('preserve existing evaluation preparation')
    inventory = read(inventory_path)
    if not {'training', 'development_validation'} <= {e['role'] for e in inventory['entries']}:
        raise ValueError('inventory must cover existing training and development validation')
    known, protected = [], {str(p.resolve()): digest(p) for p in (source, plan_path, inventory_path)}
    for entry in inventory['entries']:
        path = Path(entry['path'])
        if digest(path) != entry['sha256']:
            raise ValueError('inventory source changed')
        protected[str(path.resolve())] = entry['sha256']
        known.extend(load_lines(path))
    rows, near = partition(load_lines(source), read(plan_path), known)
    if read(plan_path).get('source_sha256', digest(source)) != digest(source):
        raise ValueError('plan belongs to a different capture')
    # Render cards with no predictions, suggestions or inferred labels. One input per card.
    for i, row in enumerate(rows, 1):
        row['source_number'] = str(i)
    cards = [{'case_id': f'R{i:04d}', 'round': 1 + (i-1)//20, 'source_numbers': [str(i)],
              'title': '알림 관련도 검수', 'question': '현재 작업에 대한 관련도를 1·2·3 또는 보류로 판단해 줘.',
              'ambiguity': '입력에 없는 대화나 작업 내용을 추측하지 않아도 돼.'} for i in range(1, len(rows)+1)]
    packet = build_packet(rows, {'cases': cards, 'blind': True, 'quality_labels': {}, 'previously_held_reason': '__none__'})
    packet.pop('packet_id')
    packet.update(evaluation_policy=POLICY, source_sha256=digest(source))
    packet['packet_id'] = key(packet)
    output.mkdir(parents=True)
    write_lines(output / 'rows.jsonl', rows)
    write_lines(output / 'near-duplicate-links.jsonl', near)
    write_packet(output / 'review', packet, source_sha256=digest(source))
    if any(digest(p) != h for p, h in protected.items()):
        raise ValueError('evaluation source changed')
    write(output / 'manifest.json', {'purpose': 'Prepared blind real evaluation, not certified final performance',
          'created_date': created_date(), 'policy': POLICY, 'status': 'awaiting_human_labels',
          'final_evaluation_used': False, 'training_ready': False, 'protected_sources': protected,
          'capture_declaration': read(plan_path),
          'files': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()},
          'limitations': ['Novelty/naturalness partly rely on operator declaration.',
                          'Lexical near-duplicate detection cannot prove semantic independence.']})
    return output


def apply_answers(folder, answers_path, output):
    folder, answers_path, output = map(Path, (folder, answers_path, output))
    if output.exists():
        raise ValueError('preserve previous reviewed evaluation')
    manifest = read(folder / 'manifest.json')
    for name, expected in manifest['files'].items():
        if digest(folder / name) != expected:
            raise ValueError('prepared evaluation changed')
    for path, expected in manifest['protected_sources'].items():
        if digest(path) != expected:
            raise ValueError('source changed since review preparation')
    rows = load_lines(folder / 'rows.jsonl')
    decisions = [{'id': r['id'], 'input_sha256': input_key(r), 'proposed_score': None} for r in rows]
    reviewed = answer_decisions(read(folder / 'review/packet.json'), read(answers_path), decisions)
    for row, decision in zip(rows, reviewed):
        row['score'] = decision['proposed_score']
        if targetless(row['state']['notification']['body']) and row['score'] not in (None, 1):
            raise ValueError('human label conflicts with the confirmed targetless-reaction policy')
        row['evaluation_review'] = decision
    output.mkdir(parents=True)
    for role in ROLES:
        write_lines(output / (role + '.jsonl'), [r for r in rows if r['split'] == role and r['score'] is not None])
    write_lines(output / 'pending-or-held.jsonl', [r for r in rows if r['score'] is None])
    write(output / 'manifest.json', {'purpose': 'Explicit human review; final rows sealed from training/development',
          'created_date': created_date(), 'policy': POLICY, 'final_evaluation_used': False, 'training_ready': False,
          'prepared_manifest_sha256': digest(folder / 'manifest.json'), 'answers_sha256': digest(answers_path),
          'pending_or_held_rows': sum(r['score'] is None for r in rows),
          'files': {p.name: digest(p) for p in output.iterdir() if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--answers', type=Path)
    parser.add_argument('--draft-plan', action='store_true')
    args = parser.parse_args()
    if args.draft_plan:
        draft_plan(args.source, args.output)
    elif args.answers:
        apply_answers(args.source, args.answers, args.output)
    elif args.plan and args.inventory:
        prepare(args.source, args.plan, args.inventory, args.output)
    else:
        parser.error('--plan and --inventory, or --answers, required')
