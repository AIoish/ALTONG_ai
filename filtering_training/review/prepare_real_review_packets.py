"""용도: 실제 알림의 경계 사례를 로컬 HTML·Markdown 카드로 묶고 명시적 답변만 반영한다.
생성일: 2026-11-08
"""

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

from filtering_training.common.artifact_io import created_date, digest, load_lines, read, write
from filtering_training.preparation.curate_relevance import key, write_lines
from filtering_training.preparation.prepare_real_relevance import OUTPUT
from filtering_training.review.apply_real_relevance_review import apply_review


TEMPLATE = Path(__file__).parent / 'templates/real_relevance_packets.html'


def latest(folder):
    pointer = read(folder / 'review/current-review.json')
    return folder / Path(pointer['manifest']).parent


def build_packet(rows, plan):
    by_number = {str(row['source_number']): row for row in rows}
    if len(by_number) != len(rows):
        raise ValueError('duplicate source numbers')
    cases, seen, ids = [], set(), set()
    for spec in plan['cases']:
        case = deepcopy(spec)
        if case['case_id'] in ids or case['round'] not in (1, 2, 3):
            raise ValueError('duplicate case or invalid round')
        ids.add(case['case_id'])
        members = []
        for number in case.pop('source_numbers'):
            number = str(number)
            if number in seen or number not in by_number:
                raise ValueError('duplicate or missing source number')
            seen.add(number)
            row = by_number[number]
            members.append({'source_number': number, 'id': row['id'],
                            'input_sha256': row['review']['input_sha256'], 'state': row['state'],
                            'received_kst': row['provenance']['received_kst'],
                            'quality_labels': [plan['quality_labels'].get(flag, flag)
                                               for flag in row['provenance']['quality_flags']]})
        scores = {by_number[m['source_number']]['score'] for m in members}
        if len(scores) != 1:
            raise ValueError('grouped sources must share proposed score')
        case.update(members=members, suggested_score=scores.pop())
        cases.append(case)
    held = sum(row['review'].get('reason_code') == plan['previously_held_reason'] for row in rows)
    packet = {'purpose': 'Private local relevance review; unanswered cases are not approvals',
              'created_date': created_date(), 'total_source_rows': len(rows),
              'selected_source_rows': len(seen), 'previously_held_rows': held,
              'remaining_unselected_rows': len(rows) - len(seen) - held, 'cases': cases}
    packet['packet_id'] = key(packet)
    return packet


def prepare(folder, plan_path, batch='user-packets-01'):
    folder = Path(folder)
    if not batch or Path(batch).name != batch or batch in ('.', '..'):
        raise ValueError('invalid batch')
    output = folder / 'review' / batch
    if output.exists():
        raise ValueError('preserve existing packet; choose a new batch')
    source = latest(folder)
    packet = build_packet(load_lines(source / 'proposed.jsonl'), read(plan_path))
    # Frozen source references are part of the packet fingerprint.
    packet.pop('packet_id')
    packet['source_proposals_sha256'] = digest(source / 'proposed.jsonl')
    packet['packet_id'] = key(packet)
    output.mkdir(parents=True)
    write(output / 'packet.json', packet)
    payload = json.dumps(packet, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = TEMPLATE.read_text(encoding='utf-8')
    if template.count('__PACKET_JSON__') != 1:
        raise ValueError('expected one packet placeholder')
    (output / '검수하기.html').write_text(template.replace('__PACKET_JSON__', payload), encoding='utf-8')
    lines = ['<!-- 용도: 실제 알림 경계 사례 검수 카드. 생성일: ' + created_date() + ' -->',
             '# 관련도 검수', '', '먼저 1차 A~F만 검수. 답변: `A=2, B=3, C=보류`.',
             '1=무관, 2=일부 연결·도움 불명확, 3=현재 작업에 유용. 미응답은 승인되지 않음.', '']
    with (output / '카드목록.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['카드', '묶음', '원본 번호', '판단할 내용', '제안', '내 점수', '메모'])
        for case in packet['cases']:
            suggestion = case['suggested_score'] if case['suggested_score'] is not None else '보류'
            numbers = ', '.join(m['source_number'] for m in case['members'])
            writer.writerow([case['case_id'], case['round'], numbers, case['question'], suggestion, '', ''])
            lines += [f"## {case['round']}차 {case['case_id']}. {case['title']}", '',
                      case['question'], '', f"내 제안: **{suggestion}** · {case['ambiguity']}", '']
            for m in case['members']:
                n, c = m['state']['notification'], m['state']['context']
                lines += [f"원본 {m['source_number']} · {m['received_kst']} KST", '',
                          '```text', f"알림: {n['app_name']} / {n['sender']} / {n['title']}",
                          '본문: ' + (n['body'] or '(빈 본문)'),
                          f"현재 창: {c['active_process']} / {c['window_title']}", '최근 창:']
                lines += ['  ' + w['app_name'] + ' / ' + w['window_title'] for w in c['recent_windows']]
                lines += ['```', '', '자료 주의사항: ' + ' · '.join(m['quality_labels']), '']
    (output / '검수카드.md').write_text('\n'.join(lines), encoding='utf-8')
    write(output / 'manifest.json', {'purpose': 'Local private review cards; no labels changed',
          'created_date': created_date(), 'packet_id': packet['packet_id'],
          'plan_sha256': digest(plan_path), 'source': str(source), 'training_ready': False,
          'files': {p.name: {'sha256': digest(p)} for p in output.iterdir() if p.is_file()}})
    return output


def answer_decisions(packet, answers, decisions):
    frozen = deepcopy(packet)
    signature = frozen.pop('packet_id')
    if key(frozen) != signature or answers.get('packet_id') != signature:
        raise ValueError('packet fingerprint mismatch')
    cases = {case['case_id']: case for case in packet['cases']}
    chosen = answers.get('answers')
    if not isinstance(chosen, dict) or not chosen or set(chosen) - set(cases):
        raise ValueError('explicit known card answers required')
    result = deepcopy(decisions)
    by_id = {d['id']: d for d in result}
    if len(by_id) != len(result):
        raise ValueError('duplicate decision IDs')
    for case_id, answer in chosen.items():
        score, note = answer['score'], answer.get('note', '')
        if (score is not None and (type(score) is not int or score not in (1, 2, 3))) or not isinstance(note, str):
            raise ValueError('expected 1-3 or null and text note')
        for member in cases[case_id]['members']:
            decision = by_id[member['id']]
            if decision['input_sha256'] != member['input_sha256']:
                raise ValueError('source input fingerprint changed')
            previous = deepcopy(decision)
            decision.update(proposed_score=score, reviewer='user', human_reviewed=True,
                            reason=f'사용자가 검수 카드 {case_id}에서 명시적으로 선택',
                            reason_code='user_card_answer' if score is not None else 'user_held_card',
                            method='explicit_user_packet_answer', user_note=note,
                            packet_id=signature, case_id=case_id, previous_review=previous)
    return result


def apply_answers(folder, packet_path, answers_path, batch):
    folder = Path(folder)
    decisions = answer_decisions(read(packet_path), read(answers_path),
                                 load_lines(latest(folder) / 'decisions.jsonl'))
    draft = folder / 'review' / (batch + '-answers.jsonl')
    if draft.exists() or (folder / 'review' / batch).exists():
        raise ValueError('preserve previous answers; choose a new batch')
    write_lines(draft, decisions)
    return apply_review(folder, decisions_path=draft, batch=batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--answers', type=Path)
    parser.add_argument('--batch', required=True)
    args = parser.parse_args()
    if args.answers:
        if not args.packet:
            parser.error('--packet is required with --answers')
        apply_answers(args.folder, args.packet, args.answers, args.batch)
    else:
        if not args.plan:
            parser.error('--plan is required to prepare cards')
        print(prepare(args.folder, args.plan, args.batch))
