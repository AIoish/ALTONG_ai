"""용도: 사용자 CSV를 KST 기준 관련도 1~3 검수 입력으로 옮기고 기존 예측·품질 이력을 분리 보존한다.
생성일: 2026-11-08
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from filtering_training.common.artifact_io import ROOT, created_date, digest, write
from filtering_training.common.encoder_input import INPUT_SCHEMA, RECENT_WINDOWS_POLICY, input_text, previous_windows
from filtering_training.preparation.curate_relevance import body_group, key, write_lines

OUTPUT = ROOT / 'filtering_training/outputs/v16_real_relevance_review_01'
KST = timezone(timedelta(hours=9))
COLUMNS = ('번호', '받은 시각', '앱', '발신자', '제목', '본문', '모델 긴급도', '모델 관련도', '통과',
           '현재 창 앱', '현재 창 제목', '최근 창 1', '최근 창 2', '최근 창 3', '참고',
           '보낼지(O/X)', '기대 긴급도(1~5)', '기대 관련도(1~5)', '메모')
FLAGS = {'historical_model_without_recent_windows': '11/01 이전', 'pre_ocr_fix': 'OCR 수정 전',
         'manual_test_message': '시험 메시지', 'mock_notification': '모의 알림',
         'redacted_window': '창 제목 일부 가림', 'replaced_personal_fields': '개인·학교 정보 대체',
         'room_name_ocr_corrected': '방 이름만 OCR 오인식'}
BLIND_COLUMNS = ('id', 'source_number', 'provisional_group', 'received_kst', 'notification_app', 'sender',
                 'title', 'body', 'current_app', 'current_title', 'recent_windows', 'quality_flags',
                 'reviewed_relevance_1_to_3', 'reviewer_note')


def kst_to_utc(value):
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is not None:
        raise ValueError('expected the user-confirmed timezone-naive KST export')
    return parsed.replace(tzinfo=KST).astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_window(value):
    if not value.strip():
        return None
    app, separator, title = value.partition(' | ')
    if not separator or not app.strip():
        raise ValueError('recent window must contain app | title')
    return {'app_name': app.strip(), 'window_title': title.strip()}


def prediction_reference(raw):
    scores = {}
    for column, name in (('모델 긴급도', 'urgency5'), ('모델 관련도', 'relevance5')):
        value = raw[column].strip()
        if value and value not in ('1', '2', '3', '4', '5'):
            raise ValueError('invalid stored model prediction')
        scores[name] = int(value) if value else None
    return {'source_number': raw['번호'], 'stored_model_predictions': scores,
            'stored_pass_text': raw['통과'], 'source_notes': raw['참고'], 'source_memo': raw['메모'],
            'source_expected_fields': {'urgency_1_to_5': raw['기대 긴급도(1~5)'],
                                       'relevance_header_1_to_5': raw['기대 관련도(1~5)']},
            'use': 'triage_only_not_gold_or_replay_baseline', 'model_hash': None}


def convert_rows(source_rows, source_hash):
    records, references, seen = [], [], set()
    for ordinal, raw in enumerate(source_rows, 1):
        if set(raw) != set(COLUMNS):
            raise ValueError('unexpected CSV columns or extra field values')
        if any(value is None for value in raw.values()):
            raise ValueError('missing CSV field value')
        number = raw['번호'].strip()
        if not number or number in seen:
            raise ValueError('missing or duplicate source number')
        seen.add(number)
        quality = [flag for flag, phrase in FLAGS.items() if phrase in raw['참고']]
        if not raw['본문'].strip():
            quality.append('empty_body_review_title_before_excluding')
        notification = {'app_name': raw['앱'], 'sender': raw['발신자'], 'title': raw['제목'],
                        'body': raw['본문'], 'timestamp': kst_to_utc(raw['받은 시각'])}
        windows = [parse_window(raw[column]) for column in ('최근 창 1', '최근 창 2', '최근 창 3')]
        raw_windows = [window for window in windows if window is not None]
        context = {'active_process': raw['현재 창 앱'], 'window_title': raw['현재 창 제목'],
                   'recent_windows': raw_windows}
        clean = previous_windows(context)
        if len(clean) != len(raw_windows):
            quality.append('current_window_removed_from_recent')
        context['recent_windows'] = clean
        ident = f'real_{source_hash[:12]}_{number}'
        if raw['기대 관련도(1~5)'].strip():
            quality.append('source_expected_score_requires_scale_review')
        record = {'id': ident, 'source_number': number, 'group_id': None, 'group_status': 'provisional',
                  'body_group_id': body_group(raw['본문']) if raw['본문'].strip() else None,
                  'split': 'unassigned', 'state': {'notification': notification, 'context': context},
                  'score': None, 'review_status': 'needs_relevance_label', 'training_eligible': False,
                  'provenance': {'source_sha256': source_hash, 'csv_record': ordinal, 'source_number': number,
                                 'received_kst': raw['받은 시각'], 'source_timezone': 'Asia/Seoul',
                                 'real_user_data': True, 'capture_origin': 'mixed_export_not_all_natural_messages',
                                 'quality_flags': quality, 'raw_recent_windows': raw_windows,
                                 'historical_model_used_recent_windows': False if 'historical_model_without_recent_windows' in quality else None,
                                 'stored_prediction_reference': ident}}
        reference = prediction_reference(raw)
        reference['id'] = ident
        records.append(record)
        references.append(reference)
        payload = json.loads(input_text(record['state'], 'relevance'))
        if payload['recent_windows'] != clean:
            raise ValueError('normalized recent-window input changed unexpectedly')
    return records, references


def provisional_groups(records, *, session_minutes=30):
    """Conservative links; never automatically assign train/test on this small mixed export."""
    if type(session_minutes) is not int or session_minutes < 1:
        raise ValueError('session_minutes must be positive')
    parents = {r['id']: r['id'] for r in records}

    def root(ident):
        while parents[ident] != ident:
            parents[ident] = parents[parents[ident]]
            ident = parents[ident]
        return ident

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    bodies, conversations = {}, defaultdict(list)
    for row in records:
        if row['body_group_id']:
            if row['body_group_id'] in bodies:
                union(row['id'], bodies[row['body_group_id']])
            bodies[row['body_group_id']] = row['id']
        n = row['state']['notification']
        app = n['app_name'].casefold()
        if app in ('kakaotalk.exe', 'kakaotalk', '카카오톡'):
            app = 'kakaotalk'
        if app in ('discord', 'discord.exe'):
            app = 'discord'
        conversations[(app, n['title'])].append(row)
    for rows in conversations.values():
        ordered = sorted(rows, key=lambda r: r['state']['notification']['timestamp'])
        for before, after in zip(ordered, ordered[1:]):
            a = datetime.fromisoformat(before['state']['notification']['timestamp'].replace('Z', '+00:00'))
            b = datetime.fromisoformat(after['state']['notification']['timestamp'].replace('Z', '+00:00'))
            if b - a <= timedelta(minutes=session_minutes):
                union(before['id'], after['id'])
    groups = defaultdict(list)
    for row in records:
        groups[root(row['id'])].append(row)
    for members in groups.values():
        ident = 'capture_' + key(sorted(r['id'] for r in members))[:16]
        for row in members:
            row['group_id'] = ident
    return {'groups': len(groups), 'largest_group_rows': max(map(len, groups.values()), default=0),
            'session_minutes': session_minutes, 'links': ['same_nonempty_normalized_body', 'same_app_title_nearby_time'],
            'status': 'provisional_requires_conversation_scenario_review',
            'note': 'Repeated generic replies can conservatively join unrelated sessions; do not claim these groups establish independence.'}


def blind_markdown(records):
    lines = ['<!-- 용도: 사용자 알림의 관련도 1~3을 기존 모델 예측 없이 검수한다.',
             f'생성일: {created_date()} -->', '', '# 실제 CSV 관련도 검수', '',
             '개인 자료다. 원본의 시험·모의·OCR·가림 여부를 함께 확인한다. 기존 모델 예측은 별도 파일에만 있다.',
             '현재 작업이 명확하면 우선하고, 모호하면 최근 창으로 보완한다. 관련도 1·2·3을 새로 검수한다.', '']
    for row in records:
        n, c = row['state']['notification'], row['state']['context']
        lines.extend([f"## 원본 {row['source_number']} · {row['id']}", '',
                      f"- 받은 시각(KST): {row['provenance']['received_kst']}",
                      f"- 알림: {n['app_name']} / {n['sender']} / {n['title']}",
                      f"- 본문: {n['body'] or '(빈 본문 — 제목과 함께 판단)'}",
                      f"- 현재 창: {c['active_process']} / {c['window_title']}",
                      '- 최근 창: ' + (' → '.join(w['app_name'] + ' / ' + w['window_title'] for w in c['recent_windows']) or '(없음)'),
                      '- 주의: ' + (', '.join(row['provenance']['quality_flags']) or '(별도 표시 없음)'),
                      f"- 임시 묶음: {row['group_id']}", '',
                      '- 관련도 정답(1~3):', '- 판단 근거:', '- 검수 결과: 유지 / 수정 / 제외 / 보류', ''])
    return '\n'.join(lines)


def prepare(source_csv, output=OUTPUT):
    source_csv, output = Path(source_csv).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('use a new output folder; source and previous review are preserved')
    source_hash = digest(source_csv)
    with source_csv.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != set(COLUMNS):
            raise ValueError('expected the confirmed notification CSV schema')
        source_rows = list(reader)
    records, references = convert_rows(source_rows, source_hash)
    grouping = provisional_groups(records)
    if not records:
        raise ValueError('empty CSV')
    assert digest(source_csv) == source_hash
    output.mkdir(parents=True)
    write_lines(output / 'relevance/unlabeled.jsonl', records)
    write_lines(output / 'reference/old-predictions.jsonl', references)
    (output / 'review').mkdir()
    with (output / 'review/blind.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=BLIND_COLUMNS)
        writer.writeheader()
        for row in records:
            n, c = row['state']['notification'], row['state']['context']
            writer.writerow({'id': row['id'], 'source_number': row['source_number'], 'provisional_group': row['group_id'],
                             'received_kst': row['provenance']['received_kst'], 'notification_app': n['app_name'],
                             'sender': n['sender'], 'title': n['title'], 'body': n['body'],
                             'current_app': c['active_process'], 'current_title': c['window_title'],
                             'recent_windows': json.dumps(c['recent_windows'], ensure_ascii=False),
                             'quality_flags': ', '.join(row['provenance']['quality_flags']),
                             'reviewed_relevance_1_to_3': '', 'reviewer_note': ''})
    (output / 'review/blind.md').write_text(blind_markdown(records), encoding='utf-8')
    if digest(source_csv) != source_hash:
        raise ValueError('source CSV changed during import')
    manifest = {'purpose': 'User CSV to relevance-only unlabeled review; stored model scores are not gold',
                'created_date': created_date(), 'status': 'unlabeled_review_pending', 'input_schema': INPUT_SCHEMA,
                'recent_windows_policy': RECENT_WINDOWS_POLICY, 'source': str(source_csv), 'source_sha256': source_hash,
                'source_files_unchanged': True, 'rows': len(records), 'source_timezone': 'Asia/Seoul',
                'source_timezone_confirmation': 'user_confirmed_KST', 'unused_column': '보낼지(O/X)',
                'unused_column_policy': 'ignored_not_an_inclusion_gate_or_gold_pass_label',
                'target_relevance_classes': [1, 2, 3], 'labels_assigned': False, 'splits_assigned': False,
                'training_ready': False, 'grouping': grouping,
                'quality_flag_counts': dict(Counter(flag for r in records for flag in r['provenance']['quality_flags'])),
                'body_length_counts': {'empty': sum(not r['state']['notification']['body'].strip() for r in records),
                                      '1_to_2': sum(1 <= len(r['state']['notification']['body'].strip()) <= 2 for r in records),
                                      '3_to_10': sum(3 <= len(r['state']['notification']['body'].strip()) <= 10 for r in records)},
                'files': {name: {'sha256': digest(output / name)} for name in (
                    'relevance/unlabeled.jsonl', 'reference/old-predictions.jsonl', 'review/blind.csv', 'review/blind.md')},
                'limitations': ['Mixed natural, test, mock, pre-OCR-fix and redacted material, not 177 independent natural examples.',
                                'Old model hashes/raw inputs are unavailable; stored predictions are triage references only.',
                                'Modern input uses exported recent windows even where the historical model omitted them.',
                                'No current-window update timestamps or durations were invented.',
                                'No existing training or evaluation set was modified or combined.']}
    write(output / 'manifest.json', manifest)
    print(json.dumps({name: manifest[name] for name in ('status', 'rows', 'grouping', 'quality_flag_counts', 'body_length_counts')}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', type=Path)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.source_csv is None:
        sources = sorted((ROOT / 'filtering_training/data').glob('*.csv'))
        if len(sources) != 1:
            parser.error('specify --source-csv when zero or multiple CSV files are present')
        args.source_csv = sources[0]
    prepare(args.source_csv, args.output_dir)
