"""용도: 실제 CSV의 시각·최근 창·빈 본문을 보존하고 기존 예측을 정답으로 가져오지 않는지 확인한다.
생성일: 2026-11-08
"""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from filtering_training.common.artifact_io import digest, load_lines, read
from filtering_training.common.encoder_input import input_text
from filtering_training.preparation.curate_relevance import key, write_lines
from filtering_training.preparation.prepare_real_relevance import (
    COLUMNS, convert_rows, kst_to_utc, parse_window, prepare, provisional_groups,
)
from filtering_training.review.apply_real_relevance_review import apply_review, merge_decisions


def raw(number='1', timestamp='2026-11-08 01:30:00', body='Fix shipped.'):
    row = {column: '' for column in COLUMNS}
    row.update({'번호': number, '받은 시각': timestamp, '앱': 'KakaoTalk.exe', '발신자': 'colleague',
                '제목': 'Project room', '본문': body, '모델 긴급도': '4', '모델 관련도': '5', '통과': '통과',
                '현재 창 앱': 'Code.exe', '현재 창 제목': 'main.py',
                '최근 창 1': 'Code.exe | main.py', '최근 창 2': 'chrome.exe | Login issue | #184'})
    return row


class RealRelevanceImportTests(unittest.TestCase):
    def decision(self, row, score=2):
        return {'id': row['id'], 'input_sha256': key(json.loads(input_text(row['state'], 'relevance'))),
                'proposed_score': score, 'reason': 'Reviewed supplied context, not the old model score',
                'reviewer': 'assistant', 'reason_code': 'test_review'}

    def test_kst_conversion_crosses_utc_date_without_guessing(self):
        self.assertEqual(kst_to_utc('2026-11-08 01:30:00'), '2026-11-07T16:30:00Z')
        with self.assertRaises(ValueError):
            kst_to_utc('2026-11-08T01:30:00+09:00')

    def test_recent_window_preserves_title_pipes_and_exact_current_comparison(self):
        self.assertEqual(parse_window('chrome.exe | Login | #184')['window_title'], 'Login | #184')
        self.assertIsNone(parse_window(''))
        with self.assertRaises(ValueError):
            parse_window('missing separator')
        item = raw()
        item['최근 창 3'] = 'code.exe | main.py'
        records, _ = convert_rows([item], 'a' * 64)
        windows = records[0]['state']['context']['recent_windows']
        self.assertEqual(windows, [{'app_name': 'chrome.exe', 'window_title': 'Login issue | #184'},
                                   {'app_name': 'code.exe', 'window_title': 'main.py'}])

    def test_model_predictions_never_become_gold_and_unused_x_does_not_drop_row(self):
        item = raw()
        item['보낼지(O/X)'] = 'X'
        item['기대 관련도(1~5)'] = '5'
        records, references = convert_rows([item], 'a' * 64)
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]['score'])
        self.assertFalse(records[0]['training_eligible'])
        self.assertNotIn('stored_model_predictions', records[0])
        self.assertEqual(references[0]['stored_model_predictions']['relevance5'], 5)
        self.assertEqual(references[0]['source_expected_fields']['relevance_header_1_to_5'], '5')

    def test_empty_body_and_missing_sender_are_preserved_for_title_review(self):
        item = raw(body='')
        item['발신자'] = ''
        before = deepcopy(item)
        records, _ = convert_rows([item], 'a' * 64)
        self.assertEqual(records[0]['state']['notification']['body'], '')
        self.assertEqual(records[0]['state']['notification']['sender'], '')
        self.assertIn('empty_body_review_title_before_excluding', records[0]['provenance']['quality_flags'])
        self.assertEqual(item, before)

    def test_historical_missing_history_is_reference_metadata_not_modern_input_loss(self):
        item = raw()
        item['참고'] = '11/01 이전: 모델 입력에 최근 창 없음; 카톡 OCR 수정 전(잡음 가능); 창 제목 일부 가림'
        records, _ = convert_rows([item], 'a' * 64)
        self.assertFalse(records[0]['provenance']['historical_model_used_recent_windows'])
        self.assertEqual(len(records[0]['state']['context']['recent_windows']), 1)
        self.assertEqual(len(records[0]['provenance']['raw_recent_windows']), 2)
        self.assertNotIn('last_updated', records[0]['state']['context'])

    def test_duplicate_numbers_and_malformed_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate source'):
            convert_rows([raw(), raw()], 'a' * 64)
        item = raw()
        item.pop('본문')
        with self.assertRaisesRegex(ValueError, 'columns'):
            convert_rows([item], 'a' * 64)

    def test_groups_link_close_conversations_and_repeated_bodies_without_assigning_splits(self):
        items = [raw('1', body='Message A'), raw('2', '2026-11-08 01:40:00', 'Message B'),
                 raw('3', '2026-11-09 12:00:00', 'Message A')]
        records, _ = convert_rows(items, 'a' * 64)
        grouping = provisional_groups(records)
        self.assertEqual(grouping['groups'], 1)
        self.assertEqual({r['split'] for r in records}, {'unassigned'})
        self.assertTrue(all(r['group_status'] == 'provisional' for r in records))

    def test_blank_bodies_do_not_merge_unrelated_rooms(self):
        a, b = raw('1', body=''), raw('2', body='')
        b['제목'] = 'Other room'
        records, _ = convert_rows([a, b], 'a' * 64)
        self.assertEqual(provisional_groups(records)['groups'], 2)

    def test_round_trip_csv_keeps_multiline_text_and_separates_blind_review(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'source.csv', Path(temp) / 'review'
            item = raw(body='Line one, "quoted"\nLine two')
            with source.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerow(item)
            source_hash = digest(source)
            manifest = prepare(source, output)
            self.assertEqual(manifest['rows'], 1)
            self.assertEqual(digest(source), source_hash)
            records = load_lines(output / 'relevance/unlabeled.jsonl')
            self.assertEqual(records[0]['state']['notification']['body'], item['본문'])
            with (output / 'review/blind.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['body'], item['본문'])
            self.assertEqual(rows[0]['reviewed_relevance_1_to_3'], '')
            self.assertNotIn('모델 관련도', rows[0])
            self.assertNotIn('stored_model_predictions', rows[0])
            with self.assertRaisesRegex(ValueError, 'preserved'):
                prepare(source, output)

    def test_annotation_requires_full_coverage_and_exact_input_fingerprints(self):
        rows, _ = convert_rows([raw('1'), raw('2', body='Other notification')], 'a' * 64)
        decisions = [self.decision(r) for r in rows]
        with self.assertRaisesRegex(ValueError, 'cover every'):
            merge_decisions(rows, decisions[:1])
        changed = deepcopy(rows)
        changed[0]['state']['context']['window_title'] = 'Different task'
        with self.assertRaisesRegex(ValueError, 'input changed'):
            merge_decisions(changed, decisions)

    def test_annotation_is_a_proposal_and_does_not_copy_old_prediction(self):
        rows, references = convert_rows([raw()], 'a' * 64)
        before = deepcopy(rows)
        result = merge_decisions(rows, [self.decision(rows[0], 2)])
        self.assertEqual(references[0]['stored_model_predictions']['relevance5'], 5)
        self.assertEqual(result[0]['score'], 2)
        self.assertEqual(result[0]['review_status'], 'ai_proposed_unreviewed')
        self.assertFalse(result[0]['training_eligible'])
        self.assertEqual(rows, before)
        held = merge_decisions(rows, [self.decision(rows[0], None)])
        self.assertIsNone(held[0]['score'])
        self.assertEqual(held[0]['review_status'], 'needs_context_review')

    def test_saved_annotation_preserves_import_and_has_no_old_score_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'source.csv', Path(temp) / 'review'
            with source.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerow(raw())
            prepare(source, output)
            rows = load_lines(output / 'relevance/unlabeled.jsonl')
            source_hash = digest(output / 'relevance/unlabeled.jsonl')
            path = Path(temp) / 'decisions.jsonl'
            write_lines(path, [self.decision(rows[0])])
            manifest = apply_review(output, decisions_path=path)
            self.assertEqual(manifest['proposed_score_counts'], {2: 1})
            self.assertEqual(digest(output / 'relevance/unlabeled.jsonl'), source_hash)
            with (output / 'review/assistant-pass-01/review.csv').open(encoding='utf-8-sig', newline='') as stream:
                columns = next(csv.reader(stream))
            self.assertNotIn('stored_model_predictions', columns)
            self.assertNotIn('모델 관련도', columns)


if __name__ == '__main__':
    unittest.main()
