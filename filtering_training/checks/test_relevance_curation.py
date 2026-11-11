"""용도: 반복 표현 정리가 문맥·라벨 대비를 보존하고 새 관련도 검수안이 입력 통제·분할 규칙을 지키는지 확인한다.
생성일: 2026-11-08
"""

from copy import deepcopy
import re
import unittest

from filtering_training.generation.prepare_relevance_diversity import make_drafts, validate_drafts
from filtering_training.preparation.curate_relevance import body_group, curate
from filtering_training.quality.audit_relevance_diversity import template_report


def source(body, score='5', *, window='Login bug', recent=None):
    return {'state': {'notification': {'app_name': 'Slack', 'sender': 'colleague', 'title': 'Login',
                                      'body': body, 'timestamp': '2026-11-08T03:00:00Z'},
                      'context': {'active_process': 'Code.exe', 'window_title': window,
                                  'recent_windows': recent or []}},
            'expected': {'relevance': score}}


class RelevanceCurationTests(unittest.TestCase):
    def test_reduces_only_matching_wrappers_and_records_every_exclusion(self):
        rows = [source(body) for body in ('Fix deployed.', '확인 부탁드립니다. Fix deployed.',
                                         '현장 전달: Fix deployed.', '중요 연락입니다. Fix deployed.')]
        before = deepcopy(rows)
        retained, excluded, repeats = curate(rows)
        self.assertEqual(len(retained), 2)
        self.assertEqual(len(excluded), 2)
        self.assertEqual(len(repeats), 1)
        self.assertEqual(retained[0]['state']['notification']['body'], 'Fix deployed.')
        self.assertEqual({r['id'] for r in retained} | {r['id'] for r in excluded},
                         {'v13rel_00001', 'v13rel_00002', 'v13rel_00003', 'v13rel_00004'})
        self.assertEqual(rows, before)

    def test_distinct_contexts_labels_and_meaningful_negation_survive(self):
        rows = [source('Fix deployed.'), source('Fix deployed.', '1', window='Travel booking'),
                source('Fix NOT deployed.'), source('Fix deployed.', '4'),
                source('Fix deployed.', recent=[{'app_name': 'chrome.exe', 'window_title': 'New evidence'}])]
        retained, excluded, _ = curate(rows, keep_variants=1)
        self.assertEqual(len(retained), len(rows))
        self.assertEqual(excluded, [])
        for old, new in zip(rows, retained):
            self.assertEqual(old['state'], new['state'])
            self.assertEqual(int(old['expected']['relevance']), new['source_score5'])
            self.assertEqual(new['split'], 'train_only')

    def test_metadata_changes_do_not_get_deleted_as_duplicates(self):
        rows = [source('Fix deployed.') for _ in range(3)]
        rows[1]['state']['notification']['sender'] = 'other sender'
        rows[2]['state']['notification']['timestamp'] = '2026-11-08T04:00:00Z'
        retained, excluded, _ = curate(rows, keep_variants=1)
        self.assertEqual(len(retained), 3)
        self.assertEqual(excluded, [])

    def test_group_key_keeps_prefix_variants_together(self):
        self.assertEqual(body_group('Fix deployed.'), body_group('현장 전달: Fix deployed.'))
        for bad in (0, -1, True):
            with self.assertRaises(ValueError):
                curate([source('Fix deployed.')], keep_variants=bad)

    def test_template_report_flags_repetition_without_deleting_or_merging_negation(self):
        records, _, _ = curate([source('Login A is fixed.'), source('Login B is fixed.'), source('Login C is NOT fixed.')])
        before = deepcopy(records)
        report = template_report(records, [('Login {item} is fixed.', re.compile(r'Login [ABC] is fixed\.'))])
        self.assertEqual(report['matched_rows'], 2)
        self.assertEqual(report['unmatched_rows'], 1)
        self.assertEqual(records, before)

    def test_drafts_are_reproducible_and_not_approved_training_data(self):
        rows = make_drafts()
        self.assertEqual(rows, make_drafts())
        self.assertEqual(len(rows), 148)
        validate_drafts(rows)
        self.assertTrue(all(r['review_status'] == 'ai_proposed_unreviewed' for r in rows))
        self.assertTrue(all(r['split'] == 'unassigned' for r in rows))

    def test_controlled_app_and_history_pairs_change_only_one_input_field(self):
        rows = make_drafts()
        by_id = {r['id']: r for r in rows}
        count = 0
        for row in rows:
            if row['case_kind'] not in ('app_invariance', 'irrelevant_history_invariance'):
                continue
            original = by_id[row['paired_with']]
            current = deepcopy(row['state'])
            if row['case_kind'] == 'app_invariance':
                self.assertNotEqual(current['notification']['app_name'], original['state']['notification']['app_name'])
                current['notification']['app_name'] = original['state']['notification']['app_name']
            else:
                self.assertNotEqual(current['context']['recent_windows'], original['state']['context']['recent_windows'])
                current['context']['recent_windows'] = original['state']['context']['recent_windows']
            self.assertEqual(current, original['state'])
            self.assertEqual(row['score'], original['score'])
            self.assertEqual(row['group_id'], original['group_id'])
            count += 1
        self.assertEqual(count, 12)

    def test_current_history_conflicts_keep_the_notification_unchanged(self):
        rows = make_drafts()
        by_id = {r['id']: r for r in rows}
        pairs = [r for r in rows if r['case_kind'] == 'current_history_conflict']
        self.assertEqual(len(pairs), 16)
        for row in pairs:
            original = by_id[row['paired_with']]
            self.assertEqual(row['state']['notification'], original['state']['notification'])
            self.assertNotEqual(row['state']['context']['window_title'], original['state']['context']['window_title'])
            self.assertEqual((original['score'], row['score']), (3, 1))

    def test_unknown_character_pairs_have_no_other_input_difference(self):
        rows = [r for r in make_drafts() if r['case_kind'] == 'uninformative_character']
        for first, second in zip(rows[::2], rows[1::2]):
            a, b = deepcopy(first['state']), deepcopy(second['state'])
            self.assertEqual((a['notification'].pop('body'), b['notification'].pop('body')), ('ㅇ', 'ㅁ'))
            self.assertEqual(a, b)
            self.assertEqual((first['score'], second['score']), (1, 1))

    def test_new_reaction_pairs_have_contextual_labels_and_identical_other_fields(self):
        from collections import defaultdict
        groups = defaultdict(list)
        for row in make_drafts():
            if row.get('invariance_group'):
                groups[row['invariance_group']].append(row)
        self.assertEqual(len(groups), 24)
        for pair in groups.values():
            self.assertEqual(len(pair), 2)
            first, second = pair
            a, b = deepcopy(first['state']), deepcopy(second['state'])
            self.assertEqual((a['notification'].pop('body'), b['notification'].pop('body')), ('ㅇ', 'ㅁ'))
            self.assertEqual(a, b)
            self.assertEqual(first['score'], 2 if first['case_kind'] == 'project_room_character' else 1)
            self.assertEqual(first['score'], second['score'])

    def test_ambiguous_current_pair_changes_only_history(self):
        rows = make_drafts()
        by_id = {r['id']: r for r in rows}
        for row in rows:
            if row['case_kind'] != 'generic_current_without_history':
                continue
            original = by_id[row['paired_with']]
            expected = 2 if row['scenario'] == 'login_session' else 1
            self.assertEqual((original['score'], row['score']), (3, expected))
            a, b = deepcopy(original['state']), deepcopy(row['state'])
            self.assertTrue(a['context'].pop('recent_windows'))
            self.assertEqual(b['context'].pop('recent_windows'), [])
            self.assertEqual(a, b)

    def test_user_confirmed_generic_development_window_is_partial_not_unrelated(self):
        rows = [r for r in make_drafts() if r['case_kind'] == 'generic_current_without_history'
                and r['scenario'] == 'login_session']
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn('main.py', row['state']['context']['window_title'])
        self.assertEqual(row['state']['context']['recent_windows'], [])
        self.assertEqual(row['score'], 2)
        self.assertEqual(row['review_status'], 'ai_proposed_unreviewed')

    def test_validation_rejects_duplicate_inputs_and_group_leakage(self):
        rows = make_drafts()
        duplicate = deepcopy(rows[0])
        duplicate['id'] = 'duplicate'
        with self.assertRaisesRegex(ValueError, 'duplicated full'):
            validate_drafts(rows + [duplicate])
        rows[-1]['group_id'] = 'separate_evaluation'
        with self.assertRaisesRegex(ValueError, 'split across'):
            validate_drafts(rows)


if __name__ == '__main__':
    unittest.main()
