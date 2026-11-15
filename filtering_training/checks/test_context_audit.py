"""용도: 맥락 민감도 진단의 단일 요인 변경과 변형 입력 정답 비전파를 검증한다.
생성일: 2026-11-13
"""

from copy import deepcopy
import unittest

from filtering_training.quality.audit_relevance_context import make_variants, summarize
from filtering_training.quality.audit_relevance_reactions import authored_contexts


class ContextAuditTests(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'sample', 'score': 1, 'experiment': {'origin': 'pilot'},
                    'state': authored_contexts()[2]['state'], 'case_kind': 'current_recent_conflict'}

    def test_only_selected_field_changes_and_source_preserved(self):
        before = deepcopy(self.row)
        variants = make_variants([self.row])
        self.assertEqual(before, self.row)
        for v in variants:
            expected = deepcopy(before['state'])
            if v['variant'].startswith('app:'):
                expected['notification']['app_name'] = v['variant'][4:]
            elif v['variant'] == 'without_recent':
                expected['context']['recent_windows'] = []
            self.assertEqual(expected, v['state'])
            self.assertNotIn('score', v)
        self.assertEqual(len({v['input_sha256'] for v in variants}), len(variants))

    def test_no_op_history_is_not_a_comparison(self):
        self.row['state']['context']['recent_windows'] = []
        variants = make_variants([self.row])
        self.assertNotIn('without_recent', {v['variant'] for v in variants})
        report, _ = summarize([self.row], variants, [1] * len(variants))
        self.assertEqual(report['history_eligible_rows'], 0)

    def test_transformed_labels_not_used_as_accuracy(self):
        variants = make_variants([self.row])
        scores = [1 if v['variant'] == 'original' else 3 for v in variants]
        report, details = summarize([self.row], variants, scores)
        self.assertEqual(report['original_validation']['exact_accuracy'], 1)
        self.assertEqual(report['app_changed_rows'], 1)
        self.assertEqual(report['history_changed_rows'], 1)
        self.assertTrue(details[0]['original_correct'])
        self.assertNotIn('transformed_accuracy', report)

    def test_coverage_and_invalid_scores_rejected(self):
        variants = make_variants([self.row])
        for scores in ([1], [True] * len(variants), [0] * len(variants)):
            with self.assertRaises(ValueError):
                summarize([self.row], variants, scores)
        with self.assertRaises(ValueError):
            summarize([self.row], variants[1:], [1] * (len(variants) - 1))
        with self.assertRaises(ValueError):
            make_variants([self.row, self.row])


if __name__ == '__main__':
    unittest.main()
