import unittest

from src.filtering.policy import decision_label, should_pass


class FilteringPolicyTests(unittest.TestCase):
    def test_all_score_combinations(self):
        # Rows are urgency 1..5; columns are relevance 1..5.
        expected = (
            (False, False, False, True, True),
            (False, False, False, True, True),
            (False, False, False, True, True),
            (True, True, True, True, True),
            (True, True, True, True, True),
        )
        for urgency, row in enumerate(expected, 1):
            for relevance, passed in enumerate(row, 1):
                with self.subTest(urgency=urgency, relevance=relevance):
                    self.assertEqual(should_pass(urgency, relevance), passed)
                    self.assertEqual(decision_label(urgency, relevance), "PASS" if passed else "BLOCK")

    def test_invalid_scores_are_rejected(self):
        for urgency, relevance in ((0, 4), (6, 4), (4, 0), (4, 6)):
            with self.subTest(urgency=urgency, relevance=relevance):
                with self.assertRaises(ValueError):
                    should_pass(urgency, relevance)
