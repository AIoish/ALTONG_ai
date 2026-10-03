import unittest

from filtering_training.evaluate import score_predictions
from src.filtering.schema import FilterLabel


def label(urgency: int, relevance: int, category: str = "일반 업무") -> FilterLabel:
    return FilterLabel(
        urgency_score=urgency,
        relevance_score=relevance,
        category=category,
        ai_summary_reason="합성 테스트",
    )


class EvaluationMetricTests(unittest.TestCase):
    def test_relevant_low_urgency_alert_is_a_gold_pass(self) -> None:
        report = score_predictions([label(2, 4), label(1, 1)], [label(2, 3), label(1, 1)])
        self.assertEqual(report["policy_accuracy"], 0.5)
        self.assertEqual(report["gold_block_count"], 1)
        self.assertEqual(report["unnecessary_false_pass_rate"], 0)

    def test_urgent_miss_and_unnecessary_pass_are_separate(self) -> None:
        gold = [label(5, 1), label(1, 1), label(3, 4)]
        predicted = [label(2, 1), label(4, 1), None]
        report = score_predictions(gold, predicted)
        self.assertEqual(report["json_valid_rate"], 2 / 3)
        self.assertEqual(report["policy_accuracy"], 0)
        self.assertEqual(report["urgent_false_block_rate"], 1)
        self.assertEqual(report["unnecessary_false_pass_rate"], 1)
        self.assertEqual(report["urgency_score"]["mae_on_valid_json"], 3.0)

    def test_all_invalid_json_keeps_metrics_defined(self) -> None:
        report = score_predictions([label(5, 1)], [None])
        self.assertEqual(report["json_valid_rate"], 0)
        self.assertIsNone(report["urgency_score"]["mae_on_valid_json"])
        self.assertIsNone(report["category_macro_f1_on_valid_json"])
        self.assertEqual(report["policy_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
