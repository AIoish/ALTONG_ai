import unittest

from filtering_training.datasets.audit_dataset import audit_samples
from filtering_training.datasets.prepare_dataset import DATASET_PATH, load_samples


class DatasetAuditTests(unittest.TestCase):
    def test_report_counts_and_context_variants(self) -> None:
        samples = load_samples(DATASET_PATH)
        report = audit_samples(samples)
        self.assertEqual(report["count"], len(samples))
        self.assertEqual(sum(report["category_counts"].values()), len(samples))
        self.assertEqual(sum(report["policy_counts"].values()), len(samples))
        self.assertGreaterEqual(report["notification_variant_groups"], 1)
        self.assertEqual(report["contradictory_notification_groups"], [])
        self.assertEqual(report["conflicting_identical_inputs"], [])

    def test_same_notification_with_conflicting_urgency_is_flagged(self) -> None:
        sample = load_samples(DATASET_PATH)[0]
        other = sample.model_copy(deep=True)
        other.notification.id = "different_id"
        other.label.urgency_score = 1
        report = audit_samples([sample, other])
        self.assertEqual(report["contradictory_notification_groups"],
                         [[sample.notification.id, "different_id"]])
        self.assertEqual(report["conflicting_identical_inputs"],
                         [[sample.notification.id, "different_id"]])


if __name__ == "__main__":
    unittest.main()
