import tempfile
import unittest
from pathlib import Path

from filtering_training.evaluate import load_split_samples
from filtering_training.generation.generate_rapid_dataset import generate
from filtering_training.datasets.prepare_dataset import DATASET_PATH, load_samples
from filtering_training.datasets.prepare_holdout import prepare_holdout, validate_holdout


class HoldoutPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.holdout_path = self.root / "holdout.jsonl"
        generate(self.holdout_path, target=24, holdout=self.root / "missing.jsonl")

    def test_holdout_is_independent_and_covers_categories(self) -> None:
        holdout = load_samples(self.holdout_path)
        training = load_samples(DATASET_PATH)
        report = validate_holdout(holdout, training)
        self.assertEqual(report["count"], 24)
        self.assertEqual(report["distinct_notification_texts"], 24)
        self.assertEqual(set(report["category_counts"].values()), {3})
        self.assertTrue(all(len(item.context.recent_processes) <= 3 for item in holdout))
        self.assertGreaterEqual(len({item.notification.app_name for item in holdout}), 8)

    def test_overlapping_input_is_rejected(self) -> None:
        training = load_samples(DATASET_PATH)
        holdout = load_samples(self.holdout_path)
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            validate_holdout([training[0]], training)
        copied = training[0].model_copy(deep=True)
        copied.notification.id = "holdout_new_id"
        with self.assertRaisesRegex(ValueError, "text overlaps"):
            validate_holdout([copied], training)
        self.assertEqual(len(holdout), 24)

    def test_manifest_contains_test_split_only(self) -> None:
        manifest = prepare_holdout(holdout_path=self.holdout_path,
                                   output_dir=self.root / "prepared")
        self.assertEqual(set(manifest["splits"]), {"test"})
        self.assertEqual(manifest["splits"]["test"]["count"], 24)
        self.assertEqual(len(manifest["splits"]["test"]["notification_ids"]), 24)
        self.assertEqual(manifest["review_status"], "pending independent label review")
        evaluated = load_split_samples(self.holdout_path, self.root / "prepared", "test")
        self.assertEqual(len(evaluated), 24)


if __name__ == "__main__":
    unittest.main()