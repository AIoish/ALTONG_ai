import json
import tempfile
import unittest
from pathlib import Path

from filtering_training.generate_rapid_dataset import generate
from filtering_training.prepare_dataset import DATASET_PATH, load_samples
from filtering_training.prepare_holdout import notification_key
from src.filtering.prompt import CATEGORIES


class RapidDatasetGenerationTests(unittest.TestCase):
    def test_deterministic_balanced_candidates_and_lineage(self):
        seed = load_samples(DATASET_PATH)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.jsonl"
            holdout = root / "holdout.jsonl"
            training.write_text(seed.model_dump_json() + "\n", encoding="utf-8")
            holdout.write_text(seed.model_dump_json() + "\n", encoding="utf-8")
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            report = generate(first, target=80, training=training, holdout=holdout)
            generate(second, target=80, training=training, holdout=holdout)
            samples = load_samples(first)
            lineage = json.loads(first.with_suffix(".lineage.json").read_text(encoding="utf-8"))
            self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(report["count"], 80)
        self.assertEqual(report["distinct_notification_texts"], 80)
        self.assertEqual(set(report["category_counts"]), set(CATEGORIES))
        self.assertEqual(set(report["category_counts"].values()), {10})
        self.assertEqual(len(lineage["items"]), 80)
        self.assertEqual(len({notification_key(item) for item in samples}), 80)
        self.assertTrue(all(len(item.context.recent_processes) <= 3 for item in samples))


    def test_generation_without_local_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "candidates.jsonl"
            report = generate(output, target=80, holdout=root / "missing.jsonl")
            self.assertEqual(report["count"], 80)
            self.assertEqual(len(load_samples(output)), 80)

if __name__ == "__main__":
    unittest.main()