import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from filtering_training.compare_evaluations import compare
from filtering_training.evaluate import score_predictions
from filtering_training.prepare_dataset import load_samples


class CompareEvaluationsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = Path("filtering_training/data/sample_notifications.jsonl")
        self.samples = load_samples(self.dataset)

    def write_run(self, name, predictions=None, digest=None, prompt="fixed"):
        predictions = predictions if predictions is not None else [s.label for s in self.samples]
        report = {"model": name, "adapter": None,
                  "dataset_sha256": digest or hashlib.sha256(self.dataset.read_bytes()).hexdigest(),
                  "prompt_sha256": prompt,
                  "metrics": score_predictions([s.label for s in self.samples], predictions)}
        records = {"prediction_examples": [
            {"notification_id": s.notification.id, "model_output": p.model_dump() if p else None}
            for s, p in zip(self.samples, predictions)]}
        paths = self.root / f"{name}.json", self.root / f"{name}_predictions.json"
        for path, value in zip(paths, (report, records)):
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return paths

    def test_invalid_urgent_output_is_reported_without_hiding_policy_error(self):
        predictions = [s.label for s in self.samples]
        index = next(i for i, s in enumerate(self.samples) if s.label.urgency_score >= 4)
        predictions[index] = None
        result = compare(self.dataset, [self.write_run("candidate", predictions)])
        run = result["runs"][0]
        self.assertEqual(run["urgent_invalid_json_count"], 1)
        self.assertEqual(run["category_correct"], len(self.samples) - 1)
        self.assertTrue(run["errors"][0]["policy_error"])

    def test_rejects_dataset_or_prompt_mismatch(self):
        first = self.write_run("first")
        with self.assertRaisesRegex(ValueError, "frozen dataset"):
            compare(self.dataset, [self.write_run("wrong", digest="other")])
        with self.assertRaisesRegex(ValueError, "different prompts"):
            compare(self.dataset, [first, self.write_run("other_prompt", prompt="changed")])

    def test_rejects_incomplete_predictions_and_inconsistent_metrics(self):
        report, predictions = self.write_run("candidate")
        original = predictions.read_text(encoding="utf-8")
        value = json.loads(original)
        value["prediction_examples"].pop()
        predictions.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "every dataset ID"):
            compare(self.dataset, [(report, predictions)])
        predictions.write_text(original, encoding="utf-8")
        value = json.loads(report.read_text(encoding="utf-8"))
        value["metrics"]["policy_accuracy"] = 0
        report.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metrics disagree"):
            compare(self.dataset, [(report, predictions)])
