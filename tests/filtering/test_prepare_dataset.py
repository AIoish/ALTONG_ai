import json
import tempfile
import unittest
from pathlib import Path

from filtering_training.datasets.prepare_dataset import (
    DATASET_PATH,
    load_samples,
    prepare_dataset,
)
from src.filtering.prompt import parse_model_output


class PrepareDatasetTests(unittest.TestCase):
    def test_context_variant_stays_in_one_split_and_sft_output_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = prepare_dataset(DATASET_PATH, output)
            split_for_id = {
                notification_id: name
                for name, info in manifest["splits"].items()
                for notification_id in info["notification_ids"]
            }
            self.assertEqual(len(split_for_id), len(load_samples(DATASET_PATH)))
            self.assertEqual(split_for_id["noti_031"], split_for_id["noti_032"])
            samples_by_id = {s.notification.id: s for s in load_samples(DATASET_PATH)}
            for name in ("validation", "test"):
                self.assertTrue(any(
                    samples_by_id[identifier].label.urgency_score >= 4
                    for identifier in manifest["splits"][name]["notification_ids"]
                ))
            self.assertEqual(manifest["source_sha256"],
                             prepare_dataset(DATASET_PATH, output)["source_sha256"])

            for name, info in manifest["splits"].items():
                records = [json.loads(line) for line in (output / f"{name}.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()]
                self.assertEqual(len(records), info["count"])
                for record in records:
                    messages = record["messages"]
                    self.assertEqual([m["role"] for m in messages],
                                     ["system", "user", "assistant"])
                    parse_model_output(messages[-1]["content"])
                    user_payload = json.loads(messages[1]["content"])
                    self.assertIn("duration_seconds", user_payload["context"])
                    self.assertIn("recent_processes", user_payload["context"])
                    self.assertNotIn("label", user_payload)

    def test_duplicate_ids_are_rejected(self) -> None:
        first_line = DATASET_PATH.read_text(encoding="utf-8").splitlines()[0]
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "duplicates.jsonl"
            dataset.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate notification id"):
                load_samples(dataset)


if __name__ == "__main__":
    unittest.main()
