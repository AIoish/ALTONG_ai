import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from filtering_training.generation.generate_rapid_dataset import generate as generate_base
from filtering_training.generation.generate_targeted_dataset import generate
from filtering_training.datasets.prepare_dataset import load_samples
from filtering_training.datasets.prepare_rapid_dataset import prepare
from src.filtering.policy import should_pass


class TargetedDatasetTests(unittest.TestCase):
    def test_pairs_preserve_urgency_and_category_and_change_relevance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            generate_base(base, target=240)
            output, combined = root / "added.jsonl", root / "combined.jsonl"
            report = generate(base, output, combined, report_path=root / "audit.json")
            first_bytes = output.read_bytes()
            generate(base, output, combined, report_path=root / "audit.json")
            self.assertEqual(first_bytes, output.read_bytes())
            self.assertEqual(report["added_count"], 2000)
            self.assertEqual(report["added_unique_notification_texts"], 1000)
            self.assertEqual(set(report["added_category_counts"].values()), {250})
            self.assertEqual(report["combined_audit"]["count"], 2240)
            samples = {s.notification.id: s for s in load_samples(output)}
            lineage = json.loads(output.with_suffix(".lineage.json").read_text(encoding="utf-8"))
            pairs = defaultdict(list)
            for entry in lineage["items"]:
                pairs[entry["pair_id"]].append(samples[entry["id"]])
            self.assertEqual(len(pairs), 1000)
            for left, right in pairs.values():
                self.assertEqual(left.notification.title, right.notification.title)
                self.assertEqual(left.notification.body, right.notification.body)
                self.assertEqual(left.label.urgency_score, right.label.urgency_score)
                self.assertEqual(left.label.category, right.label.category)
                self.assertNotEqual(left.label.relevance_score, right.label.relevance_score)
                if left.label.urgency_score >= 4:
                    self.assertTrue(should_pass(right.label.urgency_score, right.label.relevance_score))

    def test_expansion_keeps_previous_evaluation_families_and_context_pairs_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            generate_base(base, target=240)
            previous = prepare(base, root / "old_prepared")
            combined = root / "combined.jsonl"
            generate(base, root / "added.jsonl", combined, report_path=root / "audit.json")
            current = prepare(combined, root / "new_prepared", preserve_manifest=root / "old_prepared" / "manifest.json")
            for name, split in previous["splits"].items():
                self.assertTrue(set(split["notification_ids"]).issubset(current["splits"][name]["notification_ids"]))
                self.assertTrue(set(split["families"]).issubset(current["splits"][name]["families"]))
            families = [set(s["families"]) for s in current["splits"].values()]
            self.assertTrue(all(families[i].isdisjoint(families[j]) for i in range(3) for j in range(i+1,3)))
            split_by_id = {identifier:name for name,info in current["splits"].items() for identifier in info["notification_ids"]}
            pair_splits = defaultdict(set)
            for entry in json.loads(combined.with_suffix(".lineage.json").read_text(encoding="utf-8"))["items"]:
                if "pair_id" in entry:
                    pair_splits[entry["pair_id"]].add(split_by_id[entry["id"]])
            self.assertTrue(all(len(names)==1 for names in pair_splits.values()))


if __name__ == "__main__":
    unittest.main()
