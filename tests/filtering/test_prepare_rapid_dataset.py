import json
import tempfile
import unittest
from pathlib import Path

from filtering_training.generate_rapid_dataset import generate
from filtering_training.prepare_rapid_dataset import prepare


class PrepareRapidDatasetTests(unittest.TestCase):
    def test_scenario_families_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "rapid.jsonl"
            generate(dataset, target=240)
            manifest = prepare(dataset, root / "prepared")
            self.assertEqual(sum(split["count"] for split in manifest["splits"].values()), 240)
            family_sets = [set(split["families"]) for split in manifest["splits"].values()]
            self.assertTrue(all(family_sets[i].isdisjoint(family_sets[j])
                                for i in range(3) for j in range(i + 1, 3)))
            for split in manifest["splits"].values():
                self.assertTrue(all(count > 0 for count in split["category_counts"].values()))
            for name, split in manifest["splits"].items():
                records = (root / "prepared" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(records), split["count"])
                self.assertEqual(len(json.loads(records[0])["messages"]), 3)


if __name__ == "__main__":
    unittest.main()
