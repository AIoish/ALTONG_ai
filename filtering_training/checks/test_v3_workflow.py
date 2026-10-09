"""Verify v3 split isolation and preservation of frozen approval records.
Created: 2026-10-03 (Asia/Seoul).
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from filtering_training.common.dataset import load_samples
from filtering_training.generation import generate_v3_seed
from filtering_training.generation.expand_v3_dataset import expand
from filtering_training.preparation.prepare_reviewed_v3 import SEMANTIC_FAMILIES, prepare


class ReviewedV3WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.seed = self.root / "seed"
        self.expansion = self.root / "expansion"
        self.reviewed = self.root / "reviewed"
        with patch.object(sys, "argv", ["generate_v3_seed", "--output-dir", str(self.seed)]):
            with contextlib.redirect_stdout(io.StringIO()):
                generate_v3_seed.main()
        expand(self.seed, self.expansion, "synthetic test fixture approval", "2026-10-03")

    def freeze(self):
        return prepare(self.expansion, self.reviewed, "synthetic test fixture approval", "2026-10-03")

    def test_variants_windows_and_authored_families_stay_in_one_split(self):
        manifest = self.freeze()
        self.assertEqual({name: info["count"] for name, info in manifest["splits"].items()},
                         {"train": 150, "validation": 24, "test": 24})
        self.assertEqual(manifest["human_reviewed_notification_count"], 24)
        original = load_samples(self.expansion / "candidates.jsonl")
        frozen = load_samples(self.reviewed / "dataset.jsonl")
        self.assertEqual([s.model_dump() for s in frozen[:192]], [s.model_dump() for s in original])
        by_id = {s.notification.id: s for s in frozen}
        owner = {}
        windows = defaultdict(set)
        covered = set()
        for split, info in manifest["splits"].items():
            for identifier in info["notification_ids"]:
                self.assertNotIn(identifier, covered)
                covered.add(identifier)
                message = identifier.rsplit("_", 1)[0]
                self.assertEqual(owner.setdefault(message, split), split)
                context = by_id[identifier].context
                if context.active_process:
                    windows[split].add((context.active_process, context.window_title))
        self.assertEqual(covered, set(by_id))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(windows[left] & windows[right])
        for family in SEMANTIC_FAMILIES.values():
            self.assertEqual(len({owner[message] for message in family}), 1)
        approval = json.loads((self.reviewed / "approval.json").read_text(encoding="utf-8"))
        self.assertEqual(approval["remaining_notification_count"], 42)

    def test_changed_review_source_is_rejected_before_output(self):
        source = self.expansion / "candidates.jsonl"
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source or policy changed"):
            self.freeze()
        self.assertFalse(self.reviewed.exists())

    def test_existing_seed_and_frozen_approval_are_not_overwritten(self):
        seed_bytes = (self.seed / "candidates.jsonl").read_bytes()
        with patch.object(sys, "argv", ["generate_v3_seed", "--output-dir", str(self.seed)]):
            with self.assertRaisesRegex(ValueError, "empty output directory"):
                generate_v3_seed.main()
        self.assertEqual((self.seed / "candidates.jsonl").read_bytes(), seed_bytes)
        self.freeze()
        approval_bytes = (self.reviewed / "approval.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "output already exists"):
            self.freeze()
        self.assertEqual((self.reviewed / "approval.json").read_bytes(), approval_bytes)


if __name__ == "__main__":
    unittest.main()
