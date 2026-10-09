"""용도: 숫자 토큰 진단 범위와 새 검수 묶음의 문맥·보류 그룹 격리 검증.
생성일: 2026-10-03
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from filtering_training.generation.generate_score_review_batch import build_candidates, generate, CASES
from filtering_training.quality.audit_score_learning import numeric_spans
from filtering_training.generation import generate_v3_seed
from filtering_training.generation.expand_v3_dataset import expand
from filtering_training.preparation.prepare_reviewed_v3 import prepare
from filtering_training.preparation.prepare_score_experiment import digest
from src.filtering.policy import should_pass


class ScoreLearningReviewTests(unittest.TestCase):
    def test_expansion_preview_covers_boundaries_and_context_invariance(self):
        from collections import Counter
        from filtering_training.common.score_tasks import unique_urgency_samples
        samples = build_candidates(expansion=True)
        self.assertEqual(len(samples),10)
        self.assertEqual(len(unique_urgency_samples(samples)),8)
        self.assertEqual(Counter(s.label.urgency_score for s in samples),Counter({i:2 for i in range(1,6)}))
        self.assertEqual(set(s.label.relevance_score for s in samples),set(range(1,6)))
        self.assertEqual(next(s.label.relevance_score for s in samples if s.notification.id=="sep03_02_related"),2)
        self.assertEqual(next(s.label.relevance_score for s in samples if s.notification.id=="sep03_01_related"),3)

    def test_numeric_span_selects_value_not_digits_in_field_or_other_text(self):
        text='{"urgency_score":5,"reason":"4 and 3 are unrelated"}'
        spans=numeric_spans(text,"urgency_score")
        self.assertEqual([text[a:b] for a,b in spans],["5"])
        self.assertEqual(numeric_spans(text,"relevance_score"),[])
        self.assertEqual(numeric_spans('{"urgency_score":50}',"urgency_score"),[])

    def test_context_variants_preserve_urgency_and_useful_vs_topical_labels(self):
        samples=build_candidates()
        self.assertEqual(len(samples),24)
        self.assertEqual(len({s.notification.id for s in samples}),24)
        for index,case in enumerate(CASES):
            variants=samples[index*3:index*3+3]
            self.assertEqual({(s.label.urgency_score,s.label.category) for s in variants},
                             {(case["urgency"],case["category"])})
            self.assertEqual([s.label.relevance_score for s in variants], [case["relevance"],1,1])
            self.assertEqual(variants[2].context.window_title,"")
            self.assertEqual(variants[2].context.duration_seconds,0)
            if case["urgency"] >= 4:
                self.assertTrue(all(should_pass(s.label.urgency_score,s.label.relevance_score) for s in variants))
        # The same populated study window receives helpful, advertising and joke messages.
        self.assertEqual(samples[9].context,samples[15].context)
        self.assertEqual(samples[9].context,samples[18].context)
        self.assertEqual([samples[i].label.relevance_score for i in (9,15,18)], [4,3,3])

    def test_generation_freezes_candidates_and_rejects_link_to_heldout_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(sys,"argv",["seed","--output-dir",str(root/"seed")]):
                with contextlib.redirect_stdout(io.StringIO()):
                    generate_v3_seed.main()
            expand(root/"seed",root/"expanded","synthetic test fixture approval","2026-10-03")
            prepare(root/"expanded",root/"source","synthetic test fixture approval","2026-10-03")
            original_hash=digest(root/"source/dataset.jsonl")
            generate(root/"source",root/"review")
            manifest=json.loads((root/"review/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"],"unapproved_synthetic_candidates")
            self.assertEqual(manifest["populated_window_overlap"],0)
            self.assertFalse((root/"review/approval.json").exists())
            self.assertEqual(digest(root/"source/dataset.jsonl"),original_hash)
            with self.assertRaisesRegex(ValueError,"overwrite"):
                generate(root/"source",root/"review")
            path=root/"source/prepared/manifest.json"
            m=json.loads(path.read_text(encoding="utf-8"))
            m["splits"]["train"]["message_ids"].remove("v3_37")
            path.write_text(json.dumps(m),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"train-only"):
                generate(root/"source",root/"rejected")
            self.assertFalse((root/"rejected").exists())


if __name__=="__main__":
    unittest.main()
