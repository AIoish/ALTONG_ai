"""용도: 수정 점수·사용자 예외 승인 보존과 원본 분할 무변경 검증.
생성일: 2026-10-04
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from filtering_training.generation import generate_v3_seed
from filtering_training.generation.expand_v3_dataset import expand
from filtering_training.generation.generate_score_review_batch import generate
from filtering_training.preparation.prepare_reviewed_v3 import prepare
from filtering_training.preparation.prepare_score_experiment import digest,write_json,write_lines
from filtering_training.preparation.freeze_score_review import freeze
from filtering_training.common.dataset import load_samples


class FreezeScoreReviewTests(unittest.TestCase):
    def test_urgent_boundaries_preserve_context_urgency_and_reject_heldout_links(self):
        from filtering_training.generation import generate_score_review_batch as generator
        before = digest(self.source/"dataset.jsonl")
        output = self.root/"urgent"
        samples = generate(self.source,output,urgent=True)
        self.assertEqual(len(samples),13)
        self.assertEqual(len({s.notification.body for s in samples}),10)
        by_id = {s.notification.id:s for s in samples}
        for number in (1,3,4):
            prefix = f"sep04_{number:02d}"
            self.assertEqual(by_id[prefix+"_empty"].label.urgency_score,by_id[prefix+"_review"].label.urgency_score)
            self.assertEqual(by_id[prefix+"_empty"].label.relevance_score,1)
        self.assertEqual(by_id["sep04_08_review"].label.urgency_score,4)
        self.assertEqual(by_id["sep04_09_review"].label.urgency_score,2)
        self.assertEqual(by_id["sep04_10_review"].label.urgency_score,1)
        manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["human_reviewed_originals"],0)
        self.assertFalse(manifest["training_used"])
        self.assertEqual(digest(self.source/"dataset.jsonl"),before)
        source_manifest=json.loads((self.source/"prepared/manifest.json").read_text(encoding="utf-8"))
        heldout=source_manifest["splits"]["validation"]["message_ids"][0]
        modified=[dict(case) for case in generator.URGENT_CASES]
        modified[0]["train_links"]=[heldout]
        with patch.object(generator,"URGENT_CASES",modified):
            with self.assertRaisesRegex(ValueError,"not train-only"):
                generate(self.source,self.root/"rejected_urgent",urgent=True)
        self.assertFalse((self.root/"rejected_urgent").exists())

    def test_expansion_freeze_and_bulk_keep_review_scope_and_context_independence(self):
        from collections import defaultdict
        from filtering_training.generation.generate_score_review_batch import generate_bulk, revise_bulk, expand_urgent_states
        preview=self.root/"expansion_preview"
        generate(self.source,preview,expansion=True)
        approved=self.root/"expansion_approved"
        freeze(self.source,preview,approved,"synthetic fixture preview approval")
        original_hash=digest(self.source/"dataset.jsonl")
        output=self.root/"bulk"
        samples=generate_bulk(self.source,output,approved)
        self.assertEqual(len(samples),5500)
        self.assertEqual(len({s.notification.body for s in samples}),5000)
        groups=defaultdict(list)
        for s in samples:
            groups[s.notification.id.rsplit("_",1)[0] if s.notification.id.endswith("primary") else s.notification.id.rsplit("_",2)[0]].append(s)
        self.assertEqual(len(groups),5000)
        paired=[g for g in groups.values() if len(g)==2]
        self.assertEqual(len(paired),500)
        for group in paired:
            self.assertEqual(len({s.label.urgency_score for s in group}),1)
            self.assertEqual(len({s.label.category for s in group}),1)
            self.assertEqual(len({(s.context.active_process,s.context.window_title) for s in group}),2)
        manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["human_reviewed_originals_in_bulk"],0)
        self.assertEqual(manifest["approved_preview_originals"],8)
        self.assertFalse(manifest["training_used"])
        self.assertEqual(digest(self.source/"dataset.jsonl"),original_hash)
        lineage = [json.loads(v) for v in (output/"lineage.jsonl").read_text(encoding="utf-8").splitlines()]
        by_id = {s.notification.id: s for s in samples}
        feedback = {"source_sha256":digest(output/"candidates.jsonl"),"approved_cases":[],"row_overrides":[]}
        for family, value in (("coupon_ad",2),("near_close",5),("live_access",5)):
            case = next(info for info in lineage if info["event_family"]==family)
            sample = by_id[case["notification_id"]]
            feedback["approved_cases"].append({"notification_id":sample.notification.id,"field":"urgency_score","approved_value":value})
            for info in lineage:
                if info["original_id"]==case["original_id"]:
                    sibling = by_id[info["notification_id"]]
                    feedback["row_overrides"].append({"notification_id":sibling.notification.id,"field":"urgency_score",
                        "candidate_value":sibling.label.urgency_score,"approved_value":value,"approval_scope":"fixture"})
        path = self.root/"bulk_feedback.json"
        write_json(path,feedback)
        before = digest(output/"candidates.jsonl")
        revised = revise_bulk(output,self.root/"bulk_revision",path)
        self.assertEqual({s.notification.id for s in revised},set(by_id))
        self.assertEqual(digest(output/"candidates.jsonl"),before)
        revised_manifest=json.loads((self.root/"bulk_revision/manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(revised_manifest["quality_gate"]["ready_for_training"])
        self.assertEqual(revised_manifest["human_reviewed_originals_in_bulk"],0)
        self.assertGreater(revised_manifest["grammar_changed_rows"],0)
        families={info["notification_id"]:info["event_family"] for info in lineage}
        for sample in revised:
            if families[sample.notification.id] in ("near_close","live_access"):
                self.assertEqual(sample.label.urgency_score,5)
            if families[sample.notification.id]=="coupon_ad":
                self.assertEqual(sample.label.urgency_score,2)
        feedback["row_overrides"][0]["candidate_value"]=5
        write_json(path,feedback)
        with self.assertRaisesRegex(ValueError,"does not match source"):
            revise_bulk(output,self.root/"rejected_revision",path)
        self.assertFalse((self.root/"rejected_revision").exists())
        urgent_review=self.root/"urgent_review"
        generate(self.source,urgent_review,urgent=True)
        urgent_approved=self.root/"urgent_approved"
        freeze(self.source,urgent_review,urgent_approved,"fixture approval")
        expanded_root=self.root/"urgent_expanded_pool"
        combined=expand_urgent_states(self.root/"bulk_revision",urgent_approved,expanded_root,original_source=self.source)
        self.assertEqual(len(combined),5753)
        expanded_manifest=json.loads((expanded_root/"manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(expanded_manifest["originals"],5090)
        self.assertEqual(expanded_manifest["added_synthetic_originals"],80)
        self.assertFalse(expanded_manifest["training_used"])
        self.assertFalse(expanded_manifest["quality_gate"]["ready_for_training"])
        self.assertEqual(digest(output/"candidates.jsonl"),before)
        self.assertFalse(expanded_manifest["audit"]["contradictory_notification_groups"])
        expanded_lineage=[json.loads(v) for v in (expanded_root/"lineage.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(info["review_status"]=="human_approved" for info in expanded_lineage),13)
        new_rows=[s for s in combined if s.notification.id.startswith("state01_")]
        self.assertEqual(len(new_rows),240)
        self.assertEqual(len({s.notification.title for s in new_rows}),20)
        self.assertEqual(len({s.context.window_title for s in new_rows if s.context.active_process=="chrome.exe"}),20)
        for stage in (2,3,4,5):
            self.assertEqual(sum(s.label.urgency_score==stage for s in new_rows),60)
        self.assertEqual(digest(self.source/"dataset.jsonl"),original_hash)
        approval=approved/"approval.json"
        approval.write_text("{}",encoding="utf-8")
        with self.assertRaises((KeyError,ValueError)):
            generate_bulk(self.source,self.root/"tampered",approved)
        self.assertFalse((self.root/"tampered").exists())

    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        with patch.object(sys,"argv",["seed","--output-dir",str(self.root/"seed")]):
            with contextlib.redirect_stdout(io.StringIO()):
                generate_v3_seed.main()
        expand(self.root/"seed",self.root/"expanded","synthetic fixture approval","2026-10-03")
        self.source=self.root/"source"
        prepare(self.root/"expanded",self.source,"synthetic fixture approval","2026-10-03")
        self.revised=self.root/"review"
        generate(self.source,self.revised)
        rows=[json.loads(line) for line in (self.revised/"candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in rows:
            identifier=row["notification"]["id"]
            if identifier=="sep02_06_related":row["label"]["relevance_score"]=2
            if identifier=="sep02_07_related":row["label"]["relevance_score"]=4
        write_lines(self.revised/"candidates.jsonl",rows)
        write_json(self.revised/"feedback.json",{"user_statement":"synthetic fixture score correction",
            "guideline_conflict_ids":["sep02_07_related"],"corrections":[
                {"notification_id":"sep02_06_related","field":"relevance_score","after":2},
                {"notification_id":"sep02_07_related","field":"relevance_score","after":4}]})
        self.refresh_manifest()

    def refresh_manifest(self):
        path=self.revised/"manifest.json"
        m=json.loads(path.read_text(encoding="utf-8"))
        m.update({"candidate_sha256":digest(self.revised/"candidates.jsonl"),
                  "feedback_sha256":digest(self.revised/"feedback.json")})
        write_json(path,m)

    def test_freeze_preserves_score_override_and_source_hash(self):
        original=digest(self.source/"dataset.jsonl")
        original_manifest=digest(self.source/"prepared/manifest.json")
        freeze(self.source,self.revised,self.root/"frozen","synthetic fixture final approval")
        by_id={s.notification.id:s for s in load_samples(self.root/"frozen/dataset.jsonl")}
        self.assertEqual(by_id["sep02_06_related"].label.relevance_score,2)
        self.assertEqual(by_id["sep02_07_related"].label.relevance_score,4)
        approval=json.loads((self.root/"frozen/approval.json").read_text(encoding="utf-8"))
        self.assertFalse(approval["global_prompt_changed"])
        self.assertEqual(approval["guideline_override_ids"],["sep02_07_related"])
        self.assertFalse(approval["training_used"])
        self.assertEqual(digest(self.source/"dataset.jsonl"),original)
        self.assertEqual(digest(self.source/"prepared/manifest.json"),original_manifest)
        with self.assertRaisesRegex(ValueError,"overwrite"):
            freeze(self.source,self.revised,self.root/"frozen","synthetic fixture final approval")

    def test_inconsistent_feedback_is_rejected_before_freezing(self):
        rows=[json.loads(line) for line in (self.revised/"candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["notification"]["id"]=="sep02_07_related":row["label"]["relevance_score"]=3
        write_lines(self.revised/"candidates.jsonl",rows)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError,"explicit user corrections"):
            freeze(self.source,self.revised,self.root/"rejected","synthetic fixture final approval")
        self.assertFalse((self.root/"rejected").exists())


if __name__=="__main__":
    unittest.main()
