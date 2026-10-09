"""용도: 승인 점수·학습 전용 보강·검증 보존과 자료 변경 차단을 검사한다.
생성일: 2026-10-04
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from filtering_training.preparation import prepare_digit_supplement as module


@unittest.skipUnless(
    Path("filtering_training/outputs/v3_reviewed_01/dataset.jsonl").is_file(),
    "local reviewed experiment data is not distributed through Git",
)
class SupplementTests(unittest.TestCase):
    def test_history_draft_preserves_approved_scores_and_context_separation(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):
                return {"input_ids":[1]*10}
        pool=Path("filtering_training/outputs/v5_training_pool_01")
        validation=Path("filtering_training/outputs/v5_validation_pool_01")
        baseline=pool.parent/"v4_training_pool_01/prepared"
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"prepared"
            report=module.prepare_history_pool(pool,validation,output,Tokenizer())
            self.assertFalse(report["relevance"]["training_ready"])
            for task in ("urgency","relevance"):
                self.assertEqual((output/task/"validation.jsonl").read_bytes(),(baseline/task/"validation.jsonl").read_bytes())
            rows=[json.loads(line) for line in (output/"relevance/train.jsonl").read_text(encoding="utf-8").splitlines()]
            by_id={json.loads(r["messages"][1]["content"])["notification"]["id"]:r for r in rows}
            case4=by_id["train05_101_joke_focus_matching"]
            case5=by_id["train05_131_color_option_matching"]
            self.assertEqual(json.loads(case4["messages"][-1]["content"])["relevance_score"],4)
            self.assertEqual(json.loads(case5["messages"][-1]["content"])["relevance_score"],3)
            self.assertEqual(len(json.loads(case4["messages"][1]["content"])["context"]["recent_windows"]),3)
            urgent=[json.loads(line) for line in (output/"urgency/train.jsonl").read_text(encoding="utf-8").splitlines()]
            for row in urgent:
                payload=json.loads(row["messages"][1]["content"])
                self.assertNotIn("context",payload)
                self.assertNotIn("id",payload["notification"])

    def test_history_draft_rejects_token_overflow_before_writing(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):
                return {"input_ids":[1]*769}
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"prepared"
            with self.assertRaisesRegex(ValueError,"token limit"):
                module.prepare_history_pool(Path("filtering_training/outputs/v5_training_pool_01"),
                    Path("filtering_training/outputs/v5_validation_pool_01"),output,Tokenizer())
            self.assertFalse(output.exists())

    def test_pool_cap_preserves_seeds_and_validation_without_context_leakage(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):
                return {"input_ids":[1]*10}
        pool=Path("filtering_training/outputs/v4_training_pool_01")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"prepared"
            report=module.prepare_pool(pool,root,tokenizer=Tokenizer())
            self.assertLessEqual(report["urgency"]["sampling"]["urgency5_digital_fraction"],.25)
            self.assertGreater(report["urgency"]["train_rows"],4000)
            self.assertGreater(report["relevance"]["train_rows"],5700)
            for task in ("urgency","relevance"):
                self.assertEqual((root/task/"validation.jsonl").read_bytes(),(module.OUTPUT/task/"validation.jsonl").read_bytes())
                old=json.loads((module.OUTPUT/task/"manifest.json").read_text(encoding="utf-8"))
                new=json.loads((root/task/"manifest.json").read_text(encoding="utf-8"))
                self.assertTrue(set(old["splits"]["train"]["notification_ids"]).issubset(new["splits"]["train"]["notification_ids"]))
                self.assertFalse(new["training_used"])
                self.assertEqual(new["sampling"]["required_mode"],"natural")
            for row in map(json.loads,(root/"urgency/train.jsonl").read_text(encoding="utf-8").splitlines()):
                payload=json.loads(row["messages"][1]["content"])
                self.assertNotIn("context",payload)
                self.assertNotIn("id",payload["notification"])

    def test_pool_rejects_overlong_prompts_before_creating_output(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):
                return {"input_ids":[1]*769}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"prepared"
            with self.assertRaisesRegex(ValueError,"token limit"):
                module.prepare_pool(Path("filtering_training/outputs/v4_training_pool_01"),root,tokenizer=Tokenizer())
            self.assertFalse(root.exists())

    def test_preserves_validation_and_original_train_with_user_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"prepared"
            module.prepare(root)
            for task,count,added in (("urgency",58,8),("relevance",174,24)):
                self.assertEqual((root/task/"validation.jsonl").read_bytes(),(module.PREPARED/task/"validation.jsonl").read_bytes())
                before = (module.PREPARED/task/"train.jsonl").read_text(encoding="utf-8").splitlines()
                after = (root/task/"train.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual([json.loads(v) for v in before],[json.loads(v) for v in after[:-added]])
                self.assertEqual(len(after),count)
            records = [json.loads(v) for v in (root/"relevance/train.jsonl").read_text(encoding="utf-8").splitlines()[-24:]]
            by_id = {json.loads(r["messages"][1]["content"])["notification"]["id"]:r for r in records}
            self.assertEqual(json.loads(by_id["sep02_06_related"]["messages"][-1]["content"])["relevance_score"],2)
            self.assertEqual(json.loads(by_id["sep02_07_related"]["messages"][-1]["content"])["relevance_score"],4)
            self.assertTrue(all("context" not in json.loads(json.loads(v)["messages"][1]["content"]) for v in
                (root/"urgency/train.jsonl").read_text(encoding="utf-8").splitlines()))

    def test_rejects_tampered_approved_data_before_output(self):
        real_digest = module.digest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"prepared"
            with patch.object(module,"digest",side_effect=lambda p:"tampered" if p == module.APPROVED/"dataset.jsonl" else real_digest(p)):
                with self.assertRaises(AssertionError):
                    module.prepare(root)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
