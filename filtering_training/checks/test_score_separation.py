"""용도: 문맥·ID 누출 차단, 잘못된 점수 출력 및 보류 그룹 격리 검증.
생성일: 2026-10-03
"""

import json
import unittest
from pathlib import Path
import tempfile

from filtering_training.common.score_tasks import messages, parse_scores, score_metrics, unique_urgency_samples
from filtering_training.preparation.prepare_score_experiment import quarantined_groups
from src.filtering.schema import FilteringSample
from filtering_training.modeling.compare_score_experiment import compare
from filtering_training.preparation.prepare_score_experiment import write_json, write_lines, digest
from filtering_training.common.score_tasks import prompt_digest
from src.filtering.policy import POLICY_VERSION


def sample(identifier="case_related", urgency=5, relevance=1, window="secret-context"):
    return FilteringSample.model_validate({
        "notification": {"id": identifier, "app_name": "chat", "sender": "sender", "title": "title",
                         "body": "immediate action", "timestamp": "2026-10-03T12:00:00Z"},
        "context": {"active_process": "editor.exe" if window else "", "window_title": window,
                    "last_updated": "2026-10-03T12:00:00Z"},
        "label": {"urgency_score": urgency, "relevance_score": relevance,
                  "category": "긴급 업무", "ai_summary_reason": "즉시 대응 필요"}})


class ScoreSeparationTests(unittest.TestCase):
    def test_partial_preview_approval_does_not_approve_unshown_rows(self):
        from filtering_training.preparation.freeze_score_review import approve_validation_preview
        rows=[sample("shown"),sample("unshown")]
        rows[1].notification.body="different original"
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            write_lines(path/"candidates.jsonl",[r.model_dump(mode="json") for r in rows])
            (path/"review.md").write_text("only shown case",encoding="utf-8")
            write_json(path/"manifest.json",{"intended_split":"validation","originals":2,
                "displayed_review_ids":["shown"],"protected_sources":[],
                "candidate_sha256":digest(path/"candidates.jsonl"),"review_sha256":digest(path/"review.md")})
            before=digest(path/"candidates.jsonl")
            approve_validation_preview(path,"fixture approval")
            approval=json.loads((path/"approval.json").read_text(encoding="utf-8"))
            self.assertEqual(approval["approved_ids"],["shown"])
            self.assertEqual(approval["reviewed_context_rows"],1)
            self.assertEqual(digest(path/"candidates.jsonl"),before)
            self.assertFalse((path/"dataset.jsonl").exists())

    def test_isolation_audit_detects_recent_window_and_body_leakage(self):
        from filtering_training.quality.audit_digit_training import audit_pool_separation
        from filtering_training.common.score_tasks import HistorySample
        train=sample("train",window="different current window").model_dump(mode="json")
        train["notification"]["body"]="different training body"
        train["context"].update(recent_processes=["editor.exe"],
                                  recent_windows=[{"app_name":"editor.exe","window_title":"secret-context"}])
        heldout=sample("heldout")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,row in (("train",HistorySample.model_validate(train)),("val",heldout)):
                folder=root/name
                folder.mkdir()
                write_lines(folder/"candidates.jsonl",[row.model_dump(mode="json")])
                write_json(folder/"manifest.json",{"intended_split":"validation" if name=="val" else "train",
                    "candidate_sha256":digest(folder/"candidates.jsonl"),"reserved_families":{}})
            result=audit_pool_separation(root/"train",root/"val")
            self.assertFalse(result["passed"])
            self.assertEqual(result["shared_current_or_recent_windows"],[("editor.exe","secret-context")])
            self.assertEqual(result["shared_normalized_bodies"],[])
            train["notification"]["body"]="  immediate   action  "
            train["context"].update(recent_processes=[],recent_windows=[])
            write_lines(root/"train/candidates.jsonl",[HistorySample.model_validate(train).model_dump(mode="json")])
            write_json(root/"train/manifest.json",{"candidate_sha256":digest(root/"train/candidates.jsonl")})
            result=audit_pool_separation(root/"train",root/"val")
            self.assertFalse(result["passed"])
            self.assertEqual(result["shared_normalized_bodies"],["immediateaction"])

    def test_recent_titles_survive_loading_and_never_reach_urgency(self):
        from filtering_training.common.score_tasks import HistorySample, load_score_samples
        from filtering_training.common.digit_scores import replace_prompt
        value = sample().model_dump(mode="json")
        value["context"].update(recent_processes=["chrome.exe"],
                                recent_windows=[{"app_name":"chrome.exe","window_title":"private task title"}])
        history = HistorySample.model_validate(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"rows.jsonl"
            write_lines(path,[history.model_dump(mode="json")])
            loaded = load_score_samples(path)[0]
            self.assertEqual(loaded.context.recent_windows[0].window_title,"private task title")
        for mode in ("none","apps","titles"):
            actual = replace_prompt(messages(history,"urgency",mode),"urgency")
            self.assertEqual(actual,replace_prompt(messages(sample(),"urgency"),"urgency"))
        full = json.loads(messages(history,"relevance","titles")[1]["content"])["context"]
        apps = json.loads(messages(history,"relevance","apps")[1]["content"])["context"]
        none = json.loads(messages(history,"relevance","none")[1]["content"])["context"]
        self.assertIn("recent_windows",full)
        self.assertNotIn("recent_windows",apps)
        self.assertEqual(apps["recent_processes"],["chrome.exe"])
        self.assertEqual(none["recent_processes"],[])

    def test_history_rejects_mismatched_names_and_more_than_three_windows(self):
        from filtering_training.common.score_tasks import HistorySample
        value = sample().model_dump(mode="json")
        value["context"].update(recent_processes=["wrong.exe"],
                                recent_windows=[{"app_name":"chrome.exe","window_title":"task"}])
        with self.assertRaises(ValueError):
            HistorySample.model_validate(value)
        value["context"].update(recent_processes=[f"app{i}.exe" for i in range(4)],
                                recent_windows=[{"app_name":f"app{i}.exe","window_title":"task"} for i in range(4)])
        with self.assertRaises(ValueError):
            HistorySample.model_validate(value)

    def test_history_rule_survives_digit_prompt_replacement_and_ablation(self):
        from filtering_training.common.score_tasks import HistorySample, HISTORY_RELEVANCE_RULE
        from filtering_training.common.digit_scores import replace_prompt, system_prompt, prompt_digest
        value = sample().model_dump(mode="json")
        value["context"].update(recent_processes=["chrome.exe"],
                                recent_windows=[{"app_name":"chrome.exe","window_title":"task target"}])
        history = HistorySample.model_validate(value)
        for mode in ("none","apps","titles"):
            actual = replace_prompt(messages(history,"relevance",mode),"relevance")
            self.assertIn(HISTORY_RELEVANCE_RULE,actual[0]["content"])
            self.assertEqual(actual[0]["content"],system_prompt("relevance",history=True))
        old = replace_prompt(messages(sample(),"relevance"),"relevance")
        self.assertEqual(old[0]["content"],system_prompt("relevance"))
        self.assertNotEqual(prompt_digest("relevance"),prompt_digest("relevance",history=True))
        self.assertEqual(prompt_digest("urgency"),prompt_digest("urgency",history=True))

    def test_validation_approval_preserves_history_and_rejects_changed_review(self):
        from filtering_training.common.score_tasks import HistorySample, load_score_samples
        from filtering_training.preparation.freeze_score_review import freeze_validation
        value = sample().model_dump(mode="json")
        value["context"].update(recent_processes=["chrome.exe"],
                                recent_windows=[{"app_name":"chrome.exe","window_title":"private task title"}])
        row = HistorySample.model_validate(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_lines(path/"candidates.jsonl",[row.model_dump(mode="json")])
            (path/"review.md").write_text("reviewed labels",encoding="utf-8")
            manifest = {"intended_split":"validation","training_used":False,"originals":1,"rows":1,
                        "candidate_sha256":digest(path/"candidates.jsonl"),"review_sha256":digest(path/"review.md"),
                        "protected_sources":[],"reserved_families":{"fixture":["case"]}}
            write_json(path/"manifest.json",manifest)
            (path/"review.md").write_text("changed labels",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"review materials changed"):
                freeze_validation(path,path,"fixture approval")
            self.assertFalse((path/"dataset.jsonl").exists())
            (path/"review.md").write_text("reviewed labels",encoding="utf-8")
            freeze_validation(path,path,"fixture approval")
            self.assertEqual(load_score_samples(path/"dataset.jsonl")[0].model_dump(),row.model_dump())
            approval=json.loads((path/"approval.json").read_text(encoding="utf-8"))
            self.assertEqual(approval["intended_split"],"validation")
            self.assertFalse(approval["training_used"])
            with self.assertRaisesRegex(ValueError,"overwrite"):
                freeze_validation(path,path,"fixture approval")

    def test_context_and_variant_id_cannot_reach_urgency(self):
        related = sample()
        empty = sample("case_empty", window="")
        self.assertEqual(messages(related, "urgency"), messages(empty, "urgency"))
        payload = json.loads(messages(related, "urgency")[1]["content"])
        self.assertEqual(set(payload), {"notification"})
        self.assertNotIn("id", payload["notification"])
        self.assertNotIn("secret-context", json.dumps(messages(related, "urgency")))
        self.assertIn("context", json.loads(messages(related, "relevance")[1]["content"]))

    def test_malformed_predictions_are_not_silent_policy_decisions(self):
        for raw in ('{"urgency_score":true}', '{"urgency_score":6}',
                    '{"urgency_score":"4"}', '{"urgency_score":4,"urgency_score":1}',
                    '{"urgency_score":4,"context":{}}', '[4]', '{"urgency_score":4.0}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_scores(raw, "urgency")
        metrics = score_metrics([sample(), sample("two_empty"), sample("three_related", urgency=1)],
                                [None, {"urgency_score":1,"relevance_score":1},
                                 {"urgency_score":1,"relevance_score":5}])
        self.assertEqual(metrics["urgent_false_blocks"], 1)
        self.assertEqual(metrics["urgent_invalid"], 1)
        self.assertEqual(metrics["urgent_failures"], 2)
        self.assertEqual(metrics["false_passes"], 1)

    def test_deduplication_rejects_conflicting_context_labels(self):
        variants = [sample(), sample("case_empty", window="")]
        self.assertEqual(len(unique_urgency_samples(variants)), 1)
        variants[1].label.urgency_score = 1
        with self.assertRaises(ValueError):
            unique_urgency_samples(variants)

    def test_quarantine_propagates_across_shared_windows(self):
        items = [sample("a_related", window="one"), sample("b_related", window="one"),
                 sample("b_unrelated", window="two"), sample("c_related", window="two"),
                 sample("d_empty", window="")]
        self.assertEqual(quarantined_groups(items, {"a": ["heldout"]}), ["a", "b", "c"])

    def test_comparison_recomputes_scores_and_rejects_fabricated_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"prepared").mkdir()
            item = sample()
            write_lines(root/"dataset.jsonl", [item.model_dump(mode="json")])
            write_json(root/"prepared/manifest.json", {
                "source_sha256": digest(root/"dataset.jsonl"),
                "splits": {"validation": {"notification_ids": [item.notification.id]}}})
            prediction = {"urgency_score":5, "relevance_score":1}
            reports = []
            for variant, tasks in {"A":("full",),"B":("scores",),"C":("urgency","relevance")}.items():
                raw = {"full":item.label.model_dump(), "scores":prediction,
                       "urgency":{"urgency_score":5}, "relevance":{"relevance_score":1}}
                rows = [{"notification_id":item.notification.id, "notification":item.notification.model_dump(mode="json"),
                         "context":item.context.model_dump(mode="json"), "gold":item.label.model_dump(),
                         "prediction":prediction, "responses":{t:{"raw":json.dumps(raw[t]),"error":None} for t in tasks}}]
                path = root/f"{variant}.json"
                write_lines(path.with_suffix(".predictions.jsonl"), rows)
                write_json(path, {"variant":variant,"sampling":None,"split":"validation", "policy_version":POLICY_VERSION,
                                  "evaluated_ids":[item.notification.id],"dataset_sha256":digest(root/"dataset.jsonl"),
                                  "source_manifest_sha256":digest(root/"prepared/manifest.json"),
                                  "prompt_hashes":{t:prompt_digest(t) for t in tasks},
                                  "predictions_sha256":digest(path.with_suffix(".predictions.jsonl")),
                                  "metrics":score_metrics([item],[prediction]),"performance":{}})
                reports.append(path)
            self.assertEqual(set(compare(root,reports,root/"summary.json")["variants"]), {"A","B","C"})
            bad_rows = [json.loads(line) for line in reports[1].with_suffix(".predictions.jsonl").read_text().splitlines()]
            bad_rows[0]["prediction"] = {"urgency_score":1,"relevance_score":1}
            write_lines(reports[1].with_suffix(".predictions.jsonl"),bad_rows)
            report=json.loads(reports[1].read_text())
            report["predictions_sha256"]=digest(reports[1].with_suffix(".predictions.jsonl"))
            write_json(reports[1],report)
            with self.assertRaisesRegex(ValueError,"raw task outputs"):
                compare(root,reports,root/"rejected.json")


if __name__ == "__main__":
    unittest.main()
