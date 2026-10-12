from __future__ import annotations

import json
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

from briefing_training import evaluate
from briefing_training.prepare_conversation_repairs import repair_cases
from briefing_training.prepare_category_dataset import validate_records
from briefing_training.train_lora import validate_split_separation
from briefing_training.evaluate import validate_evaluation_cases
from briefing_training.export_merged_model import validate_export


class ConversationRepairTests(unittest.TestCase):
    def test_training_and_validation_are_held_out(self):
        train, validation = repair_cases("train"), repair_cases("validation")
        self.assertEqual((len(train), len(validation)), (96, 24))
        validate_records(train, expected_split="train")
        validate_records(validation, expected_split="validation")
        validate_split_separation(train, validation)

    def test_evaluation_keeps_negation_and_specific_details(self):
        cases = repair_cases("evaluation")
        self.assertEqual(len(cases), 8)
        validate_evaluation_cases(cases)
        case = cases[0]
        self.assertIn(["본인 활동이 아닌"], case["expected_facts"])
        self.assertIn("로그인을 수행", case["forbidden_phrases"])
        self.assertTrue(all(case["require_korean"] for case in cases))
        as_train = [{**r, "target": {"primary_category": r["expected_category"], "summary_lines": r["reference_summary_lines"]}} for r in cases]
        validate_split_separation(repair_cases("train"), as_train)
        validate_split_separation(repair_cases("validation"), as_train)

    def test_export_checks_actual_base_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = root / "adapter"
            adapter.mkdir()
            config = adapter / "adapter_config.json"
            config.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-0.6B"}), encoding="utf-8")
            (adapter / "adapter_model.safetensors").touch()
            self.assertEqual(validate_export(adapter, root / "merged"), "Qwen/Qwen3-0.6B")
            with self.assertRaisesRegex(ValueError, "already exists"):
                validate_export(adapter, adapter)
            config.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                validate_export(adapter, root / "merged")

    def test_english_only_response_fails_required_korean_check(self):
        case = repair_cases("evaluation")[0]
        case["expected_facts"] = [["PDF"]]
        case["reference_summary_lines"] = ["PDF를 업로드했습니다."]
        case["forbidden_phrases"] = []
        for line, expected in (("Uploaded PDF.", False), ("PDF를 업로드했습니다.", True)):
            raw = json.dumps({"primary_category": case["expected_category"], "summary_lines": [line]})
            output = io.StringIO()
            with patch("sys.argv", ["evaluate", "--task", "briefing", "--greedy"]), \
                 patch.object(evaluate, "load_cases", return_value=[case]), \
                 patch.object(evaluate, "load_model", return_value=(None, None)), \
                 patch.object(evaluate, "generate_summary", return_value=(raw, 0.1)), \
                 redirect_stdout(output):
                evaluate.main()
            result = json.loads(output.getvalue())["results"][0]
            self.assertEqual(result["language_passed"], expected)
            self.assertEqual(result["passed"], expected)
