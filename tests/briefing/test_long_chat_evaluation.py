"""The challenging long-chat cases must reject misleading example outputs."""
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from briefing_training import evaluate
from briefing_training.smoke_test_model import load_cases


class LongChatEvaluationTests(unittest.TestCase):
    def test_rejects_missing_file_naming_and_outdated_cancellation(self):
        path = Path("briefing_training/data/category_briefing_v9/long_chat_evaluation_v2.jsonl")
        cases = {case["case_id"]: case for case in load_cases(path)}
        examples = [
            ("long_chat_v2_video_upload", "일반 업무", [
                "영상과 자막을 11월 14일 오후 5시까지 공유 폴더에 올려야 합니다.",
                "썸네일을 포함해야 합니다.",
            ]),
            ("long_chat_v2_meeting_reinstated", "일정/회의", [
                "11월 20일 오후 2시 회의실 C에서 진행합니다.",
                "팀장님이 어렵다고 했으므로 일단 취소합니다.",
            ]),
            ("long_chat_v2_delivery_changed", "개인 중요", [
                "무인 택배함에 배송이 도착했습니다.",
                "택배는 경비실에 보관합니다.",
            ]),
        ]
        for case_id, category, lines in examples:
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                raw = json.dumps({"primary_category": category, "summary_lines": lines}, ensure_ascii=False)
                output = io.StringIO()
                with patch("sys.argv", ["evaluate", "--task", "briefing", "--greedy"]), \
                     patch.object(evaluate, "load_cases", return_value=[case]), \
                     patch.object(evaluate, "load_model", return_value=(None, None)), \
                     patch.object(evaluate, "generate_summary", return_value=(raw, 0.1)), \
                     redirect_stdout(output):
                    evaluate.main()
                result = json.loads(output.getvalue())["results"][0]
                self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
