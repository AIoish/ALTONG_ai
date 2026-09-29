import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from briefing_training.evaluate import (
    _expected_facts,
    _forbidden_phrases,
    _max_summary_lines,
)
from briefing_training.prompts import (
    SummaryResponseError,
    build_messages,
    parse_summary_response,
)
from briefing_training.smoke_test_model import attach_adapter


SYNTHETIC_GROUP = {
    "app_name": "Slack",
    "sender": "가상 팀장",
    "category": "긴급 업무",
    "urgency_score": {"min": 4, "max": 5, "average": 4.5},
    "relevance_score": {"min": 4, "max": 5, "average": 4.5},
    "notifications": [
        {
            "id": "noti_test_001",
            "timestamp": "2026-09-25T09:00:00Z",
            "title": "가상 서버 오류",
            "body": "로그인 API 오류를 확인해 주세요.",
        }
    ],
}


class QwenSummaryBaselineTests(unittest.TestCase):
    def test_adapter_is_loaded_from_the_requested_path(self) -> None:
        base_model = object()
        adapted_model = object()

        class FakePeftModel:
            @staticmethod
            def from_pretrained(model, adapter_path):
                self.assertIs(model, base_model)
                self.assertEqual(adapter_path, "saved-adapter")
                return adapted_model

        with patch.dict(
            sys.modules,
            {"peft": SimpleNamespace(PeftModel=FakePeftModel)},
        ):
            actual = attach_adapter(base_model, "saved-adapter")

        self.assertIs(actual, adapted_model)

    def test_prompt_contains_group_context_and_output_contract(self) -> None:
        messages = build_messages(SYNTHETIC_GROUP, max_summary_lines=1)
        actual_prompt = messages[-1]["content"]

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user"],
        )
        self.assertIn('"summary_lines"', messages[0]["content"])
        self.assertIn("최상위 값은 배열이 아니라", messages[0]["content"])
        self.assertIn("가상 서버 오류", actual_prompt)
        self.assertIn("긴급 업무", actual_prompt)
        self.assertIn("notifications_oldest_to_newest", actual_prompt)
        self.assertNotIn("noti_test_001", actual_prompt)
        self.assertNotIn("[알림 1]", actual_prompt)
        self.assertIn("최대 요약 줄 수: 1", actual_prompt)
        self.assertIn("최대 1개의 문장", actual_prompt)
        self.assertIn("제목만 복사하지 말고", actual_prompt)
        self.assertIn("반드시 { 문자로 시작", actual_prompt)
        self.assertIn("마지막 항목이 가장 최신", actual_prompt)
        self.assertIn("서로 다른 정보를 보태면", actual_prompt)

    def test_parser_accepts_one_to_three_summary_lines(self) -> None:
        parsed = parse_summary_response(
            '{"summary_lines":["로그인 API 오류를 확인해야 합니다.","현재 조치 중입니다."]}'
        )

        self.assertEqual(
            parsed,
            ("로그인 API 오류를 확인해야 합니다.", "현재 조치 중입니다."),
        )

    def test_parser_accepts_json_code_fence(self) -> None:
        parsed = parse_summary_response(
            '```json\n{"summary_lines":["가상 알림 요약"]}\n```'
        )

        self.assertEqual(parsed, ("가상 알림 요약",))

    def test_parser_rejects_thinking_or_explanatory_text(self) -> None:
        with self.assertRaises(SummaryResponseError):
            parse_summary_response(
                '<think>reasoning</think>\n{"summary_lines":["요약"]}'
            )

    def test_parser_rejects_more_than_three_lines(self) -> None:
        with self.assertRaisesRegex(SummaryResponseError, "one to three"):
            parse_summary_response(
                '{"summary_lines":["첫째","둘째","셋째","넷째"]}'
            )

    def test_parser_rejects_unexpected_fields(self) -> None:
        with self.assertRaisesRegex(SummaryResponseError, "only"):
            parse_summary_response(
                '{"summary_lines":["요약"],"reason":"추가 필드"}'
            )

    def test_empty_notification_group_is_rejected(self) -> None:
        group = dict(SYNTHETIC_GROUP)
        group["notifications"] = []

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_messages(group)

    def test_invalid_max_summary_lines_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer from 1 to 3"):
            build_messages(SYNTHETIC_GROUP, max_summary_lines=4)

    def test_evaluation_case_supports_fact_alternatives_and_latest_state(self) -> None:
        case = {
            "expected_facts": [["복구", "정상화"], ["로그인"]],
            "forbidden_phrases": ["장애 진행 중"],
            "max_summary_lines": 1,
        }

        self.assertEqual(
            _expected_facts(case),
            (("복구", "정상화"), ("로그인",)),
        )
        self.assertEqual(_forbidden_phrases(case), ("장애 진행 중",))
        self.assertEqual(_max_summary_lines(case), 1)


if __name__ == "__main__":
    unittest.main()
