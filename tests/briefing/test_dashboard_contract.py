from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from briefing_training.smoke_test_runtime_provider import DEFAULT_SAMPLE_DIR, load_json_array
from src.briefing import QwenBriefingProvider, SessionBriefingService


ROOT = Path(__file__).resolve().parents[2]
CARD_FIELDS = {
    "session_id", "group_id", "app_name", "sender", "primary_category",
    "summary_lines", "schedule_summaries",
}
SCHEDULE_FIELDS = {
    "summary_id", "schedule_status", "is_all_day", "end_at",
    "who", "when", "where", "what", "why", "how",
}


class DashboardContractTests(unittest.TestCase):
    def check_contract(self, report) -> None:
        self.assertEqual(set(report), {"session_id", "groups"})
        for card in report["groups"]:
            self.assertEqual(set(card), CARD_FIELDS)
            self.assertEqual(card["session_id"], report["session_id"])
            for schedule in card["schedule_summaries"]:
                self.assertEqual(set(schedule), SCHEDULE_FIELDS)
                self.assertNotIn("schedule_details", schedule)
                self.assertNotIn("status", schedule)

    def test_readme_example_matches_real_pipeline_serialization(self) -> None:
        markdown = (ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(r"```json\s*\n(.*?)\n```", markdown, re.DOTALL)
        self.assertIsNotNone(match)
        example = json.loads(match.group(1))
        self.check_contract(example)
        expected_card = example["groups"][0]

        class FakeBackend:
            def generate(self, messages):
                return json.dumps({"primary_category": expected_card["primary_category"], "summary_lines": expected_card["summary_lines"]}, ensure_ascii=False)

        service = SessionBriefingService(
            provider=QwenBriefingProvider(backend=FakeBackend(), allow_fallback=False)
        )
        actual = json.loads(service.build_json(
            session_id="session_001",
            notifications=[{
                "id": "chat_001", "app_name": "KakaoTalk", "sender": "팀장",
                "title": "중간 점검 회의",
                "body": (
                    "참석자는 개발팀이고 무엇: 중간 점검 회의입니다. "
                    "10월 12일 오후 7시에 진행됩니다. 장소는 창의관 402호입니다."
                ),
                "timestamp": "2026-10-07T09:00:00Z",
            }],
            filter_results=[{
                "notification_id": "chat_001", "is_passed": False,
                "category": "일정/회의", "ai_summary_reason": "합성 테스트",
            }],
        ))
        self.check_contract(actual)
        # Documentation uses readable placeholder IDs; the pipeline hashes IDs.
        expected_card["group_id"] = actual["groups"][0]["group_id"]
        expected_card["schedule_summaries"][0]["summary_id"] = (
            actual["groups"][0]["schedule_summaries"][0]["summary_id"]
        )
        self.assertEqual(actual, example)

    def test_runtime_fixture_uses_the_same_dashboard_contract(self) -> None:
        report = json.loads(SessionBriefingService().build_json(
            session_id="runtime_fixture",
            notifications=load_json_array(DEFAULT_SAMPLE_DIR / "raw_notifications.json"),
            filter_results=load_json_array(DEFAULT_SAMPLE_DIR / "filter_results.json"),
        ))
        self.check_contract(report)
        self.assertEqual(len(report["groups"]), 3)
        self.assertEqual(sum(len(card["schedule_summaries"]) for card in report["groups"]), 2)


if __name__ == "__main__":
    unittest.main()
