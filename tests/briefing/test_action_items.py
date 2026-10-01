from __future__ import annotations

from datetime import datetime, timezone
import unittest

from briefing_training.evaluate import load_cases
from src.briefing.pipeline import SessionBriefingService


def notification(
    notification_id: str,
    *,
    title: str,
    body: str,
    timestamp: str = "2026-09-13T18:05:00Z",
    app_name: str = "Slack",
    sender: str = "가상 팀장",
) -> dict[str, object]:
    return {
        "id": notification_id,
        "app_name": app_name,
        "sender": sender,
        "title": title,
        "body": body,
        "timestamp": timestamp,
    }


def filter_result(
    notification_id: str,
    *,
    is_passed: bool = False,
    category: str = "일반 업무",
) -> dict[str, object]:
    return {
        "notification_id": notification_id,
        "is_passed": is_passed,
        "urgency_score": 3,
        "relevance_score": 4,
        "category": category,
        "ai_summary_reason": "가상 테스트 판단",
    }


class ActionItemExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = SessionBriefingService()
        self.generated_at = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)

    def build(
        self,
        notifications: list[dict[str, object]],
        results: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self.service.build(
            session_id="session_actions",
            notifications=notifications,
            filter_results=(
                results
                if results is not None
                else [filter_result(item["id"]) for item in notifications]
            ),
            generated_at=self.generated_at,
        ).to_dict()

    def test_empty_input_returns_empty_candidate_lists(self) -> None:
        briefing = self.build([])

        self.assertEqual(briefing["todo_candidate_count"], 0)
        self.assertEqual(briefing["todo_candidates"], [])
        self.assertEqual(briefing["calendar_candidate_count"], 0)
        self.assertEqual(briefing["calendar_candidates"], [])
        self.assertEqual(briefing["schedule_summary_count"], 0)
        self.assertEqual(briefing["schedule_summaries"], [])

    def test_passed_notification_is_not_extracted(self) -> None:
        item = notification(
            "n1",
            title="보고서 제출 요청",
            body="보고서를 내일까지 제출해 주세요.",
        )
        briefing = self.build([item], [filter_result("n1", is_passed=True)])

        self.assertEqual(briefing["todo_candidates"], [])
        self.assertEqual(briefing["calendar_candidates"], [])
        self.assertEqual(briefing["schedule_summaries"], [])

    def test_todo_with_korean_due_date_is_structured(self) -> None:
        briefing = self.build(
            [
                notification(
                    "n1",
                    title="보고서 제출 요청",
                    body="보고서를 9월 15일까지 제출해 주세요.",
                )
            ]
        )

        candidate = briefing["todo_candidates"][0]
        self.assertEqual(candidate["due_at"], "2026-09-15T00:00:00Z")
        self.assertTrue(candidate["is_all_day"])
        self.assertEqual(candidate["source_notification_ids"], ["n1"])
        self.assertIn("제출", candidate["matched_cues"])

    def test_calendar_candidate_parses_explicit_date_and_time(self) -> None:
        briefing = self.build(
            [
                notification(
                    "n1",
                    title="프로젝트 회의 일정",
                    body="프로젝트 회의는 2026년 9월 20일 오후 3시에 진행됩니다.",
                )
            ]
        )

        candidate = briefing["calendar_candidates"][0]
        self.assertEqual(candidate["scheduled_at"], "2026-09-20T15:00:00Z")
        self.assertFalse(candidate["is_all_day"])
        self.assertEqual(candidate["title"], "프로젝트 회의 일정")
        self.assertEqual(candidate["status"], "scheduled")
        self.assertEqual(
            candidate["schedule_details"],
            {
                "who": None,
                "when": "2026-09-20T15:00:00Z",
                "where": None,
                "what": "프로젝트 회의 일정",
                "why": None,
                "how": None,
            },
        )

    def test_relative_date_is_based_on_notification_timestamp(self) -> None:
        briefing = self.build(
            [
                notification(
                    "n1",
                    title="프로젝트 회의",
                    body="프로젝트 회의는 내일 오전 10시 30분에 진행됩니다.",
                )
            ]
        )

        self.assertEqual(
            briefing["calendar_candidates"][0]["scheduled_at"],
            "2026-09-14T10:30:00Z",
        )

    def test_repeated_candidates_are_merged_with_all_sources(self) -> None:
        items = [
            notification(
                "n1",
                title="과제 제출",
                body="과제를 9월 15일까지 제출해 주세요.",
                timestamp="2026-09-13T18:05:00Z",
            ),
            notification(
                "n2",
                title="과제 제출",
                body="과제를 9월 15일까지 제출해 주세요.",
                timestamp="2026-09-13T19:05:00Z",
            ),
        ]
        briefing = self.build(items)

        self.assertEqual(briefing["todo_candidate_count"], 1)
        self.assertEqual(briefing["calendar_candidate_count"], 1)
        self.assertEqual(
            briefing["todo_candidates"][0]["source_notification_ids"],
            ["n1", "n2"],
        )
        self.assertEqual(
            briefing["calendar_candidates"][0]["source_notification_ids"],
            ["n1", "n2"],
        )

    def test_invalid_calendar_date_is_not_guessed(self) -> None:
        briefing = self.build(
            [
                notification(
                    "n1",
                    title="팀 회의 일정",
                    body="팀 회의 일정은 2월 30일입니다.",
                )
            ]
        )

        self.assertEqual(briefing["calendar_candidates"], [])
        summary = briefing["schedule_summaries"][0]
        self.assertIsNone(summary["schedule_details"]["when"])

    def test_fragmented_schedule_messages_form_one_six_question_candidate(self) -> None:
        items = [
            notification(
                "chat_20260913_001",
                title="회의 공지",
                body="참석자는 백엔드 팀이고 무엇: 주간 회의입니다.",
                timestamp="2026-09-13T18:59:00Z",
            ),
            notification(
                "chat_20260913_002",
                title="시간 안내",
                body="내일 오후 3시에 진행합니다.",
                timestamp="2026-09-13T19:00:00Z",
            ),
            notification(
                "chat_20260913_003",
                title="장소 안내",
                body="장소는 B강의실입니다.",
                timestamp="2026-09-13T19:01:00Z",
            ),
            notification(
                "chat_20260913_004",
                title="진행 안내",
                body="이유는 배포 일정 점검이고 방식은 대면입니다.",
                timestamp="2026-09-13T19:02:00Z",
            ),
        ]

        briefing = self.build(items)

        self.assertEqual(briefing["calendar_candidate_count"], 1)
        candidate = briefing["calendar_candidates"][0]
        self.assertEqual(candidate["scheduled_at"], "2026-09-14T15:00:00Z")
        self.assertEqual(candidate["status"], "scheduled")
        self.assertEqual(
            candidate["source_notification_ids"],
            [item["id"] for item in items],
        )
        self.assertEqual(
            candidate["schedule_details"],
            {
                "who": "백엔드 팀",
                "when": "2026-09-14T15:00:00Z",
                "where": "B강의실",
                "what": "주간 회의",
                "why": "배포 일정 점검",
                "how": "대면",
            },
        )

    def test_latest_schedule_change_sets_changed_status_and_time(self) -> None:
        briefing = self.build(
            [
                notification(
                    "schedule_001",
                    title="팀 회의 일정",
                    body="팀 회의는 9월 20일 오후 2시입니다.",
                    timestamp="2026-09-13T18:05:00Z",
                ),
                notification(
                    "schedule_002",
                    title="팀 회의 일정 변경",
                    body="9월 21일 오후 4시로 변경되었습니다.",
                    timestamp="2026-09-13T18:06:00Z",
                ),
            ]
        )

        self.assertEqual(briefing["calendar_candidate_count"], 1)
        candidate = briefing["calendar_candidates"][0]
        self.assertEqual(candidate["scheduled_at"], "2026-09-21T16:00:00Z")
        self.assertEqual(candidate["status"], "changed")
        self.assertEqual(
            candidate["source_notification_ids"],
            ["schedule_001", "schedule_002"],
        )

    def test_schedule_category_emits_candidate_without_calendar_keyword(self) -> None:
        item = notification(
            "study_001",
            title="AI 스터디 안내",
            body="내일 오후 3시에 만나요.",
        )

        briefing = self.build(
            [item],
            [filter_result("study_001", category="일정/회의")],
        )

        self.assertEqual(briefing["calendar_candidate_count"], 1)
        self.assertEqual(
            briefing["calendar_candidates"][0]["matched_cues"],
            ["일정/회의"],
        )

    def test_same_message_schedule_correction_uses_last_date_and_time(self) -> None:
        briefing = self.build(
            [
                notification(
                    "correction_001",
                    title="팀 회의 일정 변경",
                    body=(
                        "기존 9월 20일 오후 2시에서 "
                        "9월 21일 오후 4시로 변경되었습니다."
                    ),
                )
            ]
        )

        candidate = briefing["calendar_candidates"][0]
        self.assertEqual(candidate["scheduled_at"], "2026-09-21T16:00:00Z")
        self.assertEqual(candidate["status"], "changed")

    def test_resumed_schedule_after_cancellation_is_scheduled(self) -> None:
        briefing = self.build(
            [
                notification(
                    "resume_001",
                    title="팀 회의 재개",
                    body=(
                        "취소했던 팀 회의를 9월 21일 오후 4시에 "
                        "다시 진행합니다."
                    ),
                )
            ]
        )

        candidate = briefing["calendar_candidates"][0]
        self.assertEqual(candidate["status"], "scheduled")

    def test_schedule_without_time_keeps_fixed_fields_and_null_when(self) -> None:
        item = notification(
            "study_002",
            title="AI 스터디 공지",
            body="장소는 B강의실입니다.",
        )

        briefing = self.build(
            [item],
            [filter_result("study_002", category="일정/회의")],
        )

        self.assertEqual(briefing["calendar_candidates"], [])
        candidate = briefing["schedule_summaries"][0]
        self.assertEqual(
            set(candidate["schedule_details"]),
            {"who", "when", "where", "what", "why", "how"},
        )
        self.assertIsNone(candidate["schedule_details"]["when"])
        self.assertEqual(candidate["schedule_details"]["where"], "B강의실")

    def test_natural_fragmented_evaluation_case_populates_six_questions(self) -> None:
        case = next(
            case
            for case in load_cases()
            if case["case_id"] == "fragmented_schedule_chat"
        )
        group = case["input"]
        notifications = [
            {
                **item,
                "app_name": group["app_name"],
                "sender": group["sender"],
            }
            for item in group["notifications"]
        ]
        results = [
            filter_result(item["id"], category="일정/회의")
            for item in notifications
        ]

        briefing = self.build(notifications, results)

        self.assertEqual(briefing["group_count"], 1)
        self.assertEqual(briefing["schedule_summary_count"], 1)
        summary = briefing["schedule_summaries"][0]
        self.assertEqual(summary["title"], "졸업 작품 중간 점검")
        self.assertEqual(
            summary["schedule_details"],
            {
                "who": "캡스톤 참가자들",
                "when": "2026-10-12T19:00:00Z",
                "where": "창의관 402호",
                "what": "졸업 작품 중간 점검",
                "why": "시연 동선 확인",
                "how": "오프라인",
            },
        )

    def test_undated_same_title_schedules_from_different_senders_do_not_merge(self) -> None:
        items = [
            notification(
                "alice_001",
                title="일정 안내",
                body="장소는 A회의실입니다.",
                sender="Alice",
            ),
            notification(
                "bob_001",
                title="일정 안내",
                body="장소는 B회의실입니다.",
                sender="Bob",
                timestamp="2026-09-13T18:06:00Z",
            ),
        ]
        results = [
            filter_result(item["id"], category="일정/회의") for item in items
        ]

        briefing = self.build(items, results)

        self.assertEqual(briefing["group_count"], 2)
        self.assertEqual(briefing["calendar_candidate_count"], 0)
        self.assertEqual(briefing["schedule_summary_count"], 2)
        self.assertEqual(
            {
                summary["schedule_details"]["where"]
                for summary in briefing["schedule_summaries"]
            },
            {"A회의실", "B회의실"},
        )


if __name__ == "__main__":
    unittest.main()
