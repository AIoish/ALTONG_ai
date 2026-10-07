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
    timestamp: str = "2026-09-13T09:05:00Z",
    app_name: str = "Slack",
    sender: str = "가상 팀장",
) -> dict[str, object]:
    return {
        "id": notification_id, "app_name": app_name, "sender": sender,
        "title": title, "body": body, "timestamp": timestamp,
    }


def filter_result(
    notification_id: str, *, is_passed: bool = False, category: str = "일반 업무",
) -> dict[str, object]:
    return {
        "notification_id": notification_id, "is_passed": is_passed,
        "category": category, "ai_summary_reason": "가상 테스트 판단",
    }


class ActionItemExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = SessionBriefingService()

    def build(self, items, results=None):
        return self.service.build(
            session_id="session_schedules", notifications=items,
            filter_results=results if results is not None else [
                filter_result(item["id"]) for item in items
            ],
            generated_at=datetime(2026, 9, 13, 20, tzinfo=timezone.utc),
        )

    def schedules(self, items, results=None):
        return [
            summary for group in self.build(items, results).to_dict()["groups"]
            for summary in group["schedule_summaries"]
        ]

    def test_empty_input_returns_minimal_empty_report(self) -> None:
        self.assertEqual(
            self.build([]).to_dict(), {"session_id": "session_schedules", "groups": []}
        )

    def test_passed_notification_is_not_extracted(self) -> None:
        item = notification("n1", title="팀 회의", body="내일 오후 3시 회의입니다.")
        self.assertEqual(self.schedules([item], [filter_result("n1", is_passed=True)]), [])

    def test_non_schedule_work_does_not_create_tasks_or_schedules(self) -> None:
        result = self.build([
            notification("n1", title="보고서", body="보고서 작성 완료했습니다.")
        ])
        self.assertEqual(result.to_dict()["groups"][0]["schedule_summaries"], [])
        self.assertFalse(hasattr(result, "todo_candidates"))
        self.assertFalse(hasattr(result, "calendar_candidates"))

    def test_calendar_date_and_time_are_flattened_and_converted_from_korea(self) -> None:
        summary = self.schedules([
            notification("n1", title="프로젝트 회의 일정",
                         body="프로젝트 회의는 2026년 9월 20일 오후 3시에 진행됩니다.")
        ])[0]
        self.assertEqual(summary["when"], "2026-09-20T06:00:00Z")
        self.assertFalse(summary["is_all_day"])
        self.assertEqual(summary["schedule_status"], "scheduled")
        self.assertNotIn("status", summary)
        self.assertEqual(summary["what"], "프로젝트 회의 일정")
        self.assertEqual(set(summary), {
            "summary_id", "schedule_status", "is_all_day",
            "who", "when", "where", "what", "why", "how",
        })

    def test_date_only_uses_date_string_instead_of_fake_midnight_event(self) -> None:
        summary = self.schedules([
            notification("n1", title="과제 제출", body="과제를 9월 15일까지 제출해 주세요.")
        ])[0]
        self.assertEqual(summary["when"], "2026-09-15")
        self.assertTrue(summary["is_all_day"])

    def test_relative_date_is_based_on_korean_notification_date(self) -> None:
        summary = self.schedules([
            notification("n1", title="프로젝트 회의",
                         body="프로젝트 회의는 내일 오전 10시 30분에 진행됩니다.",
                         timestamp="2026-09-13T18:05:00Z")
        ])[0]
        # The source was received on Sep 14 in Korea, so tomorrow is Sep 15.
        self.assertEqual(summary["when"], "2026-09-15T01:30:00Z")

    def test_timestamp_offsets_produce_the_same_schedule(self) -> None:
        first = self.schedules([
            notification("n1", title="팀 회의", body="내일 오전 10시 회의입니다.",
                         timestamp="2026-09-13T18:05:00Z")
        ])[0]
        second = self.schedules([
            notification("n1", title="팀 회의", body="내일 오전 10시 회의입니다.",
                         timestamp="2026-09-14T03:05:00+09:00")
        ])[0]
        self.assertEqual(first, second)

    def test_invalid_date_is_not_guessed(self) -> None:
        summary = self.schedules([
            notification("n1", title="팀 회의", body="팀 회의 일정은 2월 30일입니다.")
        ])[0]
        self.assertIsNone(summary["when"])
        self.assertIsNone(summary["is_all_day"])

    def test_fragmented_schedule_forms_one_flat_six_question_report(self) -> None:
        items = [
            notification("chat_001", title="회의 공지",
                         body="참석자는 백엔드 팀이고 무엇: 주간 회의입니다.",
                         timestamp="2026-09-13T09:59:00Z"),
            notification("chat_002", title="시간 안내", body="내일 오후 3시에 진행합니다.",
                         timestamp="2026-09-13T10:00:00Z"),
            notification("chat_003", title="장소 안내", body="장소는 B강의실입니다.",
                         timestamp="2026-09-13T10:01:00Z"),
            notification("chat_004", title="진행 안내",
                         body="이유는 배포 일정 점검이고 방식은 대면입니다.",
                         timestamp="2026-09-13T10:02:00Z"),
        ]
        result = self.build(items)
        self.assertEqual(len(result.groups), 1)
        summary = result.groups[0].schedule_summaries[0]
        self.assertEqual(summary.source_notification_ids, tuple(item["id"] for item in items))
        output = summary.to_dict()
        self.assertEqual(
            {key: output[key] for key in ("who", "when", "where", "what", "why", "how")},
            {"who": "백엔드 팀", "when": "2026-09-14T06:00:00Z",
             "where": "B강의실", "what": "주간 회의",
             "why": "배포 일정 점검", "how": "대면"},
        )

    def test_latest_schedule_change_preserves_state_and_latest_time(self) -> None:
        summary = self.schedules([
            notification("schedule_001", title="팀 회의 일정",
                         body="팀 회의는 9월 20일 오후 2시입니다."),
            notification("schedule_002", title="팀 회의 일정 변경",
                         body="9월 21일 오후 4시로 변경되었습니다.",
                         timestamp="2026-09-13T09:06:00Z"),
        ])[0]
        self.assertEqual(summary["when"], "2026-09-21T07:00:00Z")
        self.assertEqual(summary["schedule_status"], "changed")
        self.assertNotIn("status", summary)

    def test_category_emits_schedule_without_calendar_keyword(self) -> None:
        item = notification("n1", title="AI 스터디 안내", body="내일 오후 3시에 만나요.")
        summary = self.schedules([item], [filter_result("n1", category="일정/회의")])[0]
        self.assertEqual(summary["when"], "2026-09-14T06:00:00Z")

    def test_same_message_correction_uses_last_date_and_time(self) -> None:
        summary = self.schedules([
            notification("n1", title="팀 회의 일정 변경",
                         body="기존 9월 20일 오후 2시에서 9월 21일 오후 4시로 변경되었습니다.")
        ])[0]
        self.assertEqual(summary["when"], "2026-09-21T07:00:00Z")
        self.assertEqual(summary["schedule_status"], "changed")

    def test_cancellation_returns_only_schedule_status(self) -> None:
        summary = self.schedules([
            notification("n1", title="팀 회의 취소", body="9월 21일 오후 4시 회의가 취소되었습니다.")
        ])[0]
        self.assertEqual(summary["schedule_status"], "cancelled")
        self.assertNotIn("status", summary)

    def test_resumed_schedule_after_cancellation_is_scheduled(self) -> None:
        summary = self.schedules([
            notification("n1", title="팀 회의 재개",
                         body="취소했던 팀 회의를 9월 21일 오후 4시에 다시 진행합니다.")
        ])[0]
        self.assertEqual(summary["schedule_status"], "scheduled")

    def test_missing_time_keeps_six_questions_and_unknown_all_day(self) -> None:
        item = notification("n1", title="AI 스터디 공지", body="장소는 B강의실입니다.")
        summary = self.schedules([item], [filter_result("n1", category="일정/회의")])[0]
        self.assertIsNone(summary["when"])
        self.assertIsNone(summary["is_all_day"])
        self.assertIsNone(summary["who"])
        self.assertEqual(summary["where"], "B강의실")

    def test_natural_fragmented_evaluation_case_populates_six_questions(self) -> None:
        case = next(case for case in load_cases()
                    if case["case_id"] == "fragmented_schedule_chat")
        source = case["input"]
        items = [
            {**item, "app_name": source["app_name"], "sender": source["sender"]}
            for item in source["notifications"]
        ]
        summary = self.schedules(
            items, [filter_result(item["id"], category="일정/회의") for item in items]
        )[0]
        self.assertEqual(summary["who"], "캡스톤 참가자들")
        self.assertEqual(summary["when"], "2026-10-12T10:00:00Z")
        self.assertEqual(summary["where"], "창의관 402호")
        self.assertEqual(summary["what"], "졸업 작품 중간 점검")
        self.assertEqual(summary["why"], "시연 동선 확인")
        self.assertEqual(summary["how"], "오프라인")

    def test_schedules_from_different_senders_keep_separate_cards_and_ids(self) -> None:
        items = [
            notification("alice_001", title="일정 안내", body="장소는 A회의실입니다.", sender="Alice"),
            notification("bob_001", title="일정 안내", body="장소는 B회의실입니다.", sender="Bob"),
        ]
        summaries = self.schedules(
            items, [filter_result(item["id"], category="일정/회의") for item in items]
        )
        self.assertEqual(len(summaries), 2)
        self.assertNotEqual(summaries[0]["summary_id"], summaries[1]["summary_id"])
        self.assertEqual({summary["where"] for summary in summaries}, {"A회의실", "B회의실"})

    def test_schedule_has_no_calendar_registration_field(self) -> None:
        result = self.build([
            notification("n1", title="회의", body="내일 오후 3시 회의입니다.")
        ])
        summary = result.groups[0].schedule_summaries[0]
        self.assertFalse(hasattr(summary, "registration_status"))
        self.assertNotIn("status", summary.to_dict())
        self.assertEqual(summary.to_dict()["schedule_status"], "scheduled")


if __name__ == "__main__":
    unittest.main()
