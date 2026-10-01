from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from src.briefing.pipeline import SessionBriefingService
from src.briefing.schema import ContractValidationError, FILTER_CATEGORIES


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "sample" / "briefing"
)


def load_fixture(name: str) -> list[dict[str, object]]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def notification(
    notification_id: str,
    *,
    app: str = "Slack",
    sender: str = "가상 사용자",
    title: str = "테스트 서버 오류",
    body: str = "테스트 서버 오류를 확인해 주세요.",
    timestamp: str = "2026-09-13T18:05:00Z",
) -> dict[str, object]:
    return {
        "id": notification_id,
        "app_name": app,
        "sender": sender,
        "title": title,
        "body": body,
        "timestamp": timestamp,
    }


def filter_result(
    notification_id: str,
    *,
    is_passed: bool = False,
    urgency: int = 3,
    relevance: int = 3,
    category: str = "일반 업무",
) -> dict[str, object]:
    return {
        "notification_id": notification_id,
        "is_passed": is_passed,
        "urgency_score": urgency,
        "relevance_score": relevance,
        "category": category,
        "ai_summary_reason": "가상 테스트 판단",
    }


class BriefingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = SessionBriefingService()
        self.generated_at = datetime(2026, 9, 13, 20, 0, tzinfo=timezone.utc)

    def test_empty_input_returns_empty_briefing(self) -> None:
        briefing = self.service.build(
            session_id="session_empty",
            notifications=[],
            filter_results=[],
            generated_at=self.generated_at,
        ).to_dict()

        self.assertEqual(briefing["source_notification_count"], 0)
        self.assertEqual(briefing["blocked_notification_count"], 0)
        self.assertEqual(briefing["group_count"], 0)
        self.assertEqual(briefing["groups"], [])
        self.assertEqual(briefing["generated_at"], "2026-09-13T20:00:00Z")

    def test_invalid_score_range_is_rejected(self) -> None:
        invalid_scores = ({"urgency": 6}, {"relevance": 0})
        for invalid_score in invalid_scores:
            with self.subTest(invalid_score=invalid_score):
                with self.assertRaisesRegex(ContractValidationError, "between 1 and 5"):
                    self.service.build(
                        session_id="session_invalid",
                        notifications=[notification("n1")],
                        filter_results=[filter_result("n1", **invalid_score)],
                        generated_at=self.generated_at,
                    )

    def test_only_official_filter_categories_are_accepted(self) -> None:
        for category in FILTER_CATEGORIES:
            with self.subTest(category=category):
                briefing = self.service.build(
                    session_id="session_official_category",
                    notifications=[notification("n1")],
                    filter_results=[filter_result("n1", category=category)],
                    generated_at=self.generated_at,
                ).to_dict()
                self.assertEqual(
                    briefing["groups"][0]["primary_category"],
                    category,
                )

        with self.assertRaisesRegex(
            ContractValidationError,
            "official filtering categories",
        ):
            self.service.build(
                session_id="session_invalid_category",
                notifications=[notification("n1")],
                filter_results=[filter_result("n1", category="업무")],
                generated_at=self.generated_at,
            )

    def test_build_json_returns_parseable_structured_output(self) -> None:
        encoded = self.service.build_json(
            session_id="session_json",
            notifications=[notification("n1")],
            filter_results=[filter_result("n1")],
            generated_at=self.generated_at,
        )

        decoded = json.loads(encoded)
        self.assertEqual(decoded["session_id"], "session_json")
        self.assertEqual(decoded["group_count"], 1)
        self.assertEqual(decoded["groups"][0]["notification_ids"], ["n1"])

    def test_duplicate_notifications_are_counted_once(self) -> None:
        first = notification("n1")
        duplicate = notification("n2")
        briefing = self.service.build(
            session_id="session_duplicate",
            notifications=[first, duplicate],
            filter_results=[filter_result("n1"), filter_result("n2")],
            generated_at=self.generated_at,
        ).to_dict()

        self.assertEqual(briefing["blocked_notification_count"], 1)
        self.assertEqual(briefing["duplicate_count"], 1)
        self.assertEqual(briefing["groups"][0]["notification_ids"], ["n1"])

    def test_grouping_uses_app_sender_time_span_and_text(self) -> None:
        notifications = [
            notification("n1"),
            notification(
                "n2",
                title="테스트 서버 복구",
                body="테스트 서버 오류 복구를 진행합니다.",
                timestamp="2026-09-13T18:30:00Z",
            ),
            notification("n3", sender="다른 가상 사용자"),
            notification("n4", timestamp="2026-09-13T19:05:00Z"),
            notification(
                "n5",
                title="점심 메뉴 투표",
                body="점심 메뉴를 선택해 주세요.",
            ),
        ]
        results = [filter_result(item["id"]) for item in notifications]

        briefing = self.service.build(
            session_id="session_grouping",
            notifications=notifications,
            filter_results=results,
            generated_at=self.generated_at,
        ).to_dict()

        grouped_ids = [set(group["notification_ids"]) for group in briefing["groups"]]
        self.assertIn({"n1", "n2"}, grouped_ids)
        self.assertIn({"n3"}, grouped_ids)
        self.assertIn({"n4"}, grouped_ids)
        self.assertIn({"n5"}, grouped_ids)
        self.assertEqual(briefing["group_count"], 4)

    def test_adjacent_sequence_ids_group_short_chat_fragments_across_hours(self) -> None:
        notifications = [
            notification(
                "chat_001",
                app="KakaoTalk",
                sender="가상 팀원",
                title="첫말",
                body="내일 보자.",
                timestamp="2026-09-13T18:59:00Z",
            ),
            notification(
                "chat_002",
                app="KakaoTalk",
                sender="가상 팀원",
                title="둘째말",
                body="오후 세 시고.",
                timestamp="2026-09-13T19:01:00Z",
            ),
            notification(
                "chat_003",
                app="KakaoTalk",
                sender="가상 팀원",
                title="마지막말",
                body="B강의실로 와.",
                timestamp="2026-09-13T19:03:00Z",
            ),
        ]

        briefing = self.service.build(
            session_id="session_fragmented_chat",
            notifications=notifications,
            filter_results=[filter_result(item["id"]) for item in notifications],
            generated_at=self.generated_at,
        ).to_dict()

        self.assertEqual(briefing["group_count"], 1)
        self.assertEqual(
            briefing["groups"][0]["notification_ids"],
            ["chat_001", "chat_002", "chat_003"],
        )

    def test_shared_tokens_group_across_hour_boundary_within_thirty_minutes(self) -> None:
        notifications = [
            notification(
                "notice_010",
                title="프로젝트 공지",
                body="프로젝트 공지를 올립니다.",
                timestamp="2026-09-13T18:59:00Z",
            ),
            notification(
                "notice_020",
                title="프로젝트 자료",
                body="프로젝트 자료를 공유합니다.",
                timestamp="2026-09-13T19:05:00Z",
            ),
        ]

        briefing = self.service.build(
            session_id="session_cross_hour_tokens",
            notifications=notifications,
            filter_results=[filter_result(item["id"]) for item in notifications],
            generated_at=self.generated_at,
        ).to_dict()

        self.assertEqual(briefing["group_count"], 1)

    def test_shared_tokens_do_not_group_beyond_thirty_minute_span(self) -> None:
        notifications = [
            notification(
                "notice_010",
                title="프로젝트 공지",
                body="프로젝트 공지를 올립니다.",
                timestamp="2026-09-13T18:00:00Z",
            ),
            notification(
                "notice_020",
                title="프로젝트 자료",
                body="프로젝트 자료를 공유합니다.",
                timestamp="2026-09-13T18:31:00Z",
            ),
        ]

        briefing = self.service.build(
            session_id="session_long_span",
            notifications=notifications,
            filter_results=[filter_result(item["id"]) for item in notifications],
            generated_at=self.generated_at,
        ).to_dict()

        self.assertEqual(briefing["group_count"], 2)

    def test_sequence_grouping_rejects_unreliable_or_unrelated_ids(self) -> None:
        cases = {
            "non_adjacent": (
                notification("chat_001", title="첫번째", body="내일 보자."),
                notification(
                    "chat_003",
                    title="세번째",
                    body="오후에 와.",
                    timestamp="2026-09-13T18:07:00Z",
                ),
            ),
            "different_prefix": (
                notification("chat_a_001", title="첫번째", body="내일 보자."),
                notification(
                    "chat_b_002",
                    title="둘째",
                    body="오후에 와.",
                    timestamp="2026-09-13T18:07:00Z",
                ),
            ),
            "uuid": (
                notification(
                    "550e8400-e29b-41d4-a716-446655440001",
                    title="첫번째",
                    body="내일 보자.",
                ),
                notification(
                    "550e8400-e29b-41d4-a716-446655440002",
                    title="둘째",
                    body="오후에 와.",
                    timestamp="2026-09-13T18:07:00Z",
                ),
            ),
        }

        for case_name, notifications in cases.items():
            with self.subTest(case=case_name):
                briefing = self.service.build(
                    session_id=f"session_{case_name}",
                    notifications=list(notifications),
                    filter_results=[
                        filter_result(item["id"]) for item in notifications
                    ],
                    generated_at=self.generated_at,
                ).to_dict()
                self.assertEqual(briefing["group_count"], 2)

    def test_sequence_grouping_requires_close_time_and_same_sender(self) -> None:
        cases = {
            "time_gap": (
                notification("chat_001", title="첫번째", body="내일 보자."),
                notification(
                    "chat_002",
                    title="둘째",
                    body="오후에 와.",
                    timestamp="2026-09-13T18:11:00Z",
                ),
            ),
            "different_sender": (
                notification("chat_001", title="첫번째", body="내일 보자."),
                notification(
                    "chat_002",
                    sender="다른 가상 사용자",
                    title="둘째",
                    body="오후에 와.",
                    timestamp="2026-09-13T18:07:00Z",
                ),
            ),
            "different_app": (
                notification("chat_001", title="첫번째", body="내일 보자."),
                notification(
                    "chat_002",
                    app="KakaoTalk",
                    title="둘째",
                    body="오후에 와.",
                    timestamp="2026-09-13T18:07:00Z",
                ),
            ),
        }

        for case_name, notifications in cases.items():
            with self.subTest(case=case_name):
                briefing = self.service.build(
                    session_id=f"session_{case_name}",
                    notifications=list(notifications),
                    filter_results=[
                        filter_result(item["id"]) for item in notifications
                    ],
                    generated_at=self.generated_at,
                ).to_dict()
                self.assertEqual(briefing["group_count"], 2)

    def test_sequence_grouping_is_disabled_for_sender_fallback_to_app(self) -> None:
        notifications = [
            notification(
                "chat_001",
                app="KakaoTalk",
                sender="KakaoTalk",
                title="첫번째",
                body="내일 보자.",
            ),
            notification(
                "chat_002",
                app="KakaoTalk",
                sender="KakaoTalk",
                title="둘째",
                body="오후에 와.",
                timestamp="2026-09-13T18:07:00Z",
            ),
        ]

        briefing = self.service.build(
            session_id="session_sender_fallback",
            notifications=notifications,
            filter_results=[filter_result(item["id"]) for item in notifications],
            generated_at=self.generated_at,
        ).to_dict()

        self.assertEqual(briefing["group_count"], 2)

    def test_sequence_ids_do_not_merge_unrelated_calendar_app_events(self) -> None:
        notifications = [
            notification(
                "noti_001",
                app="Calendar",
                sender="일정 알림봇",
                title="치과 예약",
                body="내일 오전 10시 치과 예약입니다.",
            ),
            notification(
                "noti_002",
                app="Calendar",
                sender="일정 알림봇",
                title="프로젝트 회의",
                body="내일 오후 3시 프로젝트 회의입니다.",
                timestamp="2026-09-13T18:07:00Z",
            ),
        ]

        briefing = self.service.build(
            session_id="session_unrelated_calendar_events",
            notifications=notifications,
            filter_results=[filter_result(item["id"]) for item in notifications],
            generated_at=self.generated_at,
        ).to_dict()

        self.assertEqual(briefing["group_count"], 2)

    def test_category_is_decided_after_grouping_and_latest_breaks_tie(self) -> None:
        notifications = [
            notification(
                "n1",
                title="프로젝트 회의 공지",
                body="프로젝트 회의 관련 내용을 공유합니다.",
            ),
            notification(
                "n2",
                title="프로젝트 회의 일정 변경",
                body="프로젝트 회의 일정은 내일 15시로 변경됩니다.",
                timestamp="2026-09-13T18:30:00Z",
            ),
        ]
        results = [
            filter_result("n1", category="일반 업무", urgency=5, relevance=5),
            filter_result("n2", category="일정/회의", urgency=1, relevance=1),
        ]

        group = self.service.build(
            session_id="session_category",
            notifications=notifications,
            filter_results=results,
            generated_at=self.generated_at,
        ).to_dict()["groups"][0]

        self.assertEqual(group["primary_category"], "일정/회의")
        self.assertEqual(group["category_evidence_notification_ids"], ["n2"])

    def test_group_category_uses_majority_before_latest_notification(self) -> None:
        notifications = [
            notification("n1", timestamp="2026-09-13T18:05:00Z"),
            notification("n2", timestamp="2026-09-13T18:15:00Z"),
            notification("n3", timestamp="2026-09-13T18:30:00Z"),
        ]
        results = [
            filter_result("n1", category="시스템/보안"),
            filter_result("n2", category="시스템/보안"),
            filter_result("n3", category="긴급 업무"),
        ]

        group = self.service.build(
            session_id="session_category_majority",
            notifications=notifications,
            filter_results=results,
            generated_at=self.generated_at,
        ).to_dict()["groups"][0]

        self.assertEqual(group["primary_category"], "시스템/보안")
        self.assertEqual(
            group["category_evidence_notification_ids"],
            ["n1", "n2"],
        )

    def test_rule_based_summary_is_extractive_and_limited_to_three_lines(self) -> None:
        notifications = [
            notification(
                f"n{index}",
                title=f"공통 서버 상태 {index}",
                body=f"공통 서버 상태 알림 원문 {index}입니다.",
                timestamp=f"2026-09-13T18:{index:02d}:00Z",
            )
            for index in range(1, 5)
        ]
        results = [
            filter_result(f"n{index}", category="긴급 업무", urgency=index)
            for index in range(1, 5)
        ]

        group = self.service.build(
            session_id="session_extractive_summary",
            notifications=notifications,
            filter_results=results,
            generated_at=self.generated_at,
        ).to_dict()["groups"][0]

        self.assertLessEqual(len(group["summary_lines"]), 3)
        source_lines = {
            f"{item['title']} — {item['body']}" for item in notifications
        }
        self.assertTrue(set(group["summary_lines"]).issubset(source_lines))
        self.assertIn(
            "공통 서버 상태 4 — 공통 서버 상태 알림 원문 4입니다.",
            group["summary_lines"],
        )

    def test_blocked_personal_group_preserves_filter_category(self) -> None:
        group = self.service.build(
            session_id="session_chat",
            notifications=[
                notification(
                    "n1",
                    app="KakaoTalk",
                    sender="가상 친구",
                    title="저녁 메뉴",
                    body="저녁 메뉴를 같이 정해 보자.",
                )
            ],
            filter_results=[filter_result("n1", category="개인 일반")],
            generated_at=self.generated_at,
        ).to_dict()["groups"][0]

        self.assertEqual(group["primary_category"], "개인 일반")

    def test_fixture_filters_passed_items_and_aggregates_scores(self) -> None:
        briefing = self.service.build(
            session_id="session_fixture",
            notifications=load_fixture("raw_notifications.json"),
            filter_results=load_fixture("filter_results.json"),
            generated_at=self.generated_at,
        ).to_dict()

        all_ids = {
            notification_id
            for group in briefing["groups"]
            for notification_id in group["notification_ids"]
        }
        self.assertNotIn("noti_20260913_004", all_ids)
        self.assertNotIn("noti_20260913_006", all_ids)
        self.assertEqual(briefing["source_notification_count"], 6)
        self.assertEqual(briefing["blocked_notification_count"], 4)
        self.assertEqual(briefing["duplicate_count"], 1)

        server_group = next(
            group
            for group in briefing["groups"]
            if "noti_20260913_001" in group["notification_ids"]
        )
        self.assertEqual(server_group["urgency_score"], {"min": 4, "max": 5, "average": 4.5})
        self.assertEqual(server_group["relevance_score"], {"min": 4, "max": 4, "average": 4.0})
        self.assertEqual(server_group["primary_category"], "일반 업무")
        self.assertLessEqual(len(server_group["summary_lines"]), 3)

    def test_legacy_timezone_less_timestamp_is_accepted_and_output_as_utc(self) -> None:
        briefing = self.service.build(
            session_id="session_legacy_time",
            notifications=[notification("n1", timestamp="2026-09-13T18:05:00")],
            filter_results=[filter_result("n1")],
            generated_at=self.generated_at,
        ).to_dict()

        self.assertEqual(
            briefing["groups"][0]["time_bucket_start"],
            "2026-09-13T18:00:00Z",
        )


if __name__ == "__main__":
    unittest.main()
