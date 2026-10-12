from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.briefing.pipeline import SessionBriefingService
from src.briefing.sqlite_adapter import BriefingDatabaseError, SQLiteBriefingAdapter


class SQLiteBriefingAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self._temporary_directory.name) / "altong.db"
        self._create_database()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _create_database(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE notifications (
                    id TEXT PRIMARY KEY,
                    app_name TEXT NOT NULL,
                    sender TEXT,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    is_passed INTEGER,
                    urgency_score INTEGER,
                    relevance_score INTEGER,
                    category TEXT,
                    ai_summary_reason TEXT,
                    session_id TEXT,
                    created_at TEXT
                );

                CREATE TABLE focus_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    target_duration_minutes INTEGER NOT NULL,
                    blocked_count INTEGER DEFAULT 0,
                    is_completed INTEGER DEFAULT 0
                );
                """
            )
            connection.commit()

    def _insert_focus_session(
        self,
        session_id: str,
        started_at: str = "2026-09-23T09:00:00Z",
        ended_at: str | None = "2026-09-23T10:00:00Z",
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO focus_sessions (
                    session_id, started_at, ended_at, target_duration_minutes
                ) VALUES (?, ?, ?, 60);
                """,
                (session_id, started_at, ended_at),
            )
            connection.commit()

    def _insert_notification(
        self,
        notification_id: str,
        *,
        received_at: str,
        is_passed: bool,
        session_id: str | None,
        category: str = "일반 업무",
        sender: str | None = "가상 팀장",
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO notifications (
                    id, app_name, sender, title, body, received_at,
                    is_passed, urgency_score, relevance_score, category,
                    ai_summary_reason, session_id
                ) VALUES (?, 'Slack', ?, ?, ?, ?, ?, 4, 5, ?, ?, ?);
                """,
                (
                    notification_id,
                    sender,
                    f"테스트 알림 {notification_id}",
                    f"테스트 본문 {notification_id}",
                    received_at,
                    1 if is_passed else 0,
                    category,
                    "가상 필터 판단",
                    session_id,
                ),
            )
            connection.commit()

    def test_load_session_selects_only_blocked_notifications_for_session(self) -> None:
        self._insert_focus_session("session_1")
        self._insert_notification(
            "blocked",
            received_at="2026-09-23T09:10:00Z",
            is_passed=False,
            session_id="session_1",
            category="긴급 업무",
        )
        self._insert_notification(
            "passed",
            received_at="2026-09-23T09:20:00Z",
            is_passed=True,
            session_id="session_1",
        )
        self._insert_notification(
            "other_session",
            received_at="2026-09-23T09:30:00Z",
            is_passed=False,
            session_id="session_2",
        )

        source = SQLiteBriefingAdapter(self.database_path).load_session("session_1")

        self.assertEqual([item["id"] for item in source.notifications], ["blocked"])
        self.assertEqual(
            [item["notification_id"] for item in source.filter_results],
            ["blocked"],
        )
        self.assertFalse(source.used_time_range_fallback)

        briefing = SessionBriefingService().build(
            session_id=source.session_id,
            notifications=source.notifications,
            filter_results=source.filter_results,
        )
        self.assertEqual(sum(len(group.notification_ids) for group in briefing.groups), 1)
        # Stored filtering category is no longer the briefing category source.
        self.assertEqual(briefing.to_dict()["groups"][0]["primary_category"], "기타")

    def test_load_session_uses_only_unassigned_rows_inside_focus_period(self) -> None:
        self._insert_focus_session("session_1")
        self._insert_notification(
            "inside_unassigned",
            received_at="2026-09-23T09:15:00Z",
            is_passed=False,
            session_id=None,
        )
        self._insert_notification(
            "outside_unassigned",
            received_at="2026-09-23T10:15:00Z",
            is_passed=False,
            session_id=None,
        )
        self._insert_notification(
            "inside_other_session",
            received_at="2026-09-23T09:30:00Z",
            is_passed=False,
            session_id="session_2",
        )

        source = SQLiteBriefingAdapter(self.database_path).load_session("session_1")

        self.assertEqual(
            [item["id"] for item in source.notifications],
            ["inside_unassigned"],
        )
        self.assertTrue(source.used_time_range_fallback)

    def test_missing_sender_uses_app_name_for_stable_grouping(self) -> None:
        self._insert_focus_session("session_1")
        self._insert_notification(
            "missing_sender",
            received_at="2026-09-23T09:15:00Z",
            is_passed=False,
            session_id="session_1",
            sender=None,
        )

        source = SQLiteBriefingAdapter(self.database_path).load_session("session_1")

        self.assertEqual(source.notifications[0]["sender"], "Slack")
        briefing = SessionBriefingService().build(
            session_id=source.session_id,
            notifications=source.notifications,
            filter_results=source.filter_results,
        ).to_dict()
        self.assertEqual(briefing["groups"][0]["sender"], "Slack")

    def test_active_session_does_not_use_time_range_fallback(self) -> None:
        self._insert_focus_session("active_session", ended_at=None)
        self._insert_notification(
            "unassigned",
            received_at="2026-09-23T09:15:00Z",
            is_passed=False,
            session_id=None,
        )

        source = SQLiteBriefingAdapter(self.database_path).load_session(
            "active_session"
        )

        self.assertEqual(source.notifications, ())
        self.assertFalse(source.used_time_range_fallback)

    def test_missing_database_is_rejected_without_creating_a_file(self) -> None:
        missing_path = Path(self._temporary_directory.name) / "missing.db"

        with self.assertRaisesRegex(BriefingDatabaseError, "does not exist"):
            SQLiteBriefingAdapter(missing_path).load_session("session_1")

        self.assertFalse(missing_path.exists())


if __name__ == "__main__":
    unittest.main()
