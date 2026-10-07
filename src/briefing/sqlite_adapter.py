"""Read-only adapter from the Altong client SQLite DB to briefing inputs."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3


class BriefingDatabaseError(RuntimeError):
    """Raised when the client database cannot provide briefing source data."""


@dataclass(frozen=True, slots=True)
class SessionBriefingInput:
    """Mappings accepted directly by ``SessionBriefingService.build``."""

    session_id: str
    notifications: tuple[dict[str, object], ...]
    filter_results: tuple[dict[str, object], ...]
    used_time_range_fallback: bool = False


class SQLiteBriefingAdapter:
    """Load blocked notifications without modifying the client database.

    Records explicitly assigned to ``session_id`` are authoritative.  During
    the client MVP, older records may have a null ``session_id``; those records
    are included only when they fall inside the matching focus session period.
    Records assigned to another session are never included by the fallback.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path).expanduser()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def load_session(self, session_id: str) -> SessionBriefingInput:
        normalized_session_id = self._validate_session_id(session_id)

        try:
            with closing(self._connect_read_only()) as connection:
                period = self._focus_session_period(connection, normalized_session_id)
                rows = self._blocked_rows(
                    connection,
                    normalized_session_id,
                    period,
                )
        except sqlite3.Error as exc:
            raise BriefingDatabaseError(
                f"failed to read briefing data from {self._database_path}"
            ) from exc

        notifications: list[dict[str, object]] = []
        filter_results: list[dict[str, object]] = []
        used_time_range_fallback = False

        for row in rows:
            if row["session_id"] is None:
                used_time_range_fallback = True

            notification_id = row["id"]
            sender = row["sender"]
            if not isinstance(sender, str) or not sender.strip():
                sender = row["app_name"]
            notifications.append(
                {
                    "id": notification_id,
                    "app_name": row["app_name"],
                    "sender": sender,
                    "title": row["title"],
                    "body": row["body"],
                    "timestamp": row["received_at"],
                }
            )
            filter_results.append(
                {
                    "notification_id": notification_id,
                    "is_passed": False,
                    "category": row["category"],
                    "ai_summary_reason": row["ai_summary_reason"],
                }
            )

        return SessionBriefingInput(
            session_id=normalized_session_id,
            notifications=tuple(notifications),
            filter_results=tuple(filter_results),
            used_time_range_fallback=used_time_range_fallback,
        )

    def _connect_read_only(self) -> sqlite3.Connection:
        resolved = self._database_path.resolve()
        if not resolved.is_file():
            raise BriefingDatabaseError(f"database file does not exist: {resolved}")

        connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        return session_id.strip()

    @staticmethod
    def _focus_session_period(
        connection: sqlite3.Connection,
        session_id: str,
    ) -> tuple[str, str] | None:
        row = connection.execute(
            """
            SELECT started_at, ended_at
            FROM focus_sessions
            WHERE session_id = ?
            LIMIT 1;
            """,
            (session_id,),
        ).fetchone()

        if row is None or row["ended_at"] is None:
            return None
        return str(row["started_at"]), str(row["ended_at"])

    @staticmethod
    def _blocked_rows(
        connection: sqlite3.Connection,
        session_id: str,
        period: tuple[str, str] | None,
    ) -> list[sqlite3.Row]:
        fields = """
            id, app_name, sender, title, body, received_at,
            category, ai_summary_reason,
            session_id
        """

        if period is None:
            query = f"""
                SELECT {fields}
                FROM notifications
                WHERE is_passed = 0 AND session_id = ?
                ORDER BY julianday(received_at) ASC, id ASC;
            """
            parameters = (session_id,)
        else:
            query = f"""
                SELECT {fields}
                FROM notifications
                WHERE is_passed = 0
                  AND (
                    session_id = ?
                    OR (
                      session_id IS NULL
                      AND julianday(received_at) >= julianday(?)
                      AND julianday(received_at) < julianday(?)
                    )
                  )
                ORDER BY julianday(received_at) ASC, id ASC;
            """
            parameters = (session_id, period[0], period[1])

        return list(connection.execute(query, parameters).fetchall())
