"""Temporary, briefing-local data contracts for the first MVP.

These classes mirror the fields currently described for ``RawNotification`` and
``FilterResult`` without changing or claiming ownership of the shared contract.
They can later be replaced by adapters around the models owned by ``common``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


FILTER_CATEGORIES = (
    "긴급 업무",
    "일반 업무",
    "일정/회의",
    "시스템/보안",
    "개인 중요",
    "개인 일반",
    "광고/홍보",
    "기타",
)
_FILTER_CATEGORY_SET = frozenset(FILTER_CATEGORIES)


class ContractValidationError(ValueError):
    """Raised when input data cannot satisfy the temporary MVP contract."""


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp and normalize it to UTC.

    The README examples omit a timezone.  For backward-compatible MVP parsing,
    such values are temporarily interpreted as UTC.  New fixtures and generated
    output use an explicit ``Z`` suffix.
    """

    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError("timestamp must be a non-empty ISO 8601 string")

    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ContractValidationError(f"invalid ISO 8601 timestamp: {value!r}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    """Serialize a datetime as a UTC ISO 8601 string with a ``Z`` suffix."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _required_string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _filter_category(data: Mapping[str, Any]) -> str:
    category = _required_string(data, "category")
    if category not in _FILTER_CATEGORY_SET:
        allowed = ", ".join(FILTER_CATEGORIES)
        raise ContractValidationError(
            f"category must be one of the official filtering categories: {allowed}"
        )
    return category


@dataclass(frozen=True, slots=True)
class RawNotification:
    id: str
    app_name: str
    sender: str
    title: str
    body: str
    timestamp: datetime

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RawNotification":
        return cls(
            id=_required_string(data, "id"),
            app_name=_required_string(data, "app_name"),
            sender=_required_string(data, "sender"),
            title=_required_string(data, "title"),
            body=_required_string(data, "body"),
            timestamp=parse_timestamp(_required_string(data, "timestamp")),
        )


@dataclass(frozen=True, slots=True)
class FilterResult:
    notification_id: str
    is_passed: bool
    category: str
    ai_summary_reason: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FilterResult":
        is_passed = data.get("is_passed")
        if not isinstance(is_passed, bool):
            raise ContractValidationError("is_passed must be a boolean")
        return cls(
            notification_id=_required_string(data, "notification_id"),
            is_passed=is_passed,
            category=_filter_category(data),
            ai_summary_reason=_required_string(data, "ai_summary_reason"),
        )


@dataclass(frozen=True, slots=True)
class BriefingItem:
    notification: RawNotification
    filter_result: FilterResult


@dataclass(frozen=True, slots=True)
class CategoryDecision:
    primary_category: str
    evidence_notification_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BriefingGroup:
    group_id: str
    app_name: str
    sender: str
    time_bucket_start: datetime
    primary_category: str
    category_evidence_notification_ids: tuple[str, ...]
    notification_ids: tuple[str, ...]
    keywords: tuple[str, ...]
    summary_lines: tuple[str, ...]
    session_id: str = ""
    schedule_summaries: tuple[ScheduleSummary, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "session_id": self.session_id,
            "app_name": self.app_name,
            "sender": self.sender,
            "primary_category": self.primary_category,
            "summary_lines": list(self.summary_lines),
            "schedule_summaries": [
                summary.to_dict() for summary in self.schedule_summaries
            ],
        }


@dataclass(frozen=True, slots=True)
class ScheduleDetails:
    """Source-grounded six-question fields for a schedule notification."""

    who: str | None = None
    when: str | None = None
    where: str | None = None
    what: str | None = None
    why: str | None = None
    how: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "who": self.who,
            "when": self.when,
            "where": self.where,
            "what": self.what,
            "why": self.why,
            "how": self.how,
        }


@dataclass(frozen=True, slots=True)
class ScheduleSummary:
    """A schedule report that keeps the same six fields even when incomplete."""

    summary_id: str
    title: str
    status: str
    schedule_details: ScheduleDetails
    source_notification_ids: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    is_all_day: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "schedule_status": self.status,
            "is_all_day": self.is_all_day,
            **self.schedule_details.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SessionBriefing:
    session_id: str
    generated_at: datetime
    source_notification_count: int
    blocked_notification_count: int
    duplicate_count: int
    groups: tuple[BriefingGroup, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "groups": [group.to_dict() for group in self.groups],
        }
