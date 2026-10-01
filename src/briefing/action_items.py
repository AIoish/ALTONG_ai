"""Deterministic To-Do and calendar candidate extraction for the MVP.

The rules deliberately favor traceable candidates over aggressive extraction.
This provider can later be replaced by an LLM-backed implementation without
changing the session briefing contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
from typing import Protocol, Sequence
import unicodedata

from .schema import (
    BriefingItem,
    CalendarCandidate,
    ScheduleDetails,
    ScheduleSummary,
    TodoCandidate,
    format_timestamp,
)


_TODO_CUES = (
    "해 주세요",
    "해주세요",
    "부탁",
    "필요",
    "해야",
    "제출",
    "확인",
    "수정",
    "작성",
    "전달",
    "완료",
    "재시도",
    "등록",
)
_CALENDAR_CUES = (
    "회의",
    "일정",
    "마감",
    "기한",
    "시험",
    "발표",
    "면담",
    "약속",
    "예약",
    "수업",
    "세미나",
    "행사",
    "제출",
)
_RELATIVE_DAYS = {"오늘": 0, "내일": 1, "모레": 2}
_DATE_PATTERN = re.compile(
    r"(?:(?P<year>\d{4})\s*(?:년|[./-])\s*)?"
    r"(?P<month>\d{1,2})\s*(?:월|[./-])\s*"
    r"(?P<day>\d{1,2})\s*일?"
)
_AMPM_TIME_PATTERN = re.compile(
    r"(?P<ampm>오전|오후)\s*(?P<hour>\d{1,2})\s*시"
    r"(?:\s*(?P<minute>[0-5]?\d)\s*분)?"
)
_COLON_TIME_PATTERN = re.compile(
    r"(?<!\d)(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)(?!\d)"
)
_HOUR_TIME_PATTERN = re.compile(
    r"(?<!\d)(?P<hour>[01]?\d|2[0-3])\s*시"
    r"(?:\s*(?P<minute>[0-5]?\d)\s*분)?"
)
_WHO_PATTERN = re.compile(
    r"(?:참석자|대상|담당자|누가)\s*(?:는|은|이|:)\s*"
    r"(.+?)(?=\s*(?:이고|이며|,|[.!?]|$))"
)
_WHERE_PATTERN = re.compile(
    r"(?:장소|위치)\s*(?:는|은|이|:)\s*"
    r"(.+?)(?=\s*(?:이고|이며|,|[.!?]|$))"
)
_WHAT_PATTERN = re.compile(
    r"(?:일정 내용|회의 안건|안건|내용|무엇)\s*(?:는|은|이|:)\s*"
    r"(.+?)(?=\s*(?:이고|이며|,|[.!?]|$))"
)
_WHY_PATTERN = re.compile(
    r"(?:이유|목적)\s*(?:는|은|이|:)\s*"
    r"(.+?)(?=\s*(?:이고|이며|,|[.!?]|$))"
)
_WHY_BECAUSE_PATTERN = re.compile(r"([^.!?]+?)\s*때문에")
_HOW_PATTERN = re.compile(
    r"(?:방식|방법)\s*(?:는|은|이|:)\s*"
    r"(.+?)(?=\s*(?:이고|이며|,|[.!?]|$))"
)
_HOW_KEYWORDS = (
    "대면",
    "오프라인",
    "온라인",
    "화상",
    "Zoom",
    "Teams",
    "전화",
)
_NATURAL_WHO_WHAT_PATTERN = re.compile(
    r"([^.!?]+?)(?:과|와)\s+([^.!?]+?)\s*(?:을|를)?\s*"
    r"(?:진행|개최|함께|만나)"
)
_NATURAL_WHO_WHAT_FALLBACK_PATTERN = re.compile(
    r"([^.!?]+?)(?:과|와)\s+([^.!?]+?)(?:해요|합니다)(?=[.!?]|$)"
)
_NATURAL_WHERE_PATTERN = re.compile(
    r"([0-9A-Za-z가-힣 ]+?(?:강의실|회의실|세미나실|공학관|창의관|\d호|\d층))"
    r"\s*(?:로|에서)\s*(?:와|오|진행|모여|만나)"
)
_NATURAL_WHY_PATTERN = re.compile(r"([^.!?]+?)(?:을|를)\s*위해")
_CANCELLED_CUES = ("취소", "무산")
_CHANGED_CUES = ("변경", "정정", "연기", "순연", "앞당")
_RESCHEDULED_CUES = ("재개", "다시 진행", "재예정")


@dataclass(frozen=True, slots=True)
class CandidateExtraction:
    todos: tuple[TodoCandidate, ...] = ()
    calendar: tuple[CalendarCandidate, ...] = ()
    schedules: tuple[ScheduleSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class _TemporalMatch:
    value: datetime
    is_all_day: bool


class ActionItemProvider(Protocol):
    """Replaceable boundary for To-Do and calendar candidate extraction."""

    def extract(
        self,
        *,
        group_id: str,
        items: Sequence[BriefingItem],
    ) -> CandidateExtraction: ...


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _candidate_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}|{key}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}_{digest}"


def _display_text(item: BriefingItem) -> str:
    notification = item.notification
    text = f"{notification.title} — {notification.body}".strip()
    if len(text) > 240:
        return f"{text[:237].rstrip()}..."
    return text


def _matched_cues(text: str, cues: tuple[str, ...]) -> tuple[str, ...]:
    normalized = _normalize(text)
    return tuple(cue for cue in cues if cue in normalized)


def _parse_time(text: str) -> tuple[int, int] | None:
    ampm_match = _AMPM_TIME_PATTERN.search(text)
    if ampm_match:
        hour = int(ampm_match.group("hour"))
        if not 1 <= hour <= 12:
            return None
        if ampm_match.group("ampm") == "오전":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
        return hour, int(ampm_match.group("minute") or 0)

    colon_match = _COLON_TIME_PATTERN.search(text)
    if colon_match:
        return int(colon_match.group("hour")), int(colon_match.group("minute"))

    hour_match = _HOUR_TIME_PATTERN.search(text)
    if hour_match:
        return int(hour_match.group("hour")), int(hour_match.group("minute") or 0)
    return None


def _parse_last_time(text: str) -> tuple[int, int] | None:
    """Return the last stated clock time, used for schedule corrections."""

    candidates: list[tuple[int, int, int]] = []
    ampm_matches = list(_AMPM_TIME_PATTERN.finditer(text))
    for match in ampm_matches:
        hour = int(match.group("hour"))
        if not 1 <= hour <= 12:
            continue
        if match.group("ampm") == "오전":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
        candidates.append((match.start(), hour, int(match.group("minute") or 0)))
    for match in _COLON_TIME_PATTERN.finditer(text):
        candidates.append(
            (match.start(), int(match.group("hour")), int(match.group("minute")))
        )
    for match in _HOUR_TIME_PATTERN.finditer(text):
        if any(
            ampm_match.start() <= match.start() < ampm_match.end()
            for ampm_match in ampm_matches
        ):
            continue
        candidates.append(
            (
                match.start(),
                int(match.group("hour")),
                int(match.group("minute") or 0),
            )
        )
    if not candidates:
        return None
    _, hour, minute = max(candidates, key=lambda candidate: candidate[0])
    return hour, minute


def _parse_temporal(text: str, base: datetime) -> _TemporalMatch | None:
    base_utc = base.astimezone(timezone.utc)
    date_match = _DATE_PATTERN.search(text)
    relative_match = next((word for word in _RELATIVE_DAYS if word in text), None)
    parsed_time = _parse_time(text)

    if date_match:
        year = int(date_match.group("year") or base_utc.year)
        month = int(date_match.group("month"))
        day = int(date_match.group("day"))
    elif relative_match:
        relative_date = base_utc.date() + timedelta(days=_RELATIVE_DAYS[relative_match])
        year, month, day = relative_date.year, relative_date.month, relative_date.day
    elif parsed_time:
        year, month, day = base_utc.year, base_utc.month, base_utc.day
    else:
        return None

    hour, minute = parsed_time or (0, 0)
    try:
        value = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None
    return _TemporalMatch(value=value, is_all_day=parsed_time is None)


def _parse_group_temporal(items: Sequence[BriefingItem]) -> _TemporalMatch | None:
    """Combine a date and time that may be split across adjacent messages."""

    latest_date: tuple[int, int, int] | None = None
    latest_time: tuple[int, int] | None = None
    latest_time_base: datetime | None = None

    for item in items:
        notification = item.notification
        text = f"{notification.title} {notification.body}"
        base = notification.timestamp.astimezone(timezone.utc)
        date_matches = list(_DATE_PATTERN.finditer(text))
        date_match = date_matches[-1] if date_matches else None
        relative_positions = [
            (text.rfind(word), word) for word in _RELATIVE_DAYS if word in text
        ]
        relative_position, relative_match = max(
            relative_positions,
            default=(-1, None),
        )
        date_position = date_match.start() if date_match else -1

        if date_match and date_position > relative_position:
            latest_date = (
                int(date_match.group("year") or base.year),
                int(date_match.group("month")),
                int(date_match.group("day")),
            )
        elif relative_match:
            relative_date = base.date() + timedelta(days=_RELATIVE_DAYS[relative_match])
            latest_date = (
                relative_date.year,
                relative_date.month,
                relative_date.day,
            )

        parsed_time = _parse_last_time(text)
        if parsed_time is not None:
            latest_time = parsed_time
            latest_time_base = base

    if latest_date is None and latest_time is None:
        return None

    if latest_date is None:
        assert latest_time_base is not None
        latest_date = (
            latest_time_base.year,
            latest_time_base.month,
            latest_time_base.day,
        )

    hour, minute = latest_time or (0, 0)
    try:
        value = datetime(*latest_date, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None
    return _TemporalMatch(value=value, is_all_day=latest_time is None)


def _clean_detail(value: str) -> str | None:
    cleaned = value.strip(" \t\r\n,.;:!?")
    cleaned = re.sub(r"(?:입니다|이에요|예요)$", "", cleaned).strip()
    return cleaned or None


def _extract_detail(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return _clean_detail(match.group(1)) if match else None


def _natural_who_what(text: str) -> tuple[str | None, str | None]:
    match = _NATURAL_WHO_WHAT_PATTERN.search(text)
    if match is None:
        match = _NATURAL_WHO_WHAT_FALLBACK_PATTERN.search(text)
    if match is None:
        return None, None
    who = _clean_detail(match.group(1))
    what = _clean_detail(match.group(2))
    if what:
        what = re.sub(r"\s*(?:진행|개최)$", "", what).strip() or None
    return who, what


def _calendar_status(items: Sequence[BriefingItem]) -> str:
    status = "scheduled"
    for item in items:
        text = _normalize(f"{item.notification.title} {item.notification.body}")
        matches = [
            *((text.rfind(cue), "cancelled") for cue in _CANCELLED_CUES),
            *((text.rfind(cue), "changed") for cue in _CHANGED_CUES),
            *((text.rfind(cue), "scheduled") for cue in _RESCHEDULED_CUES),
        ]
        position, latest_status = max(matches, key=lambda match: match[0])
        if position >= 0:
            status = latest_status
    return status


def _calendar_title(items: Sequence[BriefingItem]) -> str:
    combined = " ".join(item.notification.body for item in items)
    _, natural_what = _natural_who_what(combined)
    if natural_what:
        return natural_what
    for item in items:
        source_text = f"{item.notification.title} {item.notification.body}"
        if _matched_cues(source_text, _CALENDAR_CUES):
            return item.notification.title
    return items[0].notification.title


def _schedule_details(
    items: Sequence[BriefingItem],
    *,
    title: str,
    temporal: _TemporalMatch | None,
) -> ScheduleDetails:
    combined = " ".join(
        f"{item.notification.title}. {item.notification.body}" for item in items
    )
    natural_who, natural_what = _natural_who_what(combined)
    why = _extract_detail(_WHY_PATTERN, combined)
    if why is None:
        why = _extract_detail(_WHY_BECAUSE_PATTERN, combined)
    if why is None:
        why = _extract_detail(_NATURAL_WHY_PATTERN, combined)
    how = _extract_detail(_HOW_PATTERN, combined)
    if how is None:
        how = next((keyword for keyword in _HOW_KEYWORDS if keyword in combined), None)
    who = _extract_detail(_WHO_PATTERN, combined)
    what = _extract_detail(_WHAT_PATTERN, combined)
    who = who or natural_who
    what = what or natural_what
    where = _extract_detail(_WHERE_PATTERN, combined)
    if where is None:
        where = _extract_detail(_NATURAL_WHERE_PATTERN, combined)
    return ScheduleDetails(
        who=who,
        when=format_timestamp(temporal.value) if temporal else None,
        where=where,
        what=what or title,
        why=why,
        how=how,
    )


class RuleBasedActionItemProvider:
    """Extract conservative, source-grounded candidates from blocked items."""

    def extract(
        self,
        *,
        group_id: str,
        items: Sequence[BriefingItem],
    ) -> CandidateExtraction:
        todos: list[TodoCandidate] = []

        for item in items:
            notification = item.notification
            source_text = f"{notification.title} {notification.body}"
            temporal = _parse_temporal(source_text, notification.timestamp)
            todo_cues = _matched_cues(source_text, _TODO_CUES)

            if todo_cues:
                text = _display_text(item)
                temporal_key = temporal.value.isoformat() if temporal else "no-due-date"
                key = f"{_normalize(text)}|{temporal_key}"
                todos.append(
                    TodoCandidate(
                        candidate_id=_candidate_id("todo", key),
                        text=text,
                        due_at=temporal.value if temporal else None,
                        is_all_day=temporal.is_all_day if temporal else False,
                        source_notification_ids=(notification.id,),
                        source_group_ids=(group_id,),
                        matched_cues=todo_cues,
                    )
                )

        ordered = tuple(sorted(items, key=lambda item: item.notification.timestamp))
        combined_text = " ".join(
            f"{item.notification.title} {item.notification.body}" for item in ordered
        )
        temporal = _parse_group_temporal(ordered)
        calendar_cues = _matched_cues(combined_text, _CALENDAR_CUES)
        if any(item.filter_result.category == "일정/회의" for item in ordered):
            calendar_cues = tuple(dict.fromkeys((*calendar_cues, "일정/회의")))
        calendar: tuple[CalendarCandidate, ...] = ()
        schedules: tuple[ScheduleSummary, ...] = ()
        if calendar_cues:
            title = _calendar_title(ordered)
            details = _schedule_details(ordered, title=title, temporal=temporal)
            status = _calendar_status(ordered)
            source_ids = tuple(item.notification.id for item in ordered)
            schedules = (
                ScheduleSummary(
                    summary_id=_candidate_id("schedule", group_id),
                    title=title,
                    status=status,
                    schedule_details=details,
                    source_notification_ids=source_ids,
                    source_group_ids=(group_id,),
                ),
            )
            if temporal:
                first = ordered[0].notification
                key = "|".join(
                    (
                        _normalize(first.app_name),
                        _normalize(first.sender),
                        _normalize(title),
                        temporal.value.isoformat(),
                    )
                )
                calendar = (
                    CalendarCandidate(
                        candidate_id=_candidate_id("calendar", key),
                        title=title,
                        scheduled_at=temporal.value,
                        is_all_day=temporal.is_all_day,
                        status=status,
                        schedule_details=details,
                        source_notification_ids=source_ids,
                        source_group_ids=(group_id,),
                        matched_cues=calendar_cues,
                    ),
                )

        return CandidateExtraction(
            todos=tuple(todos),
            calendar=calendar,
            schedules=schedules,
        )


def merge_candidates(
    extractions: Sequence[CandidateExtraction],
) -> CandidateExtraction:
    """Merge repeated candidates while preserving every source reference."""

    todos: dict[str, TodoCandidate] = {}
    calendar: dict[str, CalendarCandidate] = {}
    schedules: dict[str, ScheduleSummary] = {}

    for extraction in extractions:
        for candidate in extraction.todos:
            existing = todos.get(candidate.candidate_id)
            if existing is None:
                todos[candidate.candidate_id] = candidate
                continue
            todos[candidate.candidate_id] = TodoCandidate(
                candidate_id=existing.candidate_id,
                text=existing.text,
                due_at=existing.due_at or candidate.due_at,
                is_all_day=existing.is_all_day or candidate.is_all_day,
                source_notification_ids=tuple(
                    sorted(
                        set(
                            existing.source_notification_ids
                            + candidate.source_notification_ids
                        )
                    )
                ),
                source_group_ids=tuple(
                    sorted(set(existing.source_group_ids + candidate.source_group_ids))
                ),
                matched_cues=tuple(
                    sorted(set(existing.matched_cues + candidate.matched_cues))
                ),
            )

        for candidate in extraction.calendar:
            existing = calendar.get(candidate.candidate_id)
            if existing is None:
                calendar[candidate.candidate_id] = candidate
                continue
            calendar[candidate.candidate_id] = CalendarCandidate(
                candidate_id=existing.candidate_id,
                title=existing.title,
                scheduled_at=existing.scheduled_at,
                is_all_day=existing.is_all_day,
                status=candidate.status,
                schedule_details=ScheduleDetails(
                    who=candidate.schedule_details.who or existing.schedule_details.who,
                    when=candidate.schedule_details.when or existing.schedule_details.when,
                    where=(
                        candidate.schedule_details.where
                        or existing.schedule_details.where
                    ),
                    what=candidate.schedule_details.what or existing.schedule_details.what,
                    why=candidate.schedule_details.why or existing.schedule_details.why,
                    how=candidate.schedule_details.how or existing.schedule_details.how,
                ),
                source_notification_ids=tuple(
                    sorted(
                        set(
                            existing.source_notification_ids
                            + candidate.source_notification_ids
                        )
                    )
                ),
                source_group_ids=tuple(
                    sorted(set(existing.source_group_ids + candidate.source_group_ids))
                ),
                matched_cues=tuple(
                    sorted(set(existing.matched_cues + candidate.matched_cues))
                ),
            )

        for summary in extraction.schedules:
            schedules[summary.summary_id] = summary

    todo_values = tuple(
        sorted(
            todos.values(),
            key=lambda candidate: (
                candidate.due_at is None,
                candidate.due_at or datetime.max.replace(tzinfo=timezone.utc),
                candidate.text,
            ),
        )
    )
    calendar_values = tuple(
        sorted(
            calendar.values(),
            key=lambda candidate: (candidate.scheduled_at, candidate.title),
        )
    )
    return CandidateExtraction(
        todos=todo_values,
        calendar=calendar_values,
        schedules=tuple(schedules.values()),
    )
