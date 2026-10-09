"""Deterministic schedule extraction for dashboard briefing cards.

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
    ScheduleDetails,
    ScheduleSummary,
    format_timestamp,
)
from .clustering import app_identity


def _source_text(item: BriefingItem) -> str:
    notification = item.notification
    if app_identity(notification.app_name) == "kakaotalk":
        return notification.body
    return f"{notification.title} {notification.body}"


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
    "만나요",
    "스터디",
    "멘토링",
    "오리엔테이션",
)
_SCHEDULE_TIMEZONE = timezone(timedelta(hours=9), name="Asia/Seoul")
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
class ScheduleExtraction:
    schedules: tuple[ScheduleSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class _TemporalMatch:
    value: datetime
    is_all_day: bool
    end: datetime | None = None


@dataclass(frozen=True, slots=True)
class _Clock:
    start: int
    end: int
    hour: int
    minute: int
    ampm: str | None = None


class ActionItemProvider(Protocol):
    """Replaceable boundary for schedule extraction; no calendar writes."""

    def extract(
        self,
        *,
        group_id: str,
        items: Sequence[BriefingItem],
    ) -> ScheduleExtraction: ...


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _candidate_id(kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{kind}|{key}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}_{digest}"


def _matched_cues(text: str, cues: tuple[str, ...]) -> tuple[str, ...]:
    normalized = _normalize(text)
    return tuple(cue for cue in cues if cue in normalized)


def _clock_matches(text: str) -> list[_Clock]:
    candidates: list[_Clock] = []
    ampm_matches = list(_AMPM_TIME_PATTERN.finditer(text))
    for match in ampm_matches:
        hour = int(match.group("hour"))
        if not 1 <= hour <= 12:
            continue
        if match.group("ampm") == "오전":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
        candidates.append(_Clock(match.start(), match.end(), hour,
                                 int(match.group("minute") or 0), match.group("ampm")))
    for match in _COLON_TIME_PATTERN.finditer(text):
        candidates.append(
            _Clock(match.start(), match.end(), int(match.group("hour")), int(match.group("minute")))
        )
    for match in _HOUR_TIME_PATTERN.finditer(text):
        if any(
            ampm_match.start() <= match.start() < ampm_match.end()
            for ampm_match in ampm_matches
        ):
            continue
        candidates.append(
            _Clock(
                match.start(), match.end(),
                int(match.group("hour")),
                int(match.group("minute") or 0),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.start)


def _date_in(text: str, base: datetime) -> tuple[int, int, int] | None:
    dates = list(_DATE_PATTERN.finditer(text))
    match = dates[-1] if dates else None
    position, relative = max(
        ((text.rfind(word), word) for word in _RELATIVE_DAYS if word in text),
        default=(-1, None),
    )
    if match and match.start() > position:
        return (int(match.group("year") or base.year),
                int(match.group("month")), int(match.group("day")))
    if relative:
        day = base.date() + timedelta(days=_RELATIVE_DAYS[relative])
        return day.year, day.month, day.day
    return None


def _time_range(text: str, clocks: list[_Clock]) -> tuple[_Clock, _Clock, str] | None:
    """Recognize explicit ranges, not '2시에서 4시로 변경' corrections."""
    ranges = []
    for start, end in zip(clocks, clocks[1:]):
        between = text[start.end:end.start]
        connector = _DATE_PATTERN.sub("", between)
        connector = re.sub(r"다음\s*날|익일|오늘|내일|모레", "", connector).strip()
        if connector not in ("부터", "~", "～", "-", "–", "—"):
            continue
        if connector == "부터" and not re.match(r"\s*까지", text[end.end:]):
            continue
        if end.ampm is None and start.ampm and end.hour <= 12:
            hour = end.hour % 12 + (12 if start.ampm == "오후" else 0)
            end = _Clock(end.start, end.end, hour, end.minute, start.ampm)
        ranges.append((start, end, between))
    return ranges[-1] if ranges else None


def _is_end_clock(text: str, clock: _Clock) -> bool:
    before = text[:clock.start]
    after = text[clock.end:]
    return bool(
        re.search(r"(?:종료|끝나는|마치는)\s*(?:시각|시간)?\s*(?:은|는|:)?\s*$", before)
        or re.match(r"\s*(?:에\s*)?(?:종료|끝|마칩니다|마쳐요)", after)
    )


def _parse_group_temporal(items: Sequence[BriefingItem]) -> _TemporalMatch | None:
    """Combine a date and time that may be split across adjacent messages."""

    latest_date: tuple[int, int, int] | None = None
    latest_time: tuple[int, int] | None = None
    latest_time_base: datetime | None = None
    end_time: tuple[int, int] | None = None
    end_date: tuple[int, int, int] | None = None
    next_day_end = False

    for item in items:
        notification = item.notification
        text = _source_text(item)
        base = notification.timestamp.astimezone(_SCHEDULE_TIMEZONE)
        clocks = _clock_matches(text)
        time_range = _time_range(text, clocks)
        if (time_range and clocks[-1].start > time_range[1].start
                and not _is_end_clock(text, clocks[-1])):
            # A later standalone correction supersedes an earlier interval.
            time_range = None
        if clocks and not time_range and _is_end_clock(text, clocks[-1]):
            start_clocks = [clock for clock in clocks[:-1] if not _is_end_clock(text, clock)]
            if start_clocks:
                start, end = start_clocks[-1], clocks[-1]
                time_range = (start, end, text[start.end:end.start])
        if clocks and not time_range and _is_end_clock(text, clocks[-1]):
            clock = clocks[-1]
            end_time = (clock.hour, clock.minute)
            end_date = _date_in(text, base)
            next_day_end = bool(re.search(r"다음\s*날|익일", text))
            continue

        date_text = text[:time_range[0].start] if time_range else text
        parsed_date = _date_in(date_text, base)
        parsed_clock = time_range[0] if time_range else clocks[-1] if clocks else None
        parsed_time = (parsed_clock.hour, parsed_clock.minute) if parsed_clock else None
        # A changed start invalidates a previous end unless a new end is stated.
        if ((parsed_date is not None and parsed_date != latest_date)
                or (parsed_time is not None and parsed_time != latest_time)):
            end_time = end_date = None
            next_day_end = False
        if parsed_date is not None:
            latest_date = parsed_date
        if parsed_time is not None:
            latest_time = parsed_time
            latest_time_base = base
        if time_range:
            _, end, between = time_range
            end_time = (end.hour, end.minute)
            end_date = _date_in(between, base)
            next_day_end = bool(re.search(r"다음\s*날|익일", between))

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
        value = datetime(
            *latest_date, hour, minute, tzinfo=_SCHEDULE_TIMEZONE
        ).astimezone(timezone.utc)
    except ValueError:
        return None
    end_value = None
    if latest_time is not None and end_time is not None:
        try:
            day = datetime(*(end_date or latest_date), tzinfo=_SCHEDULE_TIMEZONE)
            if next_day_end and end_date is None:
                day += timedelta(days=1)
            candidate = day.replace(hour=end_time[0], minute=end_time[1]).astimezone(timezone.utc)
            if candidate > value:
                end_value = candidate
        except ValueError:
            pass
    return _TemporalMatch(value=value, is_all_day=latest_time is None, end=end_value)


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
        text = _normalize(_source_text(item))
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
    explicit = _extract_detail(_WHAT_PATTERN, combined)
    if explicit:
        return explicit
    for item in items:
        source_text = _source_text(item)
        if _matched_cues(source_text, _CALENDAR_CUES):
            if app_identity(item.notification.app_name) != "kakaotalk":
                return item.notification.title
            # A room title is not an event title. Extract a short source-backed
            # noun phrase, ignoring attendee/location sentences and dates.
            for sentence in re.split(r"[.!?\n]", item.notification.body):
                if _WHO_PATTERN.search(sentence) or _WHERE_PATTERN.search(sentence):
                    continue
                cleaned = _DATE_PATTERN.sub(" ", sentence)
                cleaned = _AMPM_TIME_PATTERN.sub(" ", cleaned)
                cleaned = _COLON_TIME_PATTERN.sub(" ", cleaned)
                cleaned = _HOUR_TIME_PATTERN.sub(" ", cleaned)
                cleaned = re.sub(r"오늘|내일|모레", " ", cleaned)
                cleaned = re.sub(r"^\s*(?:은|는|에|부터|까지|~|[-,])\s*", "", cleaned)
                match = re.search(
                    r"([가-힣A-Za-z0-9]+(?:\s+[가-힣A-Za-z0-9]+){0,4}\s+)?"
                    r"(회의|스터디|멘토링|오리엔테이션|진료|예약|세미나|면담|시험|행사|제출|마감)",
                    cleaned,
                )
                if match:
                    title = " ".join(match.group().split())
                    if title == "회의":
                        who = _extract_detail(_WHO_PATTERN, combined)
                        return f"{who} 회의" if who else title
                    return title
            return next(cue for cue in _CALENDAR_CUES if cue in source_text)
    return (items[0].notification.body if app_identity(items[0].notification.app_name) == "kakaotalk"
            else items[0].notification.title)


def _schedule_details(
    items: Sequence[BriefingItem],
    *,
    title: str,
    temporal: _TemporalMatch | None,
) -> ScheduleDetails:
    combined = " ".join(_source_text(item) for item in items)
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
        when=(
            temporal.value.astimezone(_SCHEDULE_TIMEZONE).date().isoformat()
            if temporal and temporal.is_all_day
            else format_timestamp(temporal.value) if temporal else None
        ),
        where=where,
        what=what or title,
        why=why,
        how=how,
    )


class RuleBasedActionItemProvider:
    """Extract schedule details without creating tasks or registering events."""

    def extract(
        self,
        *,
        group_id: str,
        items: Sequence[BriefingItem],
    ) -> ScheduleExtraction:
        ordered = tuple(sorted(items, key=lambda item: item.notification.timestamp))
        if not ordered:
            return ScheduleExtraction()
        combined_text = " ".join(_source_text(item) for item in ordered)
        temporal = _parse_group_temporal(ordered)
        cues = _matched_cues(combined_text, _CALENDAR_CUES)
        # Uploading presentation material or reviewing an event notice is not
        # itself an appointment. Weak action nouns need a parsed date/time.
        strong_cues = set(cues) - {"발표", "행사", "제출"}
        is_schedule = bool(strong_cues or (cues and temporal)) or any(
            item.filter_result.category == "일정/회의" for item in ordered
        )
        if not is_schedule:
            return ScheduleExtraction()

        title = _calendar_title(ordered)
        summary = ScheduleSummary(
            summary_id=_candidate_id("schedule", group_id),
            title=title,
            status=_calendar_status(ordered),
            schedule_details=_schedule_details(ordered, title=title, temporal=temporal),
            source_notification_ids=tuple(item.notification.id for item in ordered),
            source_group_ids=(group_id,),
            is_all_day=temporal.is_all_day if temporal else None,
            end_at=format_timestamp(temporal.end) if temporal and temporal.end else None,
        )
        return ScheduleExtraction(schedules=(summary,))
