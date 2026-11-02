"""Deterministic baseline grouping for blocked notifications.

This module is the rule-based comparison baseline.  Local embeddings and
unsupervised clustering can be added later without changing the pipeline API.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
import unicodedata

from .schema import BriefingItem


_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣_]+")
_SEQUENCE_ID_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_MAX_GROUP_SPAN = timedelta(minutes=30)
_MAX_SEQUENCE_GAP = timedelta(minutes=5)
_CONVERSATION_APP_KEYS = frozenset(
    {
        "kakaotalk",
        "slack",
        "teams",
        "microsoft teams",
        "discord",
        "telegram",
        "whatsapp",
        "messages",
        "messenger",
    }
)
_STOP_WORDS = {
    "관련",
    "대한",
    "부탁",
    "확인",
    "주세요",
    "합니다",
    "해주세요",
    "지금",
    "오늘",
    "내일",
    "모레",
    "오전",
    "오후",
    "예정",
    "일정",
    "안내",
    "알림",
    "the",
    "and",
    "for",
    "with",
}


def normalize_identity(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def app_identity(value: str) -> str:
    key = normalize_identity(value)
    return key[:-4] if key.endswith(".exe") else key


def kakao_room_key(item: BriefingItem) -> str | None:
    """Temporary client agreement: Kakao title is a room name, not an ID."""
    if app_identity(item.notification.app_name) != "kakaotalk":
        return None
    return normalize_identity(item.notification.title)


def text_tokens(item: BriefingItem) -> set[str]:
    text = unicodedata.normalize(
        "NFKC", item.notification.body if kakao_room_key(item) is not None
        else f"{item.notification.title} {item.notification.body}"
    ).casefold()
    return {
        token
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 1 and token not in _STOP_WORDS
    }


def hour_bucket(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def sequence_id(value: str) -> tuple[str, int] | None:
    """Return a comparable trailing numeric sequence, excluding UUIDs."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or _UUID_PATTERN.fullmatch(normalized):
        return None
    match = _SEQUENCE_ID_PATTERN.fullmatch(normalized)
    if match is None:
        return None
    return normalize_identity(match.group("prefix")), int(match.group("number"))


def has_adjacent_sequence_id(previous: BriefingItem, current: BriefingItem) -> bool:
    previous_sequence = sequence_id(previous.notification.id)
    current_sequence = sequence_id(current.notification.id)
    if previous_sequence is None or current_sequence is None:
        return False
    previous_prefix, previous_number = previous_sequence
    current_prefix, current_number = current_sequence
    return (
        previous_prefix == current_prefix
        and current_number == previous_number + 1
    )


@dataclass(slots=True)
class RuleGroup:
    app_key: str
    sender_key: str
    bucket: datetime
    items: list[BriefingItem] = field(default_factory=list)
    tokens: set[str] = field(default_factory=set)
    room_key: str | None = None

    def accepts(self, item: BriefingItem, tokens: set[str]) -> bool:
        notification = item.notification
        if app_identity(notification.app_name) != self.app_key:
            return False
        if normalize_identity(notification.sender) != self.sender_key:
            return False
        if kakao_room_key(item) != self.room_key:
            return False
        if not self.items:
            return False

        first_timestamp = self.items[0].notification.timestamp
        if notification.timestamp - first_timestamp > _MAX_GROUP_SPAN:
            return False
        if self.tokens & tokens:
            return True

        # A missing sender may be represented by the app name.  Do not use ID
        # proximity alone in that fallback form because unrelated conversations
        # from the same app could otherwise be merged.
        if self.sender_key == self.app_key:
            return False
        if self.app_key not in _CONVERSATION_APP_KEYS:
            return False

        previous = self.items[-1]
        time_gap = notification.timestamp - previous.notification.timestamp
        # Client IDs may be UUIDs/non-sequential. Known room + sender + close
        # timestamps can connect fragmented Kakao messages without ID proximity.
        known_room = self.room_key not in {None, "카카오톡", "kakaotalk"}
        return (
            timedelta(0) <= time_gap <= _MAX_SEQUENCE_GAP
            and (known_room or has_adjacent_sequence_id(previous, item))
        )

    def add(self, item: BriefingItem, tokens: set[str]) -> None:
        self.items.append(item)
        self.tokens.update(tokens)


def group_items(items: list[BriefingItem]) -> list[RuleGroup]:
    """Group related notifications using identity, text, time, and sequence IDs.

    Shared text tokens can connect notifications within a 30-minute span.
    Short message fragments without shared tokens may also connect when their
    known sequence IDs are adjacent and their timestamps are at most five
    minutes apart.
    """

    groups: list[RuleGroup] = []
    ordered = sorted(
        items,
        key=lambda item: (item.notification.timestamp, item.notification.id),
    )

    for item in ordered:
        tokens = text_tokens(item)
        matching = next((group for group in groups if group.accepts(item, tokens)), None)
        if matching is None:
            notification = item.notification
            matching = RuleGroup(
                app_key=app_identity(notification.app_name),
                sender_key=normalize_identity(notification.sender),
                bucket=hour_bucket(notification.timestamp),
                room_key=kakao_room_key(item),
            )
            groups.append(matching)
        matching.add(item, tokens)

    return groups


def representative_keywords(group: RuleGroup, limit: int = 3) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for item in group.items:
        counts.update(text_tokens(item))
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return tuple(token for token, _ in ranked[:limit])
