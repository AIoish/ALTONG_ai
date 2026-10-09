"""Provider boundaries and the deterministic MVP briefing provider."""

from __future__ import annotations

from typing import Protocol, Sequence

from .categorization import categorize_group, categorize_text
from .schema import BriefingItem, CategoryDecision, RawNotification


class NotificationSanitizer(Protocol):
    """Future boundary for masking data immediately before an external call.

    The first MVP neither implements nor invokes this boundary because it uses
    synthetic data and never calls an external provider.
    """

    def sanitize(self, notification: RawNotification) -> RawNotification: ...


class BriefingProvider(Protocol):
    """Replaceable group-summary provider boundary."""

    def summarize(self, items: Sequence[BriefingItem]) -> tuple[str, ...]: ...


class CategoryProvider(Protocol):
    """Replaceable group-level category provider boundary."""

    def categorize(self, items: Sequence[BriefingItem]) -> CategoryDecision: ...


class RuleBasedCategoryProvider:
    """Legacy, explicit compatibility with filtering-owned categories."""

    def categorize(self, items: Sequence[BriefingItem]) -> CategoryDecision:
        return categorize_group(list(items))


class TextCategoryProvider:
    """Offline source-text fallback for briefing-owned classification."""

    def categorize(self, items: Sequence[BriefingItem]) -> CategoryDecision:
        return categorize_text(list(items))


class RuleBasedBriefingProvider:
    """Select at most three source-grounded lines without a model call."""

    def summarize(self, items: Sequence[BriefingItem]) -> tuple[str, ...]:
        if not items:
            return ()

        ordered = sorted(items, key=lambda item: item.notification.timestamp)
        # Preserve context and the latest updates without filtering scores.
        selected = (ordered[0], *ordered[-2:])

        lines: list[str] = []
        selected_ids: set[str] = set()
        for item in sorted(selected, key=lambda entry: entry.notification.timestamp):
            notification = item.notification
            if notification.id in selected_ids:
                continue
            selected_ids.add(notification.id)
            line = f"{notification.title} — {notification.body}".strip()
            if len(line) > 160:
                line = f"{line[:157].rstrip()}..."
            if line not in lines:
                lines.append(line)
        return tuple(lines[:3])
