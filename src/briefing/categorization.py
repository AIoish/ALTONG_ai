"""Rule-based group categorization used as an MVP comparison baseline."""

from __future__ import annotations

from collections import defaultdict

from .schema import BriefingItem, CategoryDecision


def canonical_category(item: BriefingItem) -> str:
    """Preserve the official category produced by the filtering pipeline."""

    return item.filter_result.category


def categorize_group(items: list[BriefingItem]) -> CategoryDecision:
    """Choose a primary category after grouping all related notifications.

    Each notification has one vote.  Ties prefer the category of the latest
    notification so a later update can influence the final state.
    """

    if not items:
        raise ValueError("cannot categorize an empty group")

    votes: dict[str, int] = defaultdict(int)
    evidence: dict[str, list[str]] = defaultdict(list)
    ordered = sorted(items, key=lambda item: item.notification.timestamp)

    for item in ordered:
        category = canonical_category(item)
        votes[category] += 1
        evidence[category].append(item.notification.id)

    highest_score = max(votes.values())
    tied = {category for category, score in votes.items() if score == highest_score}
    latest_category = canonical_category(ordered[-1])
    primary = latest_category if latest_category in tied else sorted(tied)[0]

    return CategoryDecision(
        primary_category=primary,
        evidence_notification_ids=tuple(evidence[primary]),
    )
