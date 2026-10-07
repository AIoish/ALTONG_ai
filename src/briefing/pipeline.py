"""Application service for the first rule-based session briefing MVP."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
from typing import Any, Iterable, Mapping

from .action_items import (
    ActionItemProvider,
    RuleBasedActionItemProvider,
)
from .clustering import RuleGroup, group_items, representative_keywords
from .schema import (
    BriefingGroup,
    BriefingItem,
    ContractValidationError,
    FilterResult,
    RawNotification,
    SessionBriefing,
)
from .summarize import (
    BriefingProvider,
    CategoryProvider,
    RuleBasedBriefingProvider,
    RuleBasedCategoryProvider,
)


def _fingerprint(notification: RawNotification) -> tuple[str, ...]:
    return (
        notification.app_name.casefold().strip(),
        notification.sender.casefold().strip(),
        notification.title.casefold().strip(),
        notification.body.casefold().strip(),
        notification.timestamp.isoformat(),
    )


def _group_id(group: RuleGroup) -> str:
    ids = "|".join(sorted(item.notification.id for item in group.items))
    digest = hashlib.sha256(ids.encode("utf-8")).hexdigest()[:12]
    return f"group_{digest}"


class SessionBriefingService:
    """Build structured JSON from blocked synthetic notifications."""

    def __init__(
        self,
        provider: BriefingProvider | None = None,
        category_provider: CategoryProvider | None = None,
        action_item_provider: ActionItemProvider | None = None,
    ) -> None:
        self._provider = provider or RuleBasedBriefingProvider()
        self._category_provider = category_provider or RuleBasedCategoryProvider()
        self._action_item_provider = action_item_provider or RuleBasedActionItemProvider()

    def build(
        self,
        *,
        session_id: str,
        notifications: Iterable[Mapping[str, Any]],
        filter_results: Iterable[Mapping[str, Any]],
        generated_at: datetime | None = None,
    ) -> SessionBriefing:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ContractValidationError("session_id must be a non-empty string")

        parsed_notifications = [
            RawNotification.from_mapping(notification) for notification in notifications
        ]
        parsed_results = [FilterResult.from_mapping(result) for result in filter_results]
        result_by_id = self._index_results(parsed_results)

        blocked: list[BriefingItem] = []
        seen_ids: set[str] = set()
        seen_fingerprints: set[tuple[str, ...]] = set()
        duplicate_count = 0

        for notification in parsed_notifications:
            result = result_by_id.get(notification.id)
            if result is None or result.is_passed:
                continue

            fingerprint = _fingerprint(notification)
            if notification.id in seen_ids or fingerprint in seen_fingerprints:
                duplicate_count += 1
                continue

            seen_ids.add(notification.id)
            seen_fingerprints.add(fingerprint)
            blocked.append(BriefingItem(notification=notification, filter_result=result))

        rule_groups = group_items(blocked)
        groups: list[BriefingGroup] = []
        for rule_group in rule_groups:
            briefing_group = self._build_group(rule_group)
            extraction = self._action_item_provider.extract(
                group_id=briefing_group.group_id,
                items=rule_group.items,
            )
            groups.append(replace(
                briefing_group,
                session_id=session_id.strip(),
                schedule_summaries=extraction.schedules,
            ))
        timestamp = generated_at or datetime.now(timezone.utc)
        return SessionBriefing(
            session_id=session_id.strip(),
            generated_at=timestamp,
            source_notification_count=len(parsed_notifications),
            blocked_notification_count=len(blocked),
            duplicate_count=duplicate_count,
            groups=tuple(groups),
        )

    def build_json(self, **kwargs: Any) -> str:
        briefing = self.build(**kwargs)
        return json.dumps(briefing.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def _index_results(results: list[FilterResult]) -> dict[str, FilterResult]:
        indexed: dict[str, FilterResult] = {}
        for result in results:
            if result.notification_id in indexed:
                raise ContractValidationError(
                    f"duplicate FilterResult for {result.notification_id!r}"
                )
            indexed[result.notification_id] = result
        return indexed

    def _build_group(self, group: RuleGroup) -> BriefingGroup:
        first = group.items[0].notification
        category = self._category_provider.categorize(group.items)
        return BriefingGroup(
            group_id=_group_id(group),
            app_name=first.app_name,
            sender=first.sender,
            time_bucket_start=group.bucket,
            primary_category=category.primary_category,
            category_evidence_notification_ids=category.evidence_notification_ids,
            notification_ids=tuple(item.notification.id for item in group.items),
            keywords=representative_keywords(group),
            summary_lines=self._provider.summarize(group.items),
        )
