from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence
import unittest

from src.briefing.pipeline import SessionBriefingService
from src.briefing.qwen_provider import (
    QwenBriefingProvider,
    TransformersQwenBackend,
    build_group_context,
)
from src.briefing.schema import BriefingItem, FilterResult, RawNotification


def briefing_item(
    notification_id: str,
    *,
    timestamp: str,
    title: str,
    body: str,
    category: str = "일반 업무",
) -> BriefingItem:
    return BriefingItem(
        notification=RawNotification.from_mapping(
            {
                "id": notification_id,
                "app_name": "Slack",
                "sender": "가상 팀장",
                "title": title,
                "body": body,
                "timestamp": timestamp,
            }
        ),
        filter_result=FilterResult.from_mapping(
            {
                "notification_id": notification_id,
                "is_passed": False,
                "category": category,
                "ai_summary_reason": "가상 테스트 판단",
            }
        ),
    )


class FakeBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: Sequence[Mapping[str, str]] | None = None

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.messages = messages
        return self.response


class FailingBackend:
    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise RuntimeError("synthetic model failure")


class QwenRuntimeProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = (
            briefing_item(
                "older",
                timestamp="2026-09-30T09:00:00Z",
                title="로그인 API 오류",
                body="로그인 API에서 500 오류가 발생했습니다.",
                category="긴급 업무",
            ),
            briefing_item(
                "latest",
                timestamp="2026-09-30T09:10:00Z",
                title="로그인 API 정상화",
                body="배포가 완료되어 로그인 API가 정상화되었습니다.",
                category="긴급 업무",
            ),
        )

    def test_group_context_preserves_order_category_without_scores(self) -> None:
        context = build_group_context(tuple(reversed(self.items)))

        self.assertEqual(context["category"], "긴급 업무")
        self.assertNotIn("urgency_score", context)
        self.assertNotIn("relevance_score", context)
        self.assertEqual(
            [item["id"] for item in context["notifications"]],
            ["older", "latest"],
        )

    def test_provider_returns_valid_model_summary_without_exposing_ids(self) -> None:
        backend = FakeBackend(
            '{"summary_lines":["로그인 API가 정상화되었습니다."]}'
        )
        provider = QwenBriefingProvider(backend=backend)

        summary = provider.summarize(self.items)

        self.assertEqual(summary, ("로그인 API가 정상화되었습니다.",))
        self.assertIsNotNone(backend.messages)
        prompt = backend.messages[-1]["content"]
        self.assertIn("로그인 API 정상화", prompt)
        self.assertNotIn("urgency_score", prompt)
        self.assertNotIn("relevance_score", prompt)
        self.assertNotIn('"older"', prompt)
        self.assertNotIn('"latest"', prompt)

    def test_provider_uses_rule_based_fallback_when_model_fails(self) -> None:
        provider = QwenBriefingProvider(backend=FailingBackend())

        with self.assertLogs("src.briefing.qwen_provider", level="ERROR"):
            summary = provider.summarize(self.items)

        self.assertIn(
            "로그인 API 정상화 — 배포가 완료되어 로그인 API가 정상화되었습니다.",
            summary,
        )

        with self.assertNoLogs("src.briefing.qwen_provider", level="ERROR"):
            second_summary = provider.summarize(self.items)
        self.assertEqual(second_summary, summary)

    def test_provider_can_disable_fallback_for_runtime_smoke_tests(self) -> None:
        provider = QwenBriefingProvider(
            backend=FailingBackend(),
            allow_fallback=False,
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
            provider.summarize(self.items)

    def test_pipeline_accepts_qwen_provider_without_contract_changes(self) -> None:
        backend = FakeBackend(
            '{"summary_lines":["로그인 API가 정상화되었습니다."]}'
        )
        service = SessionBriefingService(
            provider=QwenBriefingProvider(backend=backend)
        )

        briefing = service.build(
            session_id="session_qwen",
            notifications=[
                {
                    "id": item.notification.id,
                    "app_name": item.notification.app_name,
                    "sender": item.notification.sender,
                    "title": item.notification.title,
                    "body": item.notification.body,
                    "timestamp": item.notification.timestamp.isoformat(),
                }
                for item in self.items
            ],
            filter_results=[
                {
                    "notification_id": item.filter_result.notification_id,
                    "is_passed": item.filter_result.is_passed,
                    "category": item.filter_result.category,
                    "ai_summary_reason": item.filter_result.ai_summary_reason,
                }
                for item in self.items
            ],
            generated_at=datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc),
        ).to_dict()

        self.assertEqual(
            briefing["groups"][0]["summary_lines"],
            ["로그인 API가 정상화되었습니다."],
        )

    def test_local_backend_defers_missing_adapter_error_until_first_use(self) -> None:
        backend = TransformersQwenBackend(adapter_path="missing-adapter")

        self.assertEqual(backend.adapter_path.name, "missing-adapter")
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            backend._ensure_loaded()


if __name__ == "__main__":
    unittest.main()
