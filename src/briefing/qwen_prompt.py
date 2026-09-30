"""Shared prompt and response contract for local Qwen briefing summaries."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_SUMMARY_LINES = 3
MAX_LINE_LENGTH = 160

SYSTEM_PROMPT = """당신은 PC 집중 세션이 끝난 뒤 차단된 알림을 요약하는 도우미입니다.

반드시 다음 규칙을 지키세요.
1. 입력 알림에 명시된 사실만 사용하고 추측하거나 새로운 사실을 만들지 마세요.
2. notifications_oldest_to_newest 배열은 오래된 알림부터 정렬되어 있으며 마지막 항목이 가장 최신입니다.
3. 동일한 사건의 반복 알림은 하나의 흐름으로 합치세요.
4. 변경, 취소, 복구, 완료처럼 상태가 달라졌다면 마지막 알림의 상태를 우선하세요.
   이미 취소되거나 바뀐 이전 일정과 상태는 최종 요약에서 제외하세요.
5. 뒤 알림이 앞 알림을 취소하거나 변경한 것이 아니라 새로운 정보를 보탠 경우에는 두 알림의 핵심 사실을 모두 보존하세요.
6. 긴급도와 연관도가 높은 내용을 먼저 쓰세요.
7. 제목만 나열하지 말고 본문에 있는 핵심 사실을 포함하세요.
8. 일정과 제출 알림은 날짜, 시간, 제출물 등 사용자가 행동하는 데 필요한 정보를 보존하세요.
9. 각 줄은 알림 표시나 필드 이름 없이 그 자체로 이해되는 완전한 문장이어야 합니다.
10. 한국어로 짧게 작성하고 사용자 메시지에 지정된 최대 요약 줄 수를 지키세요.
11. 마크다운, 설명, 사고 과정 없이 아래 JSON 객체 하나만 출력하세요.
12. 최상위 값은 배열이 아니라 반드시 summary_lines 필드가 있는 객체여야 합니다.
13. 상태가 이어지는 알림은 제목과 상태를 따로 나열하지 말고 최신 상태가 담긴 완전한 문장으로 합치세요.

출력 스키마:
{"summary_lines":["첫 번째 요약", "두 번째 요약"]}

잘못된 출력:
["첫 번째 요약", "두 번째 요약"]

올바른 출력:
{"summary_lines":["첫 번째 요약", "두 번째 요약"]}
"""


class SummaryResponseError(ValueError):
    """Raised when a model response violates the summary JSON contract."""


def _required_text(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _score_text(value: object) -> str:
    if isinstance(value, Mapping):
        minimum = value.get("min")
        maximum = value.get("max")
        average = value.get("average")
        return f"min={minimum}, max={maximum}, average={average}"
    return str(value)


def _validate_max_summary_lines(max_summary_lines: int) -> int:
    if (
        isinstance(max_summary_lines, bool)
        or not isinstance(max_summary_lines, int)
        or not 1 <= max_summary_lines <= MAX_SUMMARY_LINES
    ):
        raise ValueError("max_summary_lines must be an integer from 1 to 3")
    return max_summary_lines


def _build_user_prompt(
    group_context: Mapping[str, Any], max_summary_lines: int
) -> str:
    return "\n".join(
        (
            "다음 알림 그룹을 요약하세요.",
            f"최대 요약 줄 수: {max_summary_lines}",
            "",
            "입력 JSON:",
            json.dumps(group_context, ensure_ascii=False, separators=(",", ":")),
            "",
            "제목만 복사하지 말고 body의 구체적인 핵심 사실을 포함하세요.",
            "notifications_oldest_to_newest의 마지막 항목이 가장 최신 알림입니다.",
            "최신 알림이 취소, 변경, 복구, 완료를 알리면 이전 상태를 버리고 최종 상태만 쓰세요.",
            "알림끼리 상충하지 않고 서로 다른 정보를 보태면 날짜, 시간, 위치, 제출물, 필요한 행동을 빠뜨리지 마세요.",
            "같은 사건의 제목과 상태를 별도 줄로 나열하지 말고 완전한 문장으로 합치세요.",
            "입력의 id, 필드 이름, '알림 1' 같은 표시는 출력하지 마세요.",
            f"summary_lines 배열에는 최대 {max_summary_lines}개의 문장만 넣으세요.",
            "반드시 { 문자로 시작하고 } 문자로 끝나는 JSON 객체만 출력하세요.",
            '형식: {"summary_lines":["요약 문장"]}',
        )
    )


def build_messages(
    group: Mapping[str, Any], *, max_summary_lines: int = MAX_SUMMARY_LINES
) -> list[dict[str, str]]:
    """Build Qwen chat messages from one already-grouped notification unit."""

    max_summary_lines = _validate_max_summary_lines(max_summary_lines)

    notifications = group.get("notifications")
    if not isinstance(notifications, Sequence) or isinstance(
        notifications, (str, bytes)
    ):
        raise ValueError("notifications must be a non-empty sequence")
    if not notifications:
        raise ValueError("notifications must not be empty")

    notification_records: list[dict[str, str]] = []
    for notification in notifications:
        if not isinstance(notification, Mapping):
            raise ValueError("each notification must be an object")
        _required_text(notification, "id")
        notification_records.append(
            {
                "timestamp": _required_text(notification, "timestamp"),
                "title": _required_text(notification, "title"),
                "body": _required_text(notification, "body"),
            }
        )

    group_context = {
        "app_name": _required_text(group, "app_name"),
        "sender": _required_text(group, "sender"),
        "category": _required_text(group, "category"),
        "urgency_score": _score_text(group.get("urgency_score")),
        "relevance_score": _score_text(group.get("relevance_score")),
        "notifications_oldest_to_newest": notification_records,
    }
    user_prompt = _build_user_prompt(group_context, max_summary_lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_summary_response(raw_response: str) -> tuple[str, ...]:
    """Parse and strictly validate the JSON returned by the model."""

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise SummaryResponseError("model response must be a non-empty string")

    candidate = raw_response.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3:
            raise SummaryResponseError("invalid fenced JSON response")
        candidate = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SummaryResponseError("model response is not valid JSON") from exc

    if not isinstance(payload, dict) or set(payload) != {"summary_lines"}:
        raise SummaryResponseError(
            "model response must contain only the summary_lines field"
        )

    summary_lines = payload["summary_lines"]
    if not isinstance(summary_lines, list):
        raise SummaryResponseError("summary_lines must be an array")
    if not 1 <= len(summary_lines) <= MAX_SUMMARY_LINES:
        raise SummaryResponseError("summary_lines must contain one to three lines")

    normalized: list[str] = []
    for line in summary_lines:
        if not isinstance(line, str) or not line.strip():
            raise SummaryResponseError("each summary line must be a non-empty string")
        text = line.strip()
        if len(text) > MAX_LINE_LENGTH:
            raise SummaryResponseError(
                f"each summary line must be at most {MAX_LINE_LENGTH} characters"
            )
        normalized.append(text)
    return tuple(normalized)
