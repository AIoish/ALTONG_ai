"""Shared Qwen prompt and output validation for filtering."""

import json

from pydantic import ValidationError

from src.filtering.schema import CurrentContext, FilterLabel, RawNotification


CATEGORIES = ("긴급 업무", "일반 업무", "일정/회의", "시스템/보안", "개인 중요", "개인 일반", "광고/홍보", "기타")

SYSTEM_PROMPT = """알림과 현재 작업 맥락을 보고 JSON 객체 하나만 출력하세요.
필드는 urgency_score, relevance_score, category, ai_summary_reason 네 개입니다.
urgency_score와 relevance_score는 각각 1부터 5까지의 정수입니다.
urgency_score는 알림 자체의 긴급성, relevance_score는 현재 작업과의 관련성입니다.
relevance_score 4는 현재 목적에 직접 도움이 되는 정보나 요청, 5는 현재 작업의 정확한 대상이나 행동과 연결되는 알림입니다.
같은 분야나 주제라는 이유만으로 4 이상을 부여하지 마세요. 같은 주제여도 현재 작업에 주는 도움이 불분명하면 3 이하입니다.
집중 모드가 켜진 상태를 전제로 판단하세요. 활성 창이 비어 있어도 쉬고 있다고 추정하지 마세요.
duration_seconds는 현재 창에 머문 시간이고 recent_processes는 최근 1~2분간 교차 사용한 앱 이름을 최신순으로 보여줍니다.
앱 이름만으로 작업 주제를 단정하지 마세요. 활성 창이 비어 있으면 현재 작업을 추측하지 마세요.
category는 다음 값 중 하나만 사용하세요: 긴급 업무, 일반 업무, 일정/회의, 시스템/보안, 개인 중요, 개인 일반, 광고/홍보, 기타.
ai_summary_reason은 판단 이유를 한국어 한 문장으로 씁니다.
입력에 없는 상황을 추측하지 마세요. 마크다운이나 설명을 덧붙이지 마세요."""

LABEL_FIELDS = frozenset(FilterLabel.model_fields)


def build_messages(
    notification: RawNotification, context: CurrentContext
) -> list[dict[str, str]]:
    """Use the same message structure for inference and later SFT data."""
    payload = {
        "notification": notification.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_model_output(output: str) -> FilterLabel:
    """Reject malformed or off-contract model responses."""
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError("model output must be a single JSON object") from error

    if not isinstance(value, dict) or set(value) != LABEL_FIELDS:
        raise ValueError("model output must contain exactly the four label fields")
    try:
        label = FilterLabel.model_validate(value, strict=True)
    except ValidationError as error:
        raise ValueError("model output has invalid label values") from error

    if label.category not in CATEGORIES:
        raise ValueError("category is not in the labeling guideline")
    if not label.ai_summary_reason.strip():
        raise ValueError("ai_summary_reason must be non-empty")
    return label
