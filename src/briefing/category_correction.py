"""Source-grounded postprocessing; never changes model weights or summaries.

Each rule needs multiple signals. Conflicting categories abstain rather than
assigning a priority based on a single keyword. The manifest can be ported to
the client; GGUF conversion cannot embed this Python application code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .clustering import app_identity
from .schema import FILTER_CATEGORIES


RULE_VERSION = "category-correction-v1"
# Every pattern in `all` must match the combined original source text.
RULES = (
    {"id": "operational_incident", "category": "긴급 업무", "all": [
        r"서버|서비스|데이터베이스|\b(?:db|api)\b|배치|배포",
        r"오류|장애|실패|중단|멈|복구|정상화|롤백|재시작",
    ], "none": [r"(?:오류|장애|실패|중단).{0,10}(?:없|아니|않|안\s*났|나면|발생하면)"]},
    {"id": "credential_action", "category": "시스템/보안", "all": [
        r"비밀번호|패스워드", r"재설정|변경|초기화|로그아웃",
    ]},
    {"id": "login_security_action", "category": "시스템/보안", "all": [
        r"로그인|인증",
        r"감지|본인.{0,8}아니|제\s*로그인.{0,8}아니|기기.{0,8}차단|세션.{0,8}종료|로그아웃",
    ]},
    {"id": "maintenance", "category": "시스템/보안", "all": [
        r"엘리베이터|시설|시스템|서버", r"점검",
    ]},
    {"id": "delivery_status", "category": "개인 중요", "all": [
        r"배송", r"완료|끝났|도착|택배|경비실|예정|발송|뒀",
    ]},
    {"id": "personal_reservation", "category": "개인 중요", "all": [
        r"예약", r"치과|병원|진료|안과|치료|재활|상담|숙소|항공|여행",
    ]},
    {"id": "payment_refund_status", "category": "개인 중요", "all": [
        r"환불|결제", r"완료|반품|카드|되었습니다|됐|금액|\d[\d,]*\s*원",
    ]},
    {"id": "promotion_terms", "category": "광고/홍보", "all": [
        r"할인|쿠폰|프로모션", r"만료|소멸|기간|행사|연장|\d+\s*%|구매|적용|이벤트",
    ]},
    {"id": "event_schedule", "category": "일정/회의", "all": [
        r"회의(?!\s*(?:자료|록))|멘토링|오리엔테이션|스터디|면담|행사(?!\s*(?:안내문|포스터))|중간\s*점검",
        r"예정|진행|취소|변경|정정|바뀌|만나|내일|\d+\s*월|오후|오전|장소|참가",
    ]},
    {"id": "document_action", "category": "일반 업무", "all": [
        r"안내문|자료|보고서|문서|초안|문안|포스터|응답\s*표|과제|코드\s*리뷰",
        r"검토|교정|수정|작성|공유|업로드|올려|올리|제출|추가|완료",
    ]},
)


@dataclass(frozen=True)
class CategoryCorrection:
    model_category: str
    category: str
    matched_rule_ids: tuple[str, ...] = ()
    conflicting: bool = False

    @property
    def changed(self) -> bool:
        return self.category != self.model_category


def correct_category(group: Mapping[str, Any], model_category: str) -> CategoryCorrection:
    """Use only original notifications, never summaries/labels/IDs/scores."""
    if model_category not in FILTER_CATEGORIES:
        raise ValueError("invalid model category")
    is_chat = app_identity(str(group.get("app_name", ""))) == "kakaotalk"
    text = " ".join(
        str(notification.get("body", "")) if is_chat
        else f"{notification.get('title', '')} {notification.get('body', '')}"
        for notification in group.get("notifications", [])
    )
    text = re.sub(r"\s+", " ", text)
    matches = [rule for rule in RULES
               if all(re.search(pattern, text, re.IGNORECASE) for pattern in rule["all"])
               and not any(re.search(pattern, text, re.IGNORECASE) for pattern in rule.get("none", []))]
    categories = {rule["category"] for rule in matches}
    ids = tuple(rule["id"] for rule in matches)
    if len(categories) != 1:
        return CategoryCorrection(model_category, model_category, ids, len(categories) > 1)
    return CategoryCorrection(model_category, categories.pop(), ids)


def correction_manifest() -> dict[str, Any]:
    """Portable specification for clients, not executable GGUF metadata."""
    return {"version": RULE_VERSION, "rules": list(RULES),
            "source": "KakaoTalk: body only; other apps: title + body; join notifications with spaces",
            "matching": "case-insensitive regex; every all pattern must match; no none pattern may match",
            "conflicts": "keep the model category if zero or multiple different categories match",
            "summary": "do not modify summary_lines"}
