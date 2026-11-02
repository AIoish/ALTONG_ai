"""Rule-based group categorization used as an MVP comparison baseline."""

from __future__ import annotations

from collections import defaultdict

from .schema import BriefingItem, CategoryDecision


def canonical_category(item: BriefingItem) -> str:
    """Preserve the official category produced by the filtering pipeline."""

    return item.filter_result.category or "기타"


def categorize_text(items: list[BriefingItem]) -> CategoryDecision:
    """Conservative offline fallback; never uses filtering category/score fields.

    This is not the trained classifier and needs model evaluation before release.
    """
    from .clustering import app_identity

    text = " ".join(
        item.notification.body if app_identity(item.notification.app_name) == "kakaotalk"
        else f"{item.notification.title} {item.notification.body}"
        for item in items
    )
    if any(word in text for word in ("서버", "배치", "장애", "배포", "DB", "API")) and any(
        word in text for word in ("긴급", "오류", "장애", "실패", "중단", "멈", "복구", "정상화")
    ):
        category = "긴급 업무"
    elif any(word in text for word in ("비밀번호", "로그아웃", "보안", "의심 기기", "로그인", "인증", "시스템 점검", "서버 점검", "엘리베이터 점검")):
        category = "시스템/보안"
    elif any(word in text for word in ("쿠폰", "할인", "프로모션", "행사 기간", "광고")):
        category = "광고/홍보"
    elif any(word in text for word in ("병원", "치과", "진료", "예약", "환불", "결제", "배송", "이용권", "구독")):
        category = "개인 중요"
    elif any(word in text for word in ("회의", "일정", "시험", "오리엔테이션", "스터디", "면담", "멘토링", "만나요")):
        category = "일정/회의"
    elif any(word in text for word in ("보고서", "과제", "제출", "업무", "리뷰", "작업", "업로드", "서버", "배포", "검토", "안내문")):
        category = "일반 업무"
    elif items and app_identity(items[0].notification.app_name) == "kakaotalk":
        category = "개인 일반"
    else:
        category = "기타"
    return CategoryDecision(category, tuple(item.notification.id for item in items))


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
