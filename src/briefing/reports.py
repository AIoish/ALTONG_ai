"""Category-organized projections of the existing dashboard cards."""

from __future__ import annotations

from .schema import FILTER_CATEGORIES, SessionBriefing


def category_report(briefing: SessionBriefing) -> dict:
    cards = briefing.to_dict()["groups"]
    return {
        "session_id": briefing.session_id,
        "categories": [
            {"category": category, "groups": [
                card for card in cards if card["primary_category"] == category
            ]}
            for category in FILTER_CATEGORIES
            if any(card["primary_category"] == category for card in cards)
        ],
    }


def render_category_markdown(briefing: SessionBriefing) -> str:
    report = category_report(briefing)
    room_names = {group.group_id: group.room_name for group in briefing.groups}
    lines = ["# 집중 세션 카테고리별 요약 보고서", "", f"세션: {briefing.session_id}", ""]
    if not report["categories"]:
        lines.append("요약할 누적 알림이 없습니다.")
    for section in report["categories"]:
        lines.extend([f"## {section['category']}", ""])
        for card in section["groups"]:
            room = room_names[card["group_id"]]
            heading = f"### {card['app_name']} · {card['sender']}"
            if room is not None:
                heading += f" · {room}"
            lines.extend([heading, "", f"그룹: {card['group_id']}", ""])
            lines.extend(f"- {line}" for line in card["summary_lines"])
            for schedule in card["schedule_summaries"]:
                lines.extend(["", f"일정 상태: {schedule['schedule_status']}", ""])
                for field, label in (("who", "누가"), ("when", "시작"), ("end_at", "종료"),
                                     ("where", "어디서"), ("what", "무엇을"), ("why", "왜"), ("how", "어떻게")):
                    lines.append(f"- {label}: {schedule[field] if schedule[field] is not None else '미확인'}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
