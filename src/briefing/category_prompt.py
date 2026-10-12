"""Joint classification/summary contract, independent of upstream labels."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .qwen_prompt import (
    MAX_SUMMARY_LINES, SYSTEM_PROMPT, SummaryResponseError,
    build_messages as build_summary_messages, parse_summary_response,
)
from .schema import BriefingDecision, FILTER_CATEGORIES
from .clustering import app_identity


CATEGORY_GUIDE = """카테고리도 입력 원문을 보고 직접 판단하세요. 허용값과 기준:
- 긴급 업무: 서비스 장애·배포 실패·배치 중단 등 즉각 대응이 필요한 업무 사건 (복구·롤백 완료 알림도 같은 사건으로 유지)
- 일반 업무: 업무 요청·진행·완료·제출 및 자료 공유
- 일정/회의: 회의·행사·교육 등의 일정 안내·변경·취소
- 시스템/보안: 계정·인증·보안·시스템 및 시설 점검 안내 (엘리베이터 점검 포함)
- 개인 중요: 개인 진료·예약·결제·환불·배송 등 중요한 개인 알림
- 개인 일반: 일상 대화·개인적인 일반 소식
- 광고/홍보: 할인·쿠폰·상품 홍보
- 기타: 위 항목에 해당하지 않거나 근거가 부족한 알림
날짜가 있다는 이유만으로 일정/회의로 분류하지 마세요. 업무 제출 마감은 일반 업무,
개인 병원 예약은 개인 중요일 수 있고, 이 경우에도 별도로 일정 정보를 추출합니다.
최신 상태와 묶음의 핵심 목적을 보고 대표 카테고리 하나를 선택하세요.
앱이나 방 이름만으로 판단하지 말고 본문을 우선하세요.
오리엔테이션 장소 정정은 일정/회의, 배송·환불·이용권 갱신은 개인 중요입니다.
줄 수는 상한이지 목표가 아닙니다. 짧은 요청·완료 알림은 한 문장으로 충분합니다.
'업무 진행 관련입니다' 같은 분류 설명이나 '마감은 검토하고 올려야 합니다' 같은 빈 문장은 쓰지 마세요.
카카오톡 title은 방 이름입니다. 방 이름을 장소·업로드 위치·참석자로 옮기지 마세요.
원문에 없는 날짜·시간·마감·장소·이유는 덧붙이지 말고, 있는 대상·제출물·위치·행동은 보존하세요.
timestamp는 수신 시각입니다. 원문에 없는 일정 시각을 수신 시각에서 만들어 내지 마세요.
출력은 primary_category와 summary_lines 두 필드만 있는 JSON 객체입니다.
예: {"primary_category":"일정/회의","summary_lines":["회의가 내일 오후 1시에 진행됩니다."]}
"""

MAX_NEW_TOKENS = 256


def build_messages(group: Mapping[str, Any], *, max_summary_lines: int = MAX_SUMMARY_LINES):
    # Reuse source validation/order without exposing the old category label.
    messages = build_summary_messages(
        {**group, "category": "기타"}, max_summary_lines=max_summary_lines,
    )
    user = messages[-1]["content"]
    start = user.index("입력 JSON:\n") + len("입력 JSON:\n")
    end = user.index("\n", start)
    context = json.loads(user[start:end])
    context.pop("category")
    context["app_name"] = app_identity(context["app_name"])
    source_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    # Summary-specific schema directives are replaced, not contradicted.
    rules = SYSTEM_PROMPT.split("출력 스키마:")[0]
    rules = rules.replace(
        "12. 최상위 값은 배열이 아니라 반드시 summary_lines 필드가 있는 객체여야 합니다.",
        "12. 반드시 primary_category와 summary_lines가 있는 객체를 출력하세요.",
    )
    messages[0]["content"] = rules + "\n" + CATEGORY_GUIDE
    messages[-1]["content"] = (
        f"다음 알림 그룹의 카테고리를 분류하고 최대 {max_summary_lines}줄로 요약하세요.\n"
        f"입력 JSON:\n{source_json}\n"
        "오래된 알림부터 정렬되어 있습니다. 최신 상태와 흩어진 핵심 사실을 보존하세요.\n"
        '형식: {"primary_category":"허용 카테고리","summary_lines":["요약 문장"]}'
    )
    return messages


def parse_response(raw: str) -> BriefingDecision:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SummaryResponseError("briefing response must be valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"primary_category", "summary_lines"}:
        raise SummaryResponseError("briefing response requires primary_category and summary_lines only")
    if value["primary_category"] not in FILTER_CATEGORIES:
        raise SummaryResponseError("invalid briefing category")
    lines = parse_summary_response(
        json.dumps({"summary_lines": value["summary_lines"]}, ensure_ascii=False),
        allow_list_repair=False,
    )
    return BriefingDecision(value["primary_category"], lines)
