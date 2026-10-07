"""Create deterministic synthetic data for briefing-summary fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from src.briefing.schema import FILTER_CATEGORIES

from .prompts import MAX_SUMMARY_LINES, build_messages, parse_summary_response


DATA_DIRECTORY = Path(__file__).with_name("data")
DEFAULT_TRAIN_PATH = DATA_DIRECTORY / "train_cases.jsonl"
DEFAULT_VALIDATION_PATH = DATA_DIRECTORY / "validation_cases.jsonl"


@dataclass(frozen=True)
class NotificationTemplate:
    title: str
    body: str


@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    variants: tuple[Mapping[str, str], ...]
    notifications: tuple[NotificationTemplate, ...]
    target_lines: tuple[str, ...]
    max_summary_lines: int


BASE_SCENARIOS = (
    Scenario(
        name="service_recovery",
        category="긴급 업무",
        variants=(
            {"app": "Slack", "sender": "가상 운영팀", "subject": "로그인 API"},
            {"app": "Teams", "sender": "가상 인프라팀", "subject": "결제 API"},
            {"app": "Slack", "sender": "가상 백엔드팀", "subject": "검색 서비스"},
            {"app": "Teams", "sender": "가상 SRE팀", "subject": "파일 업로드 서비스"},
            {"app": "Slack", "sender": "가상 플랫폼팀", "subject": "알림 서비스"},
        ),
        notifications=(
            NotificationTemplate("{subject} 장애", "{subject}에서 오류가 발생했습니다."),
            NotificationTemplate("{subject} 복구", "조치가 완료되어 {subject}가 정상화되었습니다."),
        ),
        target_lines=("{subject} 장애가 복구되어 정상화되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="task_completed",
        category="일반 업무",
        variants=(
            {"app": "Slack", "sender": "가상 팀원", "subject": "주간 발표 자료", "place": "공유 폴더"},
            {"app": "Teams", "sender": "가상 동료", "subject": "회의록", "place": "팀 드라이브"},
            {"app": "Slack", "sender": "가상 개발자", "subject": "API 문서", "place": "문서 저장소"},
            {"app": "Teams", "sender": "가상 디자이너", "subject": "화면 시안", "place": "프로젝트 보드"},
            {"app": "Slack", "sender": "가상 기획자", "subject": "요구사항 명세서", "place": "공유 문서함"},
        ),
        notifications=(
            NotificationTemplate("{subject} 작업 시작", "{subject} 작성을 시작했습니다."),
            NotificationTemplate("{subject} 완료", "{subject}를 {place}에 업로드했습니다."),
        ),
        target_lines=("{subject}를 {place}에 업로드했습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="meeting_cancelled",
        category="일정/회의",
        variants=(
            {"app": "Calendar", "sender": "가상 프로젝트 리더", "subject": "주간 회의"},
            {"app": "Teams", "sender": "가상 스터디장", "subject": "AI 스터디"},
            {"app": "Calendar", "sender": "가상 조교", "subject": "과제 설명회"},
            {"app": "Slack", "sender": "가상 팀장", "subject": "배포 점검 회의"},
            {"app": "Calendar", "sender": "가상 운영자", "subject": "서비스 회고"},
        ),
        notifications=(
            NotificationTemplate("{subject} 시간 변경", "{date} {subject}를 {time}로 변경합니다."),
            NotificationTemplate("{subject} 취소", "{date} {time} {subject}는 취소되었습니다."),
        ),
        target_lines=("{date} {time} {subject}가 취소되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="security_login",
        category="시스템/보안",
        variants=(
            {"app": "Security Center", "sender": "가상 보안 시스템", "subject": "부산"},
            {"app": "Account", "sender": "가상 계정 보호팀", "subject": "대전"},
            {"app": "Security Center", "sender": "가상 보안 봇", "subject": "제주"},
            {"app": "Account", "sender": "가상 인증 시스템", "subject": "광주"},
            {"app": "Security Center", "sender": "가상 보안 센터", "subject": "인천"},
        ),
        notifications=(
            NotificationTemplate("새로운 위치에서 로그인", "{subject}에서 새로운 로그인이 감지되었습니다."),
            NotificationTemplate("계정 보호 안내", "본인이 아니라면 즉시 비밀번호를 변경해 주세요."),
        ),
        target_lines=(
            "{subject}에서 새로운 로그인이 감지되었습니다.",
            "본인이 아니라면 즉시 비밀번호를 변경해야 합니다.",
        ),
        max_summary_lines=2,
    ),
    Scenario(
        name="appointment_cancelled",
        category="개인 중요",
        variants=(
            {"app": "Calendar", "sender": "가상 치과", "subject": "치과 진료"},
            {"app": "Calendar", "sender": "가상 병원", "subject": "건강 검진"},
            {"app": "Booking", "sender": "가상 상담센터", "subject": "상담"},
            {"app": "Calendar", "sender": "가상 안과", "subject": "안과 진료"},
            {"app": "Booking", "sender": "가상 검진센터", "subject": "예방 접종"},
        ),
        notifications=(
            NotificationTemplate("{subject} 예약 안내", "{date} {time}에 {subject} 예약이 있습니다."),
            NotificationTemplate("{subject} 예약 취소", "기관 사정으로 {date} {time} {subject} 예약이 취소되었습니다."),
        ),
        target_lines=("{date} {time} {subject} 예약이 취소되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="delivery_completed",
        category="개인 일반",
        variants=(
            {"app": "Delivery", "sender": "가상 택배사", "subject": "생활용품", "place": "현관 앞"},
            {"app": "Shopping", "sender": "가상 쇼핑몰", "subject": "도서", "place": "무인 보관함"},
            {"app": "Delivery", "sender": "가상 배송팀", "subject": "전자기기", "place": "경비실"},
            {"app": "Shopping", "sender": "가상 판매자", "subject": "의류", "place": "택배 보관실"},
            {"app": "Delivery", "sender": "가상 물류센터", "subject": "문구류", "place": "현관 앞"},
        ),
        notifications=(
            NotificationTemplate("{subject} 배송 시작", "{subject} 배송이 시작되었습니다."),
            NotificationTemplate("{subject} 배송 완료", "배송 물품 {subject}이 {place}에 도착했습니다."),
        ),
        target_lines=("배송 물품 {subject}이 {place}에 도착했습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="promotion_extended",
        category="광고/홍보",
        variants=(
            {"app": "Shopping", "sender": "가상 쇼핑몰", "subject": "신학기 할인"},
            {"app": "Store", "sender": "가상 브랜드", "subject": "회원 할인"},
            {"app": "Shopping", "sender": "가상 마켓", "subject": "도서 할인"},
            {"app": "Store", "sender": "가상 스토어", "subject": "무료 배송 행사"},
            {"app": "Shopping", "sender": "가상 판매처", "subject": "쿠폰 행사"},
        ),
        notifications=(
            NotificationTemplate("{subject} 종료 안내", "{subject}은 {old_date}까지 진행됩니다."),
            NotificationTemplate("{subject} 연장", "{subject}이 {new_date}까지 연장되었습니다."),
        ),
        target_lines=("{subject}이 {new_date}까지 연장되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="notice_corrected",
        category="기타",
        variants=(
            {"app": "Notice", "sender": "가상 행정실", "subject": "오리엔테이션", "old_place": "A강의실", "place": "B강의실"},
            {"app": "Community", "sender": "가상 운영자", "subject": "동아리 모임", "old_place": "학생회관", "place": "도서관 세미나실"},
            {"app": "Notice", "sender": "가상 안내센터", "subject": "장비 교육", "old_place": "실습실 1", "place": "실습실 2"},
            {"app": "Community", "sender": "가상 관리자", "subject": "시설 점검 안내", "old_place": "본관", "place": "별관"},
            {"app": "Notice", "sender": "가상 지원팀", "subject": "신입 안내", "old_place": "회의실 1", "place": "회의실 3"},
        ),
        notifications=(
            NotificationTemplate("{subject} 장소 안내", "{subject} 장소는 {old_place}입니다."),
            NotificationTemplate("{subject} 장소 정정", "{subject} 장소를 {place}로 정정합니다."),
        ),
        target_lines=("{subject} 장소가 {place}로 정정되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="deadline_extended",
        category="일정/회의",
        variants=(
            {"app": "LMS", "sender": "가상 조교", "subject": "AI 과제"},
            {"app": "Teams", "sender": "가상 팀장", "subject": "주간 보고서"},
            {"app": "LMS", "sender": "가상 교수", "subject": "데이터베이스 과제"},
            {"app": "Slack", "sender": "가상 PM", "subject": "기능 명세서"},
            {"app": "LMS", "sender": "가상 강사", "subject": "모델 평가 보고서"},
        ),
        notifications=(
            NotificationTemplate("{subject} 제출 기한", "{subject} 제출 기한은 {old_date} {time}입니다."),
            NotificationTemplate("{subject} 제출 기한 연장", "{subject} 제출 기한이 {new_date} {time}로 연장되었습니다."),
        ),
        target_lines=("{subject} 제출 기한이 {new_date} {time}로 연장되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="submission_details",
        category="일반 업무",
        variants=(
            {"app": "LMS", "sender": "가상 조교", "subject": "AI 서비스 과제", "artifact": "보고서 PDF와 소스 코드 링크"},
            {"app": "Teams", "sender": "가상 팀장", "subject": "주간 업무 보고", "artifact": "보고서와 회의록"},
            {"app": "LMS", "sender": "가상 교수", "subject": "운영체제 과제", "artifact": "PDF와 실행 결과 화면"},
            {"app": "Slack", "sender": "가상 리뷰어", "subject": "코드 리뷰", "artifact": "PR 링크와 테스트 결과"},
            {"app": "LMS", "sender": "가상 강사", "subject": "최종 프로젝트", "artifact": "발표 자료와 저장소 링크"},
        ),
        notifications=(
            NotificationTemplate("{subject} 제출 안내", "{subject} 제출 마감은 {date} {time}입니다."),
            NotificationTemplate("{subject} 제출 형식", "{artifact}를 함께 제출해야 합니다."),
        ),
        target_lines=(
            "{subject} 제출 마감은 {date} {time}입니다.",
            "{artifact}를 함께 제출해야 합니다.",
        ),
        max_summary_lines=2,
    ),
    Scenario(
        name="fragmented_schedule_chat",
        category="일정/회의",
        variants=(
            {
                "app": "KakaoTalk",
                "sender": "가상 스터디장",
                "subject": "AI 스터디",
                "who": "프로젝트 팀원들",
                "place": "B강의실",
                "reason": "발표 순서 조정",
                "method": "대면",
            },
            {
                "app": "KakaoTalk",
                "sender": "가상 팀장",
                "subject": "기획 회의",
                "who": "기획팀과 개발팀",
                "place": "회의실 2",
                "reason": "요구사항 확정",
                "method": "대면",
            },
            {
                "app": "Slack",
                "sender": "가상 프로젝트 리더",
                "subject": "진행 상황 공유회",
                "who": "프로젝트 참여자들",
                "place": "온라인 회의실",
                "reason": "중간 결과 공유",
                "method": "화상",
            },
            {
                "app": "KakaoTalk",
                "sender": "가상 조교",
                "subject": "과제 질의응답",
                "who": "수강생들",
                "place": "공학관 301호",
                "reason": "제출 전 질문 정리",
                "method": "대면",
            },
            {
                "app": "Teams",
                "sender": "가상 운영자",
                "subject": "서비스 회고",
                "who": "운영팀 전원",
                "place": "온라인 회의실",
                "reason": "장애 대응 과정 점검",
                "method": "화상",
            },
        ),
        notifications=(
            NotificationTemplate(
                "{sender}", "참석자는 {who}이고 {subject}를 진행합니다."
            ),
            NotificationTemplate("{sender}", "{date} {time}에 만나요."),
            NotificationTemplate("{sender}", "장소는 {place}입니다."),
            NotificationTemplate(
                "{sender}", "진행 이유는 {reason}이고 {method} 방식입니다."
            ),
        ),
        target_lines=(
            "{subject} 일정은 {date} {time} {place}이며 참석자는 {who}입니다.",
            "진행 이유는 {reason}이고 {method} 방식입니다.",
        ),
        max_summary_lines=2,
    ),
    Scenario(
        name="fragmented_task_chat",
        category="일반 업무",
        variants=(
            {
                "app": "KakaoTalk",
                "sender": "가상 팀장",
                "subject": "발표 자료 수정",
                "artifact": "수정본과 검토 의견",
                "place": "팀 공유 폴더",
            },
            {
                "app": "Slack",
                "sender": "가상 리뷰어",
                "subject": "결제 모듈 코드 보완",
                "artifact": "PR 링크와 테스트 결과",
                "place": "개발 채널",
            },
            {
                "app": "KakaoTalk",
                "sender": "가상 동료",
                "subject": "회의록 정리",
                "artifact": "회의록과 결정 사항",
                "place": "팀 드라이브",
            },
            {
                "app": "Teams",
                "sender": "가상 기획자",
                "subject": "요구사항 문서 갱신",
                "artifact": "변경 내역과 최신 문서",
                "place": "프로젝트 보드",
            },
            {
                "app": "KakaoTalk",
                "sender": "가상 멘토",
                "subject": "모델 평가 결과 정리",
                "artifact": "평가 표와 실패 사례",
                "place": "공유 문서함",
            },
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 부탁드려요."),
            NotificationTemplate("{sender}", "{date} {time}까지예요."),
            NotificationTemplate("{sender}", "{artifact}도 같이 올려 주세요."),
            NotificationTemplate("{sender}", "{place}에 올리면 됩니다."),
        ),
        target_lines=(
            "{subject} 작업 마감은 {date} {time}입니다.",
            "필요 자료는 {artifact}이며 업로드 위치는 {place}입니다.",
        ),
        max_summary_lines=2,
    ),
)


ADDITIONAL_SCENARIOS = (
    Scenario(
        name="fragmented_deployment_rollback",
        category="긴급 업무",
        variants=(
            {"app": "Slack", "sender": "가상 배포팀", "subject": "결제 서비스"},
            {"app": "Teams", "sender": "가상 플랫폼팀", "subject": "검색 서비스"},
            {"app": "Slack", "sender": "가상 백엔드팀", "subject": "회원 API"},
            {"app": "Teams", "sender": "가상 운영팀", "subject": "파일 서비스"},
            {"app": "Slack", "sender": "가상 릴리스팀", "subject": "주문 서비스"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 신규 버전 배포에 실패했어요."),
            NotificationTemplate("{sender}", "추가 배포는 우선 중단했습니다."),
            NotificationTemplate("{sender}", "이전 버전으로 롤백하고 있어요."),
            NotificationTemplate("{sender}", "안정화 확인 후 다시 공유하겠습니다."),
        ),
        target_lines=("{subject} 배포가 실패해 이전 버전으로 롤백하고 있습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="database_failover",
        category="긴급 업무",
        variants=(
            {"app": "PagerDuty", "sender": "가상 DBA팀", "subject": "주문 데이터베이스"},
            {"app": "Slack", "sender": "가상 인프라팀", "subject": "회원 데이터베이스"},
            {"app": "Teams", "sender": "가상 데이터팀", "subject": "분석 데이터베이스"},
            {"app": "PagerDuty", "sender": "가상 운영팀", "subject": "정산 데이터베이스"},
            {"app": "Slack", "sender": "가상 SRE팀", "subject": "재고 데이터베이스"},
        ),
        notifications=(
            NotificationTemplate("{subject} 연결 오류", "{subject} 연결 오류가 반복되고 있습니다."),
            NotificationTemplate("{subject} 장애 조치", "대기 장비로 전환해 {subject} 연결이 복구되었습니다."),
        ),
        target_lines=("{subject}를 대기 장비로 전환해 연결을 복구했습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_suspicious_device_blocked",
        category="시스템/보안",
        variants=(
            {"app": "Security Center", "sender": "가상 보안팀", "subject": "서울", "device": "Windows PC"},
            {"app": "Account", "sender": "가상 계정보호팀", "subject": "대구", "device": "Android 기기"},
            {"app": "Security Center", "sender": "가상 인증팀", "subject": "울산", "device": "iPhone"},
            {"app": "Account", "sender": "가상 보안 봇", "subject": "수원", "device": "MacBook"},
            {"app": "Security Center", "sender": "가상 보안센터", "subject": "세종", "device": "Linux PC"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject}의 {device}에서 의심스러운 로그인이 감지됐어요."),
            NotificationTemplate("{sender}", "본인 활동이 아닌 것으로 확인됐습니다."),
            NotificationTemplate("{sender}", "해당 {device}는 차단했어요."),
            NotificationTemplate("{sender}", "모든 로그인 세션도 종료했습니다."),
        ),
        target_lines=(
            "{subject}의 {device}에서 의심스러운 로그인이 감지되었습니다.",
            "해당 기기를 차단하고 모든 로그인 세션을 종료했습니다.",
        ),
        max_summary_lines=2,
    ),
    Scenario(
        name="password_reset_completed",
        category="시스템/보안",
        variants=(
            {"app": "Account", "sender": "가상 계정팀", "subject": "학교 포털"},
            {"app": "Security Center", "sender": "가상 인증팀", "subject": "업무 계정"},
            {"app": "Account", "sender": "가상 고객센터", "subject": "쇼핑 계정"},
            {"app": "Security Center", "sender": "가상 보안팀", "subject": "개발자 계정"},
            {"app": "Account", "sender": "가상 서비스팀", "subject": "커뮤니티 계정"},
        ),
        notifications=(
            NotificationTemplate("비밀번호 재설정 요청", "{subject} 비밀번호 재설정 요청이 접수되었습니다."),
            NotificationTemplate("비밀번호 변경 완료", "{subject} 비밀번호 변경이 완료되어 기존 세션이 종료되었습니다."),
        ),
        target_lines=("{subject} 비밀번호 변경이 완료되어 기존 세션이 종료되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="payment_due",
        category="개인 중요",
        variants=(
            {"app": "Finance", "sender": "가상 카드사", "subject": "카드 대금"},
            {"app": "Banking", "sender": "가상 은행", "subject": "대출 이자"},
            {"app": "Utility", "sender": "가상 전력사", "subject": "전기 요금"},
            {"app": "Finance", "sender": "가상 보험사", "subject": "보험료"},
            {"app": "Utility", "sender": "가상 통신사", "subject": "통신 요금"},
        ),
        notifications=(
            NotificationTemplate("{subject} 납부 안내", "{subject} 납부 기한은 {date}입니다."),
            NotificationTemplate("{subject} 납부 예정", "{date} {time}에 등록 계좌에서 {subject} 자동 납부가 진행됩니다."),
        ),
        target_lines=("{subject} 자동 납부는 {date} {time}에 등록 계좌에서 진행될 예정입니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_travel_schedule_changed",
        category="개인 중요",
        variants=(
            {"app": "Travel", "sender": "가상 항공사", "subject": "제주행 항공편", "place": "3번 탑승구"},
            {"app": "Rail", "sender": "가상 철도사", "subject": "부산행 열차", "place": "5번 승강장"},
            {"app": "Travel", "sender": "가상 버스사", "subject": "대전행 버스", "place": "12번 승차장"},
            {"app": "Travel", "sender": "가상 여행사", "subject": "공항 셔틀", "place": "호텔 정문"},
            {"app": "Rail", "sender": "가상 교통센터", "subject": "광주행 열차", "place": "2번 승강장"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 출발 일정이 변경됐어요."),
            NotificationTemplate("{sender}", "새 출발 시각은 {date} {time}입니다."),
            NotificationTemplate("{sender}", "탑승 장소는 {place}예요."),
            NotificationTemplate("{sender}", "변경된 일정으로 이용해 주세요."),
        ),
        target_lines=("{subject} 출발 일정은 {date} {time}이며 탑승 장소는 {place}입니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="refund_completed",
        category="개인 일반",
        variants=(
            {"app": "Shopping", "sender": "가상 쇼핑몰", "subject": "운동화"},
            {"app": "Store", "sender": "가상 서점", "subject": "도서"},
            {"app": "Shopping", "sender": "가상 마켓", "subject": "생활용품"},
            {"app": "Store", "sender": "가상 전자상가", "subject": "충전기"},
            {"app": "Shopping", "sender": "가상 의류몰", "subject": "재킷"},
        ),
        notifications=(
            NotificationTemplate("{subject} 환불 접수", "{subject} 환불 요청이 접수되었습니다."),
            NotificationTemplate("{subject} 환불 완료", "{subject} 결제 취소와 환불 처리가 완료되었습니다."),
        ),
        target_lines=("{subject} 결제 취소와 환불 처리가 완료되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_subscription_renewal",
        category="개인 일반",
        variants=(
            {"app": "Subscription", "sender": "가상 음악 서비스", "subject": "음악 이용권"},
            {"app": "Subscription", "sender": "가상 영상 서비스", "subject": "영상 이용권"},
            {"app": "Cloud", "sender": "가상 클라우드", "subject": "저장 공간 요금제"},
            {"app": "Learning", "sender": "가상 학습 서비스", "subject": "온라인 강의 이용권"},
            {"app": "Subscription", "sender": "가상 뉴스 서비스", "subject": "뉴스 구독권"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 갱신일이 다가오고 있어요."),
            NotificationTemplate("{sender}", "{date}에 자동 갱신될 예정입니다."),
            NotificationTemplate("{sender}", "등록된 결제 수단으로 결제돼요."),
            NotificationTemplate("{sender}", "갱신 전에 결제 수단을 확인해 주세요."),
        ),
        target_lines=("{subject} 자동 갱신은 {date}이며 결제 수단을 확인해야 합니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_coupon_expiring",
        category="광고/홍보",
        variants=(
            {"app": "Shopping", "sender": "가상 쇼핑몰", "subject": "신규 회원 쿠폰"},
            {"app": "Store", "sender": "가상 카페", "subject": "음료 할인 쿠폰"},
            {"app": "Shopping", "sender": "가상 마켓", "subject": "무료 배송 쿠폰"},
            {"app": "Store", "sender": "가상 서점", "subject": "도서 할인 쿠폰"},
            {"app": "Shopping", "sender": "가상 브랜드몰", "subject": "생일 쿠폰"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject}이 계정에 발급됐어요."),
            NotificationTemplate("{sender}", "아직 사용하지 않은 상태입니다."),
            NotificationTemplate("{sender}", "사용 기한은 {date}까지예요."),
            NotificationTemplate("{sender}", "기한이 지나면 자동으로 만료됩니다."),
        ),
        target_lines=("{subject}은 {date}에 만료됩니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="promotion_started",
        category="광고/홍보",
        variants=(
            {"app": "Shopping", "sender": "가상 패션몰", "subject": "가을 의류 할인"},
            {"app": "Store", "sender": "가상 전자몰", "subject": "노트북 할인"},
            {"app": "Shopping", "sender": "가상 식품몰", "subject": "주말 식품 할인"},
            {"app": "Store", "sender": "가상 문구점", "subject": "신학기 문구 할인"},
            {"app": "Shopping", "sender": "가상 리빙몰", "subject": "생활용품 할인"},
        ),
        notifications=(
            NotificationTemplate("{subject} 시작", "{subject} 행사가 시작되었습니다."),
            NotificationTemplate("{subject} 기간", "행사는 {date}까지 진행됩니다."),
        ),
        target_lines=("{subject} 행사가 {date}까지 진행됩니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_maintenance_rescheduled",
        category="기타",
        variants=(
            {"app": "Notice", "sender": "가상 시설팀", "subject": "엘리베이터 점검"},
            {"app": "Community", "sender": "가상 관리실", "subject": "주차장 점검"},
            {"app": "Notice", "sender": "가상 전산실", "subject": "네트워크 점검"},
            {"app": "Community", "sender": "가상 운영센터", "subject": "냉난방 점검"},
            {"app": "Notice", "sender": "가상 안전팀", "subject": "소방 설비 점검"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 일정 변경 안내입니다."),
            NotificationTemplate("{sender}", "기존 일정은 {old_date}였어요."),
            NotificationTemplate("{sender}", "새 일정은 {new_date} {time}입니다."),
            NotificationTemplate("{sender}", "변경된 시간에 점검을 진행합니다."),
        ),
        target_lines=("{subject} 일정이 {new_date} {time}로 변경되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="lost_found_notice",
        category="기타",
        variants=(
            {"app": "Community", "sender": "가상 학생지원팀", "subject": "검은색 우산", "place": "학생회관 안내실"},
            {"app": "Notice", "sender": "가상 도서관", "subject": "무선 이어폰", "place": "도서관 안내 데스크"},
            {"app": "Community", "sender": "가상 관리실", "subject": "카드 지갑", "place": "본관 관리실"},
            {"app": "Notice", "sender": "가상 체육관", "subject": "운동 가방", "place": "체육관 접수대"},
            {"app": "Community", "sender": "가상 행정실", "subject": "학생증", "place": "행정실"},
        ),
        notifications=(
            NotificationTemplate("분실물 발견", "{subject}이 발견되었습니다."),
            NotificationTemplate("분실물 보관 안내", "발견된 {subject}은 {place}에 보관 중입니다."),
        ),
        target_lines=("발견된 {subject}은 {place}에 보관 중입니다.",),
        max_summary_lines=1,
    ),
)


COLLOQUIAL_SCENARIOS = (
    Scenario(
        name="fragmented_colloquial_incident",
        category="긴급 업무",
        variants=(
            {"app": "KakaoTalk", "sender": "가상 운영팀장", "subject": "결제 서버", "symptom": "응답이 계속 끊겨", "action": "긴급 복구"},
            {"app": "Slack", "sender": "가상 백엔드 리더", "subject": "로그인 API", "symptom": "500 오류가 반복돼", "action": "원인 확인"},
            {"app": "Teams", "sender": "가상 인프라 담당자", "subject": "파일 서버", "symptom": "접속이 안 돼", "action": "장비 전환"},
            {"app": "Discord", "sender": "가상 배포 담당자", "subject": "주문 서비스", "symptom": "배포 뒤에 장애 났어", "action": "롤백"},
            {"app": "KakaoTalk", "sender": "가상 SRE 리더", "subject": "검색 API", "symptom": "지연이 너무 길어", "action": "트래픽 우회"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 그거"),
            NotificationTemplate("{sender}", "아까부터 {symptom}"),
            NotificationTemplate("{sender}", "지금 {action} 중"),
            NotificationTemplate("{sender}", "끝나면 다시 말할게"),
        ),
        target_lines=("{subject}에 문제가 발생해 {action} 중입니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_colloquial_task",
        category="일반 업무",
        variants=(
            {"app": "KakaoTalk", "sender": "가상 팀장", "subject": "발표 자료", "artifact": "결론 부분", "place": "공유 폴더"},
            {"app": "Slack", "sender": "가상 리뷰어", "subject": "코드 리뷰", "artifact": "실패 테스트", "place": "개발 채널"},
            {"app": "Teams", "sender": "가상 기획자", "subject": "요구사항 문서", "artifact": "변경 내역", "place": "프로젝트 보드"},
            {"app": "Discord", "sender": "가상 전시 팀장", "subject": "부스 안내문", "artifact": "오탈자", "place": "전시 준비 채널"},
            {"app": "KakaoTalk", "sender": "가상 조교", "subject": "과제 보고서", "artifact": "참고문헌", "place": "과제 게시판"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "저번에 말한 {subject} 있잖아"),
            NotificationTemplate("{sender}", "그거 {artifact}만 고쳐서"),
            NotificationTemplate("{sender}", "{date} {time}까지"),
            NotificationTemplate("{sender}", "{place}에 다시 올려줘"),
        ),
        target_lines=("{subject}에서 {artifact} 관련 내용을 수정해 {date} {time}까지 {place}에 다시 올려야 합니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_colloquial_schedule",
        category="일정/회의",
        variants=(
            {"app": "KakaoTalk", "sender": "가상 스터디장", "subject": "AI 스터디", "old_time": "오후 3시", "place": "B강의실"},
            {"app": "Slack", "sender": "가상 프로젝트 리더", "subject": "중간 점검", "old_time": "오전 10시", "place": "회의실 2"},
            {"app": "Teams", "sender": "가상 팀장", "subject": "주간 회의", "old_time": "오후 2시", "place": "온라인 회의실"},
            {"app": "Discord", "sender": "가상 동아리 회장", "subject": "전시 준비 모임", "old_time": "오후 5시", "place": "창의관 402호"},
            {"app": "KakaoTalk", "sender": "가상 조교", "subject": "과제 질의응답", "old_time": "오전 11시", "place": "공학관 301호"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 있잖아"),
            NotificationTemplate("{sender}", "원래 {old_time}였는데"),
            NotificationTemplate("{sender}", "{date} {time}로 바뀜"),
            NotificationTemplate("{sender}", "장소는 {place} 그대로래"),
        ),
        target_lines=("{subject}가 {date} {time}로 변경되었으며 장소는 기존과 동일한 {place}입니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_colloquial_security",
        category="시스템/보안",
        variants=(
            {"app": "KakaoTalk", "sender": "가상 보안 담당자", "subject": "업무 계정", "device": "Windows PC", "place": "부산"},
            {"app": "Slack", "sender": "가상 인증팀", "subject": "개발자 계정", "device": "MacBook", "place": "대전"},
            {"app": "Teams", "sender": "가상 계정보호팀", "subject": "학교 포털", "device": "Android 기기", "place": "제주"},
            {"app": "Discord", "sender": "가상 보안 봇", "subject": "관리자 계정", "device": "Linux PC", "place": "광주"},
            {"app": "KakaoTalk", "sender": "가상 보안센터", "subject": "쇼핑 계정", "device": "iPhone", "place": "인천"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 로그인 알림 뜬 거"),
            NotificationTemplate("{sender}", "{place}에서 {device}로 들어왔대"),
            NotificationTemplate("{sender}", "내가 한 거 아니라고 했고"),
            NotificationTemplate("{sender}", "기기 차단이랑 로그아웃 처리됨"),
        ),
        target_lines=("{place}의 {device}에서 발생한 {subject} 로그인을 본인 활동이 아닌 것으로 확인해 기기를 차단하고 로그아웃 처리했습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_colloquial_appointment",
        category="개인 중요",
        variants=(
            {"app": "KakaoTalk", "sender": "가상 치과", "subject": "치과 진료", "place": "3층 진료실"},
            {"app": "KakaoTalk", "sender": "가상 병원", "subject": "건강 검진", "place": "검진센터"},
            {"app": "Slack", "sender": "가상 상담센터", "subject": "상담 예약", "place": "온라인 상담실"},
            {"app": "Teams", "sender": "가상 안과", "subject": "안과 진료", "place": "2층 접수처"},
            {"app": "KakaoTalk", "sender": "가상 검진센터", "subject": "예방 접종", "place": "본관 접종실"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "전에 잡은 {subject} 있죠"),
            NotificationTemplate("{sender}", "그거 {date} {time}로 변경됐어요"),
            NotificationTemplate("{sender}", "{place}로 오시면 되고"),
            NotificationTemplate("{sender}", "시간 맞춰 와주세요"),
        ),
        target_lines=("{subject} 예약이 {date} {time}로 변경되었으며 {place}로 방문해야 합니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_colloquial_delivery",
        category="개인 일반",
        variants=(
            {"app": "KakaoTalk", "sender": "가상 택배기사", "subject": "생활용품", "place": "현관 앞"},
            {"app": "KakaoTalk", "sender": "가상 배송기사", "subject": "도서", "place": "무인 보관함"},
            {"app": "Slack", "sender": "가상 물류팀", "subject": "전자기기", "place": "경비실"},
            {"app": "Teams", "sender": "가상 판매자", "subject": "의류", "place": "택배 보관실"},
            {"app": "KakaoTalk", "sender": "가상 배송센터", "subject": "문구류", "place": "관리실"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "{subject} 시킨 거"),
            NotificationTemplate("{sender}", "방금 도착했고"),
            NotificationTemplate("{sender}", "{place}에 뒀어요"),
        ),
        target_lines=("주문한 {subject}이 {place}에 배송 완료되었습니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_colloquial_promotion",
        category="광고/홍보",
        variants=(
            {"app": "KakaoTalk", "sender": "가상 쇼핑몰", "subject": "신규 회원 쿠폰"},
            {"app": "KakaoTalk", "sender": "가상 카페", "subject": "음료 할인 쿠폰"},
            {"app": "Slack", "sender": "가상 마켓", "subject": "무료 배송 쿠폰"},
            {"app": "Teams", "sender": "가상 서점", "subject": "도서 할인 쿠폰"},
            {"app": "KakaoTalk", "sender": "가상 브랜드몰", "subject": "생일 쿠폰"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "그 {subject} 있잖아요"),
            NotificationTemplate("{sender}", "아직 안 쓴 거"),
            NotificationTemplate("{sender}", "{date}까지만 된대요"),
            NotificationTemplate("{sender}", "지나면 없어짐"),
        ),
        target_lines=("사용하지 않은 {subject}은 {date}에 만료됩니다.",),
        max_summary_lines=1,
    ),
    Scenario(
        name="fragmented_colloquial_notice",
        category="기타",
        variants=(
            {"app": "KakaoTalk", "sender": "가상 행정실", "subject": "검은색 우산", "place": "학생회관 안내실"},
            {"app": "KakaoTalk", "sender": "가상 도서관", "subject": "무선 이어폰", "place": "도서관 안내 데스크"},
            {"app": "Slack", "sender": "가상 관리실", "subject": "카드 지갑", "place": "본관 관리실"},
            {"app": "Teams", "sender": "가상 체육관", "subject": "운동 가방", "place": "체육관 접수대"},
            {"app": "KakaoTalk", "sender": "가상 학생지원팀", "subject": "학생증", "place": "행정실"},
        ),
        notifications=(
            NotificationTemplate("{sender}", "누가 두고 간 {subject}"),
            NotificationTemplate("{sender}", "그거 찾은 거 같아요"),
            NotificationTemplate("{sender}", "지금 {place}에 있고"),
            NotificationTemplate("{sender}", "거기로 찾으러 오면 됨"),
        ),
        target_lines=("발견된 {subject}은 {place}에 보관 중이므로 해당 장소에서 찾아가야 합니다.",),
        max_summary_lines=1,
    ),
)


SCENARIOS = BASE_SCENARIOS + ADDITIONAL_SCENARIOS + COLLOQUIAL_SCENARIOS

TARGETED_SCENARIO_NAMES = (
    "appointment_cancelled",
    "fragmented_schedule_chat",
    "fragmented_task_chat",
    "password_reset_completed",
    "fragmented_colloquial_incident",
    "fragmented_colloquial_task",
    "fragmented_colloquial_schedule",
    "fragmented_colloquial_security",
    "fragmented_colloquial_appointment",
    "fragmented_colloquial_delivery",
    "fragmented_colloquial_promotion",
    "fragmented_colloquial_notice",
)
TARGETED_SCENARIOS = tuple(
    scenario for scenario in SCENARIOS
    if scenario.name in TARGETED_SCENARIO_NAMES
)

if len(TARGETED_SCENARIOS) != len(TARGETED_SCENARIO_NAMES):
    raise RuntimeError("a targeted fine-tuning scenario is not configured")


def _context_for(split: str, index: int) -> dict[str, str]:
    start = date(2026, 10, 1) if split == "train" else date(2027, 4, 1)
    current_date = start + timedelta(days=index)
    new_date = current_date + timedelta(days=2)
    times = ("오전 9시", "오전 11시", "오후 3시", "오후 6시")
    return {
        "date": f"{current_date.month}월 {current_date.day}일",
        "old_date": f"{current_date.month}월 {current_date.day}일",
        "new_date": f"{new_date.month}월 {new_date.day}일",
        "time": times[index % len(times)],
        "timestamp_date": current_date.isoformat(),
    }


def _variant_for(scenario: Scenario, split: str, index: int) -> Mapping[str, str]:
    if split == "validation":
        return scenario.variants[-1]

    variants = scenario.variants[:-1]
    variant_count = len(variants)
    identity = variants[index % variant_count]
    values = {"app": identity["app"], "sender": identity["sender"]}
    content_keys = sorted(
        key for key in variants[0] if key not in {"app", "sender"}
    )
    coordinate = index // variant_count
    for key in content_keys:
        source = variants[coordinate % variant_count]
        values[key] = source[key]
        coordinate //= variant_count
    return values


def _style_index(scenario: Scenario, split: str, index: int) -> int:
    if split == "validation":
        return index
    variant_count = len(scenario.variants) - 1
    content_field_count = len(
        {key for key in scenario.variants[0]} - {"app", "sender"}
    )
    combination_count = variant_count ** (content_field_count + 1)
    return index // combination_count


def _apply_surface_style(
    body: str,
    *,
    scenario: Scenario,
    split: str,
    index: int,
    step: int,
    sender: str,
) -> str:
    if step != 0:
        return body

    style_index = _style_index(scenario, split, index)
    if scenario.name.startswith("fragmented_"):
        train_prefixes = (
            "",
            f"{sender}에서 참고로 알려드려요. ",
            f"{sender}에서 먼저 말씀드리면, ",
            f"{sender} 추가 안내예요. ",
        )
        validation_prefixes = (
            "알려드릴게요. ",
            "확인 부탁해요. ",
            "먼저 말씀드리면, ",
            "참고해 주세요. ",
            "추가 안내예요. ",
            "이어서 알려드릴게요. ",
            "변경된 내용을 공유해요. ",
            "중요한 내용이에요. ",
            "관련 안내 남길게요. ",
            "마지막으로 확인해 주세요. ",
        )
        prefixes = train_prefixes if split == "train" else validation_prefixes
        return f"{prefixes[style_index % len(prefixes)]}{body}"

    train_prefixes = (
        "",
        f"{sender}에서 전달드립니다. ",
        f"{sender} 안내입니다. ",
        f"{sender}에서 확인을 요청했습니다. ",
    )
    validation_prefixes = (
        "새로운 안내입니다. ",
        f"{sender}에서 알려드립니다. ",
        "관련 내용을 확인해 주세요. ",
        f"{sender} 공지입니다. ",
        "중요 내용을 전달드립니다. ",
        "추가 정보를 전달드립니다. ",
        f"{sender}의 변경 안내입니다. ",
        "다음 내용을 꼭 확인해 주세요. ",
        "최신 상태를 공유드립니다. ",
        f"{sender}에서 추가로 안내합니다. ",
    )
    prefixes = train_prefixes if split == "train" else validation_prefixes
    return f"{prefixes[style_index % len(prefixes)]}{body}"


def build_record(scenario: Scenario, split: str, index: int) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if index < 0:
        raise ValueError("index must be non-negative")

    values = dict(_variant_for(scenario, split, index))
    values.update(_context_for(split, index))
    notifications = []
    for step, template in enumerate(scenario.notifications):
        body = template.body.format(**values)
        notifications.append(
            {
                "id": f"synthetic_{split}_{scenario.name}_{index:03d}_{step + 1}",
                "timestamp": f"{values['timestamp_date']}T08:{step:02d}:00Z",
                "title": template.title.format(**values),
                "body": _apply_surface_style(
                    body,
                    scenario=scenario,
                    split=split,
                    index=index,
                    step=step,
                    sender=values["sender"],
                ),
            }
        )

    return {
        "case_id": f"{split}_{scenario.name}_{index:03d}",
        "input": {
            "app_name": values["app"],
            "sender": values["sender"],
            "category": scenario.category,
            "notifications": notifications,
        },
        "max_summary_lines": scenario.max_summary_lines,
        "target": {
            "summary_lines": [line.format(**values) for line in scenario.target_lines]
        },
        "metadata": {
            "split": split,
            "scenario": scenario.name,
            "synthetic": True,
        },
    }


def generate_records(split: str, per_scenario: int) -> list[dict[str, Any]]:
    if per_scenario <= 0:
        raise ValueError("per_scenario must be positive")
    records = [
        build_record(scenario, split, index)
        for scenario in SCENARIOS
        for index in range(per_scenario)
    ]
    validate_records(records, expected_split=split)
    return records


def generate_balanced_records(split: str, total_count: int) -> list[dict[str, Any]]:
    """Build an exact-size dataset balanced across the official categories."""

    if total_count <= 0:
        raise ValueError("total_count must be positive")

    allocations: list[tuple[Scenario, int]] = []
    category_base, category_remainder = divmod(total_count, len(FILTER_CATEGORIES))
    for category_index, category in enumerate(FILTER_CATEGORIES):
        category_count = category_base + (category_index < category_remainder)
        category_scenarios = tuple(
            scenario for scenario in SCENARIOS if scenario.category == category
        )
        if not category_scenarios:
            raise ValueError(f"no scenarios configured for category: {category}")

        scenario_base, scenario_remainder = divmod(
            category_count, len(category_scenarios)
        )
        for scenario_index, scenario in enumerate(category_scenarios):
            scenario_count = scenario_base + (
                scenario_index < scenario_remainder
            )
            allocations.append((scenario, scenario_count))

    records_by_scenario: list[list[dict[str, Any]]] = []
    fingerprints: set[str] = set()
    for scenario, count in allocations:
        scenario_records: list[dict[str, Any]] = []
        candidate_index = 0
        while len(scenario_records) < count:
            if candidate_index >= 10_000:
                raise ValueError(
                    f"could not create {count} diverse records for {scenario.name}"
                )
            record = build_record(scenario, split, candidate_index)
            candidate_index += 1
            fingerprint = normalized_body_fingerprint(record)
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            scenario_records.append(record)
        records_by_scenario.append(scenario_records)

    records = [
        scenario_records[record_index]
        for record_index in range(max(map(len, records_by_scenario)))
        for scenario_records in records_by_scenario
        if record_index < len(scenario_records)
    ]

    validate_records(records, expected_split=split)
    validate_normalized_diversity(records)
    return records


def generate_targeted_records(
    split: str,
    total_count: int,
    *,
    existing_records: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Generate held-out variants focused on observed summary failure modes."""

    if total_count < 0:
        raise ValueError("targeted total_count must not be negative")
    if total_count == 0:
        return []

    fingerprints = {
        normalized_body_fingerprint(record) for record in existing_records
    }
    scenario_base, scenario_remainder = divmod(
        total_count, len(TARGETED_SCENARIOS)
    )
    records_by_scenario: list[list[dict[str, Any]]] = []
    for scenario_index, scenario in enumerate(TARGETED_SCENARIOS):
        scenario_count = scenario_base + (
            scenario_index < scenario_remainder
        )
        scenario_records: list[dict[str, Any]] = []
        candidate_index = 0
        while len(scenario_records) < scenario_count:
            if candidate_index >= 10_000:
                raise ValueError(
                    f"could not create {scenario_count} targeted records "
                    f"for {scenario.name}"
                )
            record = build_record(scenario, split, candidate_index)
            candidate_index += 1
            fingerprint = normalized_body_fingerprint(record)
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            record["case_id"] = (
                f"{split}_targeted_{scenario.name}_{candidate_index - 1:03d}"
            )
            record["metadata"]["augmentation"] = "failure_targeted"
            scenario_records.append(record)
        records_by_scenario.append(scenario_records)

    records = [
        scenario_records[record_index]
        for record_index in range(max(map(len, records_by_scenario)))
        for scenario_records in records_by_scenario
        if record_index < len(scenario_records)
    ]
    validate_records(records, expected_split=split)
    validate_normalized_diversity(records)
    return records


def _spread_targeted_records(
    base_records: Sequence[dict[str, Any]],
    targeted_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Distribute targeted examples across the generated JSONL deterministically."""

    if not targeted_records:
        return list(base_records)

    combined: list[dict[str, Any]] = []
    targeted_index = 0
    for base_index, record in enumerate(base_records, start=1):
        combined.append(record)
        desired_count = (
            base_index * len(targeted_records) // len(base_records)
        )
        while targeted_index < desired_count:
            combined.append(targeted_records[targeted_index])
            targeted_index += 1
    combined.extend(targeted_records[targeted_index:])
    return combined


def generate_augmented_records(
    split: str,
    *,
    base_count: int,
    targeted_count: int,
) -> list[dict[str, Any]]:
    """Generate the balanced base plus failure-targeted augmentation."""

    base_records = generate_balanced_records(split, base_count)
    targeted_records = generate_targeted_records(
        split,
        targeted_count,
        existing_records=base_records,
    )
    records = _spread_targeted_records(base_records, targeted_records)
    validate_records(records, expected_split=split)
    validate_normalized_diversity(records)
    return records


_DATE_PATTERN = re.compile(r"\d{1,2}월\s+\d{1,2}일")
_TIME_PATTERN = re.compile(r"(?:오전|오후)\s+\d{1,2}시")


def _normalize_generated_text(text: str) -> str:
    text = _DATE_PATTERN.sub("<DATE>", text)
    return _TIME_PATTERN.sub("<TIME>", text)


def normalized_input_fingerprint(record: Mapping[str, Any]) -> str:
    """Fingerprint visible input while ignoring IDs and date/time substitutions."""

    group = record["input"]

    visible = {
        "app_name": group["app_name"],
        "sender": group["sender"],
        "category": group["category"],
        "notifications": [
            {
                "title": _normalize_generated_text(notification["title"]),
                "body": _normalize_generated_text(notification["body"]),
            }
            for notification in group["notifications"]
        ],
    }
    return json.dumps(visible, ensure_ascii=False, sort_keys=True)


def normalized_body_fingerprint(record: Mapping[str, Any]) -> str:
    group = record["input"]
    bodies = [
        _normalize_generated_text(notification["body"])
        for notification in group["notifications"]
    ]
    return json.dumps(bodies, ensure_ascii=False)


def validate_normalized_diversity(records: Sequence[Mapping[str, Any]]) -> None:
    fingerprints: set[str] = set()
    for record in records:
        fingerprint = normalized_body_fingerprint(record)
        if fingerprint in fingerprints:
            raise ValueError(
                "dataset contains duplicate message bodies after normalizing dates and times"
            )
        fingerprints.add(fingerprint)


def training_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    group = record.get("input")
    target = record.get("target")
    max_summary_lines = record.get("max_summary_lines")
    if not isinstance(group, Mapping) or not isinstance(target, Mapping):
        raise ValueError("record must contain input and target objects")
    if not isinstance(max_summary_lines, int):
        raise ValueError("record must contain an integer max_summary_lines")
    messages = build_messages(group, max_summary_lines=max_summary_lines)
    return [
        *messages,
        {
            "role": "assistant",
            "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def validate_records(
    records: Sequence[Mapping[str, Any]], *, expected_split: str
) -> None:
    case_ids: set[str] = set()
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every record must have a case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)

        metadata = record.get("metadata")
        group = record.get("input")
        target = record.get("target")
        max_summary_lines = record.get("max_summary_lines")
        if not isinstance(metadata, Mapping) or metadata.get("split") != expected_split:
            raise ValueError(f"{case_id} has an invalid split")
        if not isinstance(group, Mapping) or group.get("category") not in FILTER_CATEGORIES:
            raise ValueError(f"{case_id} has an invalid category")
        if not isinstance(target, Mapping) or not isinstance(max_summary_lines, int):
            raise ValueError(f"{case_id} has an invalid target contract")
        raw_target = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
        parsed = parse_summary_response(raw_target, allow_list_repair=False)
        if len(parsed) > max_summary_lines or max_summary_lines > MAX_SUMMARY_LINES:
            raise ValueError(f"{case_id} exceeds its line limit")
        training_messages(record)


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    )
    path.write_text(f"{content}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument(
        "--validation-path", type=Path, default=DEFAULT_VALIDATION_PATH
    )
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=120)
    parser.add_argument("--targeted-train-count", type=int, default=200)
    parser.add_argument("--targeted-validation-count", type=int, default=40)
    parser.add_argument("--train-per-scenario", type=int)
    parser.add_argument("--validation-per-scenario", type=int)
    args = parser.parse_args()

    train_records = (
        generate_records("train", args.train_per_scenario)
        if args.train_per_scenario is not None
        else generate_augmented_records(
            "train",
            base_count=args.train_count,
            targeted_count=args.targeted_train_count,
        )
    )
    validation_records = (
        generate_records("validation", args.validation_per_scenario)
        if args.validation_per_scenario is not None
        else generate_augmented_records(
            "validation",
            base_count=args.validation_count,
            targeted_count=args.targeted_validation_count,
        )
    )
    write_jsonl(args.train_path, train_records)
    write_jsonl(args.validation_path, validation_records)
    print(f"Wrote {len(train_records)} training cases to {args.train_path}")
    print(
        f"Wrote {len(validation_records)} validation cases "
        f"to {args.validation_path}"
    )


if __name__ == "__main__":
    main()
