"""Create deterministic synthetic data for briefing-summary fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
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
    urgency: tuple[int, int, float]
    relevance: tuple[int, int, float]


SCENARIOS = (
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
        urgency=(4, 5, 4.5),
        relevance=(4, 5, 4.5),
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
        urgency=(2, 3, 2.5),
        relevance=(4, 5, 4.5),
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
        urgency=(2, 4, 3.0),
        relevance=(4, 5, 4.5),
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
        urgency=(5, 5, 5.0),
        relevance=(5, 5, 5.0),
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
        urgency=(3, 4, 3.5),
        relevance=(4, 5, 4.5),
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
            NotificationTemplate("{subject} 배송 완료", "{subject}이 {place}에 배송 완료되었습니다."),
        ),
        target_lines=("{subject}이 {place}에 배송 완료되었습니다.",),
        max_summary_lines=1,
        urgency=(1, 2, 1.5),
        relevance=(2, 3, 2.5),
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
        urgency=(1, 2, 1.5),
        relevance=(1, 3, 2.0),
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
        urgency=(1, 3, 2.0),
        relevance=(2, 4, 3.0),
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
        urgency=(3, 4, 3.5),
        relevance=(4, 5, 4.5),
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
            NotificationTemplate("{subject} 제출 안내", "{subject}은 {date} {time}까지 제출해 주세요."),
            NotificationTemplate("{subject} 제출 형식", "{artifact}를 함께 제출해야 합니다."),
        ),
        target_lines=(
            "{subject}은 {date} {time}까지 제출해야 합니다.",
            "{artifact}를 함께 제출해야 합니다.",
        ),
        max_summary_lines=2,
        urgency=(3, 4, 3.5),
        relevance=(5, 5, 5.0),
    ),
)


def _context_for(split: str, index: int) -> dict[str, str]:
    month = 10 if split == "train" else 11
    day = index + 1
    old_day = day
    new_day = day + 2
    times = ("오전 9시", "오전 11시", "오후 3시", "오후 6시")
    return {
        "date": f"{month}월 {day}일",
        "old_date": f"{month}월 {old_day}일",
        "new_date": f"{month}월 {new_day}일",
        "time": times[index % len(times)],
    }


def _variant_for(scenario: Scenario, split: str, index: int) -> Mapping[str, str]:
    if split == "validation":
        return scenario.variants[-1]
    return scenario.variants[index % (len(scenario.variants) - 1)]


def _score(values: tuple[int, int, float]) -> dict[str, int | float]:
    minimum, maximum, average = values
    return {"min": minimum, "max": maximum, "average": average}


def build_record(scenario: Scenario, split: str, index: int) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if index < 0:
        raise ValueError("index must be non-negative")

    values = dict(_variant_for(scenario, split, index))
    values.update(_context_for(split, index))
    month = 10 if split == "train" else 11
    day = index + 1
    notifications = []
    for step, template in enumerate(scenario.notifications):
        notifications.append(
            {
                "id": f"synthetic_{split}_{scenario.name}_{index:03d}_{step + 1}",
                "timestamp": f"2026-{month:02d}-{day:02d}T09:{step * 10:02d}:00Z",
                "title": template.title.format(**values),
                "body": template.body.format(**values),
            }
        )

    return {
        "case_id": f"{split}_{scenario.name}_{index:03d}",
        "input": {
            "app_name": values["app"],
            "sender": values["sender"],
            "category": scenario.category,
            "urgency_score": _score(scenario.urgency),
            "relevance_score": _score(scenario.relevance),
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
        parsed = parse_summary_response(raw_target)
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
    parser.add_argument("--train-per-scenario", type=int, default=20)
    parser.add_argument("--validation-per-scenario", type=int, default=4)
    args = parser.parse_args()

    train_records = generate_records("train", args.train_per_scenario)
    validation_records = generate_records(
        "validation", args.validation_per_scenario
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
