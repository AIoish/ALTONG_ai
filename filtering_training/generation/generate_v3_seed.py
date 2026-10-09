"""Create a fresh, human-reviewable filtering dataset seed.

Every notification and context is fictional and authored here. The seed is
provisional; it is not mixed with the previous 5,000 synthetic examples.
"""

from filtering_training.common.paths import OUTPUTS_ROOT

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.filtering.schema import FilteringSample
from src.filtering.policy import POLICY_VERSION, decision_label
from src.filtering.prompt import CATEGORIES, SYSTEM_PROMPT, build_messages, parse_model_output


OUTPUT_DIR = OUTPUTS_ROOT / "v3_seed"

# The same windows are deliberately reused for both related and unrelated alerts.
CONTEXTS = [
    ("Code.exe", "checkout_api.py - 결제 API 장애 대응 - Visual Studio Code"),
    ("WindowsTerminal.exe", "운영 DB 장애 대응 - Terminal"),
    ("Code.exe", "SearchButton.tsx - 검색 화면 버튼 수정 - Visual Studio Code"),
    ("Code.exe", "SearchButton.tsx - 검색 화면 버튼 수정 - Visual Studio Code"),
    ("ms-teams.exe", "발표 회의 입장 - Teams"),
    ("OUTLOOK.EXE", "다음 주 팀 회의 일정 조율 - Outlook"),
    ("SecHealthUI.exe", "Windows 보안 - 보호 기록"),
    ("SecHealthUI.exe", "Windows 보안 - 보호 기록"),
    ("chrome.exe", "가족 병원 진료 일정 - Chrome"),
    ("chrome.exe", "가스점검 방문 시간 확인 - Chrome"),
    ("chrome.exe", "시험 공부 계획 - Chrome"),
    ("chrome.exe", "시험 공부 계획 - Chrome"),
    ("chrome.exe", "운동화 상품 비교 - Chrome"),
    ("chrome.exe", "러닝화 R100 가격 비교 - Chrome"),
    ("msedge.exe", "강의자료 다운로드 목록 - Microsoft Edge"),
    ("ONENOTE.EXE", "스크린샷을 보고서에 붙이기 - OneNote"),
]

# app, sender, displayed title, raw body, category, urgency, related relevance.
MESSAGES = [
    ("KakaoTalk.exe", "지현", "지현", "결제 API 또 500 떠. 지금 주문 다 실패하는데 확인 가능?", "긴급 업무", 5, 5),
    ("Slack", "운영 담당", "#운영-당직", "DB 쓰기 실패 중입니다. 새 주문이 저장되지 않아요. 당직자 확인 부탁드립니다", "긴급 업무", 5, 5),
    ("KakaoTalk.exe", "준호", "준호", "검색 화면 버튼 수정한 거 올렸어요. 오늘 중으로 리뷰 부탁드려요", "일반 업무", 3, 5),
    ("GitHub", "도윤", "검색 프로젝트 · PR #27", "README 설치 안내 오타 수정했습니다. 다음 주에 시간 될 때 봐주세요", "일반 업무", 2, 3),
    ("Microsoft Teams", "발표 진행자", "회의 링크 변경", "지금 시작한 발표 회의 링크가 바뀌었어요. 새 링크로 들어와 주세요", "일정/회의", 4, 5),
    ("Outlook", "서연", "다음 주 팀 회의 시간", "다음 주 팀 회의는 화요일 10시 괜찮으세요? 월요일까지 답 주시면 일정 잡을게요", "일정/회의", 2, 4),
    ("Windows 보안", "", "위협 대응 필요", "악성 프로그램 실행을 차단하지 못했습니다. 같은 프로그램의 실행 시도가 계속 감지됩니다.", "시스템/보안", 5, 5),
    ("Windows 보안", "", "검사 완료", "빠른 검사가 완료되었습니다. 위협이 발견되지 않았습니다.", "시스템/보안", 1, 4),
    ("KakaoTalk.exe", "누나", "누나", "아빠 지금 응급실 가는 중이야. 전화 좀 받아", "개인 중요", 5, 5),
    ("문자 메시지", "관리사무소", "관리사무소", "내일 가스점검 방문 예정입니다. 오늘 중 방문 가능한 시간 알려주세요.", "개인 중요", 3, 5),
    ("KakaoTalk.exe", "민수", "민수", "야 내일 8시 독서실 가서 시험 공부 ㄱㄱ?", "개인 일반", 2, 4),
    ("KakaoTalk.exe", "엄마", "엄마", "너 어제 독서실에서 또 졸았다며ㅋㅋ 공부하러 간 거 맞아?", "개인 일반", 1, 3),
    ("KakaoTalk.exe", "운동화 브랜드 채널", "운동화 브랜드 채널", "오늘만 운동화 20% 할인! 쿠폰 받기👟", "광고/홍보", 1, 4),
    ("Gmail", "쇼핑몰 알림", "관심 상품 가격이 내려갔어요", "찜한 러닝화 R100이 79,000원 → 59,000원! 할인은 오늘 자정까지입니다.", "광고/홍보", 2, 5),
    ("Microsoft Edge", "", "다운로드 완료", "강의자료.pdf 다운로드가 끝났습니다", "기타", 1, 5),
    ("캡처 도구", "", "캡처 완료", "이미지가 클립보드에 복사되었습니다.", "기타", 1, 5),
]

# Every unique window also appears with an unrelated notification.
UNRELATED_CONTEXT = [10, 12, 8, 9, 14, 1, 15, 13, 2, 5, 0, 4, 6, 10, 1, 8]
EVIDENCE = [
    ("주문 실패가 계속되어 즉시 대응이 필요합니다", "현재 결제 API 장애 대응과 직접 연결됩니다"),
    ("새 주문 저장 실패가 진행 중이라 즉시 대응이 필요합니다", "현재 운영 DB 장애 대응에 필요한 정보입니다"),
    ("오늘 중 리뷰 요청이며 즉각적인 장애는 명시되지 않았습니다", "현재 수정 중인 검색 화면 버튼과 직접 연결됩니다"),
    ("다음 주에 확인해도 되는 문서 수정입니다", "같은 검색 프로젝트지만 현재 버튼 수정에 주는 도움은 불분명합니다"),
    ("이미 시작한 회의에 입장하려면 바로 확인해야 합니다", "현재 발표 회의 입장에 필요한 정보입니다"),
    ("월요일까지 답할 수 있는 다음 주 일정 조율입니다", "현재 다음 주 팀 회의 일정을 정하는 데 도움이 됩니다"),
    ("차단 실패와 반복 실행이 명시되어 즉시 보호 조치가 필요합니다", "현재 보안 보호 기록과 직접 연결됩니다"),
    ("검사가 끝났고 발견된 위협이 없어 긴급한 조치가 없습니다", "현재 보안 보호 기록을 확인하는 데 도움이 되는 결과입니다"),
    ("가족의 응급실 이동과 연락 요청으로 즉시 확인이 필요합니다", "현재 가족 병원 진료 일정을 확인하는 상황과 직접 연결됩니다"),
    ("오늘 중 회신이 필요하지만 즉시 대응해야 하는 상황은 아닙니다", "현재 확인 중인 가스점검 방문 시간을 정하는 요청입니다"),
    ("내일 공부 약속을 제안하며 당장 답하지 않아도 됩니다", "현재 시험 공부 계획을 세우는 데 도움이 되는 제안입니다"),
    ("장난스러운 농담으로 즉시 응답할 필요가 없습니다", "공부라는 주제는 같지만 현재 공부 계획에 주는 도움은 불분명합니다"),
    ("선택적인 할인 안내로 긴급한 조치가 필요하지 않습니다", "현재 운동화 비교와 구매 목적에 도움이 되는 할인 정보입니다"),
    ("할인을 놓쳐도 큰 피해가 명시되지 않은 선택적 구매 안내입니다", "정확히 비교 중인 러닝화 R100의 가격 정보입니다"),
    ("완료 상태 안내로 긴급한 대응은 필요하지 않습니다", "현재 다운로드 중인 강의자료의 완료 결과입니다"),
    ("클립보드 복사 완료 안내로 긴급한 조치는 없습니다", "현재 보고서에 스크린샷을 붙이는 작업에 필요한 결과입니다"),
]


def build():
    if not len(MESSAGES) == len(CONTEXTS) == len(UNRELATED_CONTEXT) == len(EVIDENCE):
        raise ValueError("messages, contexts and evidence must have equal lengths")
    if any(index == unrelated for index, unrelated in enumerate(UNRELATED_CONTEXT)):
        raise ValueError("a message cannot use its matching context as unrelated")
    start = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)
    samples = []
    review = []
    for index, (app, sender, title, body, category, urgency, related_score) in enumerate(MESSAGES):
        review.append({
            "message_id": f"v3_{index + 1:02d}",
            "app_name": app,
            "sender": sender,
            "title": title,
            "body": body,
            "category": category,
            "urgency_score": urgency,
            "related_window": CONTEXTS[index][1],
            "related_relevance": related_score,
            "related_decision": decision_label(urgency, related_score),
            "unrelated_window": CONTEXTS[UNRELATED_CONTEXT[index]][1],
            "unrelated_relevance": 1,
            "unrelated_decision": decision_label(urgency, 1),
            "empty_relevance": 1,
            "empty_decision": decision_label(urgency, 1),
            "feedback": "",
        })
        for role, context_index, relevance in (
            ("related", index, related_score),
            ("unrelated", UNRELATED_CONTEXT[index], 1),
            ("empty", None, 1),
        ):
            process, window = CONTEXTS[context_index] if context_index is not None else ("", "")
            when = start + timedelta(minutes=index)
            urgency_reason, relevance_reason = EVIDENCE[index]
            if role == "unrelated":
                relevance_reason = "현재 창의 작업과 다른 주제입니다"
            elif role == "empty":
                relevance_reason = "현재 작업 정보가 없어 관련성을 확인할 수 없습니다"
            sample = FilteringSample.model_validate({
                "notification": {
                    "id": f"v3_{index + 1:02d}_{role}",
                    "app_name": app,
                    "sender": sender,
                    "title": title,
                    "body": body,
                    "timestamp": when.isoformat().replace("+00:00", "Z"),
                },
                "context": {
                    "active_process": process,
                    "window_title": window,
                    "last_updated": (when - timedelta(seconds=3)).isoformat().replace("+00:00", "Z"),
                    "duration_seconds": (25, 65, 90, 180)[index % 4] if process else 0,
                    "recent_processes": [process] if process else [],
                },
                "label": {
                    "urgency_score": urgency,
                    "relevance_score": relevance,
                    "category": category,
                    "ai_summary_reason": f"{relevance_reason}; {urgency_reason}.",
                },
            })
            parse_model_output(sample.label.model_dump_json())
            samples.append(sample)
    by_context = defaultdict(set)
    for sample in samples:
        if sample.context.active_process:
            by_context[(sample.context.active_process, sample.context.window_title)].add(sample.label.relevance_score)
    if set(by_context) != set(CONTEXTS) or any(1 not in scores or max(scores) < 3 for scores in by_context.values()):
        raise ValueError("each context must have both related and unrelated scores")
    if Counter(row["category"] for row in review) != {category: 2 for category in CATEGORIES}:
        raise ValueError("need two messages in each category")
    return samples, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("choose an empty output directory; never overwrite an existing seed or review")
    samples, review = build()
    review_path = args.output_dir / "review.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8-sig", newline="") as file:
            if any(row.get("feedback", "").strip() for row in csv.DictReader(file)):
                raise ValueError("review feedback exists; choose another output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(sample.model_dump_json() + "\n")
    with review_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    with (args.output_dir / "model_io.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps({
                "id": sample.notification.id,
                "messages": build_messages(sample.notification, sample.context),
                "expected_output": sample.label.model_dump(),
                "policy_decision": decision_label(sample.label.urgency_score, sample.label.relevance_score),
            }, ensure_ascii=False) + "\n")
    lines = [
        "# 새 필터링 데이터: 첫 검수 16개", "",
        "모든 알림·발신자는 합성 예시입니다. 집중 모드 ON을 전제로 한 예상 정답이며 실제 모델 추론 결과가 아닙니다.",
        "긴급도 4 이상 또는 관련도 4 이상이면 PASS입니다. 같은 알림을 세 문맥에서 비교합니다.",
        "16개 원문을 읽고 어색한 문구나 점수만 review.csv의 feedback 열에 적어 저장해 주세요.",
        "예: 본문을 …로 변경 / 기본 관련도 3 / 긴급도 2 / 판단이 애매함.",
        "검수 완료를 알려주시면 의견 없는 항목은 승인으로 처리합니다. 검수가 끝날 때까지 확장·학습을 진행하지 않습니다.",
        "", "전체 모델 입력·예상 출력은 model_io.jsonl에 있습니다.", "",
    ]
    for index, row in enumerate(review):
        urgency_reason, relation = EVIDENCE[index]
        lines.extend([
            f"## {row['message_id']} · {row['category']}", "",
            f"앱: {row['app_name']} / 발신자: {row['sender'] or '(없음)'} / 제목: {row['title']}", "",
            f"> {row['body']}", "", f"긴급도: **{row['urgency_score']}** — {urgency_reason}", "",
            "| 문맥 | 관련도 | 결과 |", "|---|---:|---|",
            f"| {row['related_window']} | {row['related_relevance']} | {row['related_decision']} |",
            f"| {row['unrelated_window']} | 1 | {row['unrelated_decision']} |",
            f"| 빈 창 · 집중 모드 ON | 1 | {row['empty_decision']} |", "",
            f"기본 문맥 판단: {relation}.", "",
        ])
    (args.output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "status": "provisional seed; wording and labels require human review before expansion or training",
        "source": "new fictional authored messages; no earlier training rows copied",
        "notification_count": len(review),
        "sample_count": len(samples),
        "context_count": len(CONTEXTS),
        "unique_context_count": len(set(CONTEXTS)),
        "categories": dict(Counter(row["category"] for row in review)),
        "apps": dict(Counter(row["app_name"] for row in review)),
        "focus_mode": "ON assumed; empty window does not imply idle",
        "policy_version": POLICY_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "dataset_sha256": hashlib.sha256((args.output_dir / "candidates.jsonl").read_bytes()).hexdigest(),
        "context_balance": "each populated window has relevance 1 and relevance >=3",
        "urgency_counts": dict(Counter(row["urgency_score"] for row in review)),
        "relevance_counts": dict(Counter(sample.label.relevance_score for sample in samples)),
        "policy_counts": dict(Counter(decision_label(s.label.urgency_score, s.label.relevance_score) for s in samples)),
        "split_status": "unassigned until review; keep notification variants together",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
