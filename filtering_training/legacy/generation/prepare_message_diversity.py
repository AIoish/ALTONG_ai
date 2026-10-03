"""Build diverse, authored message-style filtering candidates for wording review.

All names and messages are fictional. Outputs are provisional and stay local.
"""

from filtering_training.common.paths import LEGACY_OUTPUTS_ROOT

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.filtering.schema import FilteringSample


ROOT = LEGACY_OUTPUTS_ROOT / "review"
CONTEXTS = {
    "study": ("chrome.exe", "시험 공부 계획 - Chrome"),
    "friends": ("KakaoTalk.exe", "친구 단체 채팅 - KakaoTalk"),
    "family": ("KakaoTalk.exe", "가족 단체 채팅 - KakaoTalk"),
    "hospital": ("chrome.exe", "가족 병원 진료 일정 - Chrome"),
    "payments": ("Code.exe", "결제 API 장애 대응 - Visual Studio Code"),
    "search": ("Code.exe", "검색 화면 버튼 수정 - Visual Studio Code"),
    "assignment": ("chrome.exe", "과제 제출 화면 - Chrome"),
    "shopping": ("chrome.exe", "운동화 상품 비교 - Chrome"),
    "report": ("EXCEL.EXE", "내일 회의 보고서 - Excel"),
    "database": ("WindowsTerminal.exe", "운영 DB 장애 대응 - Terminal"),
    "meeting": ("ms-teams.exe", "발표 회의 입장 - Teams"),
    "appointment": ("chrome.exe", "병원 예약 일정 - Chrome"),
}
# app, sender/title, actual notification body, category, urgency, relevant
# context, unrelated context, relevance in the relevant context.
MESSAGES = [
    ("KakaoTalk", "민수", "야 내일 8시 독서실 가서 시험 공부 ㄱㄱ?",
     "개인 일반", 2, "study", "payments", 5),
    ("KakaoTalk", "수빈", "너 어제 독서실에서 펜 들고 잤다며ㅋㅋ 사진 봤음",
     "개인 일반", 1, "friends", "database", 4),
    ("KakaoTalk", "엄마", "냉장고에 반찬 넣어뒀어. 또 배달시키지 마ㅋㅋ",
     "개인 일반", 1, "family", "search", 4),
    ("KakaoTalk", "누나", "아빠 지금 응급실 가는 중이야. 전화 좀 받아",
     "개인 중요", 5, "hospital", "payments", 5),
    ("KakaoTalk", "지현 · 개발팀", "결제 API 또 500 떠. 지금 주문 다 실패하는데 확인 가능?",
     "긴급 업무", 5, "payments", "study", 5),
    ("KakaoTalk", "준호 · 개발팀", "검색 화면 버튼 색 바꿨는데 내일 시간 될 때 봐줘",
     "일반 업무", 2, "search", "family", 5),
    ("KakaoTalk", "유진 · 같은 수업", "과제 제출 10분 남았어! 파일 올렸어?",
     "개인 중요", 4, "assignment", "database", 5),
    ("KakaoTalk", "운동화 브랜드 채널", "오늘만 운동화 20% 할인! 쿠폰 받기",
     "광고/홍보", 1, "shopping", "payments", 4),
    ("Slack", "서연 · 기획팀", "내일 회의용 표 숫자만 오늘 중으로 확인 부탁드려요",
     "일반 업무", 3, "report", "friends", 5),
    ("Slack", "운영 당직 채널", "운영 DB 쓰기 실패 중입니다. 새 주문 데이터가 저장되지 않아요. 확인 부탁드립니다",
     "긴급 업무", 5, "database", "study", 5),
    ("Microsoft Teams", "발표 진행자", "지금 시작한 발표 회의 링크가 바뀌었어요. 새 링크로 들어와 주세요",
     "일정/회의", 4, "meeting", "shopping", 5),
    ("문자 메시지", "병원 예약센터", "오늘 15:00 검사 예약입니다. 30분 내 접수해 주세요",
     "개인 중요", 4, "appointment", "search", 5),
]


def build():
    stamp = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
    samples = []
    review = []
    for index, (app, sender, body, category, urgency, related, unrelated, relevance) in enumerate(MESSAGES, 1):
        review.append({
            "message_id": f"msgdiv_{index:02d}",
            "app_name": app,
            "sender": sender,
            "title": sender,
            "body": body,
            "category": category,
            "urgency_score": urgency,
            "related_window": CONTEXTS[related][1],
            "related_relevance": relevance,
            "unrelated_window": CONTEXTS[unrelated][1],
            "unrelated_relevance": 1,
            "wording_feedback": "",
            "other_feedback": "",
        })
        for role, context_key, score in (
            ("related", related, relevance),
            ("unrelated", unrelated, 1),
        ):
            process, window = CONTEXTS[context_key]
            when = stamp + timedelta(minutes=index)
            samples.append(FilteringSample.model_validate({
                "notification": {
                    "id": f"msgdiv_{index:02d}_{role}",
                    "app_name": app,
                    "sender": sender,
                    "title": sender,
                    "body": body,
                    "timestamp": when.isoformat().replace("+00:00", "Z"),
                },
                "context": {
                    "active_process": process,
                    "window_title": window,
                    "last_updated": (when - timedelta(seconds=3)).isoformat().replace("+00:00", "Z"),
                    "duration_seconds": 90,
                    "recent_processes": [process],
                },
                "label": {
                    "urgency_score": urgency,
                    "relevance_score": score,
                    "category": category,
                    "ai_summary_reason": (
                        "현재 창과 알림 주제가 직접 연결됩니다."
                        if role == "related" else
                        "현재 창과 알림 주제가 다릅니다."
                    ),
                },
            }))
    return samples, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-output", type=Path, default=ROOT / "message_diversity_candidates.jsonl")
    parser.add_argument("--review-output", type=Path, default=ROOT / "message_diversity_review.csv")
    args = parser.parse_args()
    samples, review = build()
    for path in (args.samples_output, args.review_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.samples_output.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(sample.model_dump_json() + "\n")
    with args.review_output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps({
        "messages": len(review),
        "paired_candidates": len(samples),
        "apps": dict(Counter(row["app_name"] for row in review)),
        "categories": dict(Counter(row["category"] for row in review)),
        "status": "provisional wording review; not merged into training data",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
