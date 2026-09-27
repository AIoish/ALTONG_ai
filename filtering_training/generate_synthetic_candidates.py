"""Build a reviewable synthetic training batch from hand-written notification cases.

The private real-notification sample is never read or copied by this generator.
Generated JSONL belongs under the ignored outputs directory until labels are reviewed.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from filtering_training.prepare_dataset import DATASET_PATH, load_samples
from filtering_training.prepare_holdout import HOLDOUT_PATH, notification_key
from src.filtering.prompt import CATEGORIES, parse_model_output
from src.filtering.schema import FilteringSample


OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "candidates" / "synthetic_batch_01.jsonl"
BASE_TIME = datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)

# category, urgency, app, sender, title, body, task topic. All people/projects are fictional.
CASES = [
    ("긴급 업무", 5, "PagerDuty", "운영 알림", "주문 API 응답 중단", "주문 요청이 10분째 실패합니다. 담당자 확인이 필요합니다.", "code"),
    ("긴급 업무", 5, "GitHub", "", "배포 파이프라인 실패", "오늘 배포할 변경 사항이 검증 단계에서 멈췄습니다.", "code"),
    ("긴급 업무", 4, "Antigravity IDE", "", "인증 로직 충돌", "현재 수정한 인증 모듈에 충돌이 있어 병합 전 확인이 필요합니다.", "code"),
    ("긴급 업무", 4, "KakaoTalk.exe", "프로젝트 동료", "발표 자료 수정 요청", "10분 뒤 시작하는 발표의 첫 장 숫자를 다시 확인해 주세요.", "document"),
    ("긴급 업무", 5, "Slack", "운영 채널", "결제 승인 장애", "결제 승인 요청이 모두 실패하고 있습니다. 즉시 상태를 확인해 주세요.", "finance"),
    ("긴급 업무", 4, "Microsoft Teams", "검토 담당", "계약서 오류 발견", "오늘 발송할 계약서의 금액 항목이 달라 수정이 필요합니다.", "document"),
    ("긴급 업무", 5, "Jira", "", "고객 데이터 동기화 중단", "동기화 작업이 멈춰 신규 데이터가 반영되지 않고 있습니다.", "code"),
    ("긴급 업무", 4, "Outlook", "업무 담당", "오늘 제출 파일 누락", "제출 마감 전에 첨부 파일을 다시 올려 주세요.", "document"),

    ("일반 업무", 2, "GitHub", "", "코드 리뷰 요청", "새 검색 화면 변경 사항에 대한 리뷰를 부탁드립니다.", "code"),
    ("일반 업무", 1, "Antigravity IDE", "", "확장 프로그램 업데이트", "편집기 확장 프로그램의 새 버전을 설치할 수 있습니다.", "code"),
    ("일반 업무", 2, "Slack", "동료", "문서 의견 남김", "초안의 표 제목을 수정하면 좋겠다는 의견을 남겼습니다.", "document"),
    ("일반 업무", 3, "Jira", "", "작업 담당자 지정", "다음 주 진행할 검색 기능 개선 작업이 배정되었습니다.", "code"),
    ("일반 업무", 2, "Microsoft Teams", "동료", "디자인 시안 공유", "새 대시보드 화면의 시안을 공유했습니다. 편할 때 확인해 주세요.", "design"),
    ("일반 업무", 2, "Notion", "", "주간 보고서 초안", "팀 보고서 초안이 등록되었습니다. 내일까지 의견을 남겨 주세요.", "document"),
    ("일반 업무", 2, "Google Drive", "", "스프레드시트 댓글", "예산표의 한 항목에 확인 요청 댓글이 달렸습니다.", "finance"),
    ("일반 업무", 1, "KakaoTalk.exe", "동료", "지난 회의 메모", "회의 메모 정리해서 올려뒀어요. 나중에 확인해 주세요.", "document"),

    ("일정/회의", 4, "Google Calendar", "", "회의 5분 전", "곧 시작하는 서비스 점검 회의에 참여해 주세요.", "calendar"),
    ("일정/회의", 3, "Outlook", "", "오늘 오후 면담", "오후 3시에 예정된 면담 일정을 알려드립니다.", "calendar"),
    ("일정/회의", 2, "Microsoft Teams", "", "다음 주 회의 초대", "다음 주 화요일 기획 회의 초대가 도착했습니다.", "calendar"),
    ("일정/회의", 4, "Zoom", "", "회의실 입장 요청", "진행 중인 발표 회의에 참가자 입장 승인이 필요합니다.", "calendar"),
    ("일정/회의", 3, "Google Chrome", "", "예약 시간 변경", "내일 예정된 온라인 상담 시간이 한 시간 늦춰졌습니다.", "calendar"),
    ("일정/회의", 2, "KakaoTalk.exe", "동료", "스터디 일정 확인", "다음 주 스터디 시간을 정하려고 합니다. 가능할 때 답해주세요.", "calendar"),
    ("일정/회의", 4, "Slack", "진행자", "발표 순서 변경", "지금 진행 중인 회의의 발표 순서가 앞당겨졌습니다.", "calendar"),
    ("일정/회의", 1, "Notion", "", "지난 일정 기록", "지난달 회의 기록이 보관함으로 이동했습니다.", "calendar"),

    ("시스템/보안", 5, "Windows 보안", "", "계정 로그인 차단", "알 수 없는 기기에서 반복 로그인 시도가 감지되어 계정을 보호했습니다.", "system"),
    ("시스템/보안", 4, "Google Chrome", "", "비밀번호 변경 필요", "저장된 계정 비밀번호가 노출 목록과 일치합니다. 변경을 권장합니다.", "system"),
    ("시스템/보안", 4, "전원 및 배터리", "", "배터리 잔량 매우 낮음", "배터리 잔량이 5%입니다. 전원을 연결해 작업을 저장하세요.", "system"),
    ("시스템/보안", 3, "무선", "", "네트워크 연결 끊김", "현재 Wi-Fi 연결이 해제되었습니다. 필요한 경우 다시 연결하세요.", "system"),
    ("시스템/보안", 2, "Windows 업데이트", "", "업데이트 설치 가능", "보안 업데이트를 설치할 수 있습니다. 작업을 마친 뒤 재시작하세요.", "system"),
    ("시스템/보안", 2, "캡처 도구", "", "캡처 저장됨", "방금 캡처한 이미지가 클립보드에 복사되었습니다.", "system"),
    ("시스템/보안", 3, "OneDrive", "", "동기화 일시 중지", "저장 공간 부족으로 파일 동기화가 잠시 중단되었습니다.", "system"),
    ("시스템/보안", 2, "집중", "", "집중 세션 종료", "설정한 집중 시간이 끝났습니다.", "system"),

    ("개인 중요", 5, "KakaoTalk.exe", "가족", "응급실 이동 중", "가족이 다쳐 응급실로 이동 중입니다. 지금 연락해 주세요.", "personal"),
    ("개인 중요", 4, "문자 메시지", "배송 기사", "현관 앞 확인 요청", "오늘 도착한 파손 의심 물품을 인계하기 전에 확인이 필요합니다.", "personal"),
    ("개인 중요", 4, "은행 앱", "", "승인하지 않은 결제", "모르는 결제 승인 요청이 발생했습니다. 즉시 거래 내역을 확인하세요.", "personal"),
    ("개인 중요", 3, "병원 앱", "", "검사 결과 확인", "예약한 검사 결과가 도착했습니다. 내용을 확인해 주세요.", "personal"),
    ("개인 중요", 4, "KakaoTalk.exe", "가족", "약 복용 시간", "오늘 처방약을 아직 안 드셨으면 지금 확인해 주세요.", "personal"),
    ("개인 중요", 3, "이메일", "", "보험 갱신 안내", "이번 주 만료되는 보험의 갱신 내용을 확인할 수 있습니다.", "personal"),
    ("개인 중요", 4, "문자 메시지", "관리실", "누수 확인 요청", "집 현관 앞에서 누수가 발견되어 지금 확인이 필요합니다.", "personal"),
    ("개인 중요", 3, "대학 포털", "", "등록 기한 안내", "이번 주까지 제출해야 하는 등록 서류가 남아 있습니다.", "personal"),

    ("개인 일반", 1, "KakaoTalk.exe", "친구", "주말 사진", "지난 주말에 찍은 사진 올려뒀어. 시간 날 때 봐.", "personal"),
    ("개인 일반", 2, "문자 메시지", "", "택배 배송 완료", "주문한 생활용품이 문 앞에 배송되었습니다.", "personal"),
    ("개인 일반", 1, "Spotify", "", "새 재생 목록", "즐겨 듣는 장르의 새 재생 목록이 추가됐습니다.", "personal"),
    ("개인 일반", 2, "Google Photos", "", "사진 백업 완료", "오늘 찍은 사진의 백업이 완료됐습니다.", "personal"),
    ("개인 일반", 1, "Discord", "친구", "게임 접속", "저녁에 같이 게임할 사람 구하고 있어.", "personal"),
    ("개인 일반", 2, "KakaoTalk.exe", "친구", "저녁 메뉴", "오늘 저녁엔 뭐 먹을지 나중에 알려줘.", "personal"),
    ("개인 일반", 1, "YouTube", "", "새 영상 알림", "구독 중인 채널의 새 영상이 올라왔습니다.", "personal"),
    ("개인 일반", 2, "Google Chrome", "", "읽던 기사 저장", "나중에 읽기 목록에 새 기사가 추가됐습니다.", "personal"),

    ("광고/홍보", 1, "Google Chrome", "", "오늘만 할인", "선택한 상품을 오늘 할인 가격에 구매할 수 있습니다.", "shopping"),
    ("광고/홍보", 1, "KakaoTalk.exe", "쇼핑몰", "신상품 안내", "새 계절 의류가 출시됐습니다. 상품을 둘러보세요.", "shopping"),
    ("광고/홍보", 1, "이메일", "행사 운영", "웨비나 초대", "새 기능 소개 웨비나 참가 신청을 받고 있습니다.", "shopping"),
    ("광고/홍보", 2, "쇼핑 앱", "", "쿠폰 곧 만료", "보유한 할인 쿠폰이 오늘 밤 만료됩니다.", "shopping"),
    ("광고/홍보", 1, "Slack", "서비스 운영", "유료 요금제 안내", "추가 저장 공간이 포함된 요금제를 살펴보세요.", "shopping"),
    ("광고/홍보", 1, "배달 앱", "", "신규 메뉴 행사", "이번 주 신메뉴 할인 행사가 시작됐습니다.", "shopping"),
    ("광고/홍보", 1, "Google Chrome", "", "여행 상품 추천", "최근 살펴본 여행지의 특가 상품이 준비됐습니다.", "shopping"),
    ("광고/홍보", 1, "Microsoft Teams", "서비스 소식", "부가 기능 체험", "유료 부가 기능을 무료로 체험해 보세요.", "shopping"),

    ("기타", 1, "캡처 도구", "", "클립보드 기록", "", "general"),
    ("기타", 1, "Google Chrome", "", "다운로드 기록 정리", "지난달 다운로드 목록을 정리할 수 있습니다.", "general"),
    ("기타", 2, "파일 탐색기", "", "파일 이름 변경됨", "선택한 파일의 이름이 변경되었습니다.", "general"),
    ("기타", 1, "집중", "", "짧은 휴식 안내", "잠시 화면에서 눈을 떼고 쉬어 보세요.", "general"),
    ("기타", 2, "메모 앱", "", "새 메모 동기화", "다른 기기에서 작성한 메모 한 개가 추가됐습니다.", "general"),
    ("기타", 1, "Antigravity IDE", "", "최근 파일 목록 갱신", "최근 연 파일 목록이 업데이트됐습니다.", "general"),
    ("기타", 2, "무선", "", "기기 검색 완료", "연결할 수 있는 주변 기기를 찾았습니다.", "general"),
    ("기타", 1, "Windows", "", "팁 알림", "새 창 정렬 기능을 사용해 보세요.", "general"),
]

CONTEXTS = {
    "code": ("Code.exe", "order_api.py - 가상 프로젝트 - Visual Studio Code", ["Code.exe", "chrome.exe", "WindowsTerminal.exe"]),
    "document": ("WINWORD.EXE", "발표 초안 - Word", ["WINWORD.EXE", "chrome.exe"]),
    "calendar": ("OUTLOOK.EXE", "이번 주 일정 - Outlook", ["OUTLOOK.EXE", "ms-teams.exe"]),
    "finance": ("EXCEL.EXE", "가상 예산표 - Excel", ["EXCEL.EXE", "chrome.exe"]),
    "design": ("Figma.exe", "대시보드 시안 - Figma", ["Figma.exe", "chrome.exe"]),
    "system": ("SystemSettings.exe", "Windows 설정 - 시스템", ["SystemSettings.exe", "explorer.exe"]),
    "personal": ("chrome.exe", "개인 일정 정리 - Chrome", ["chrome.exe", "KakaoTalk.exe"]),
    "shopping": ("chrome.exe", "쇼핑 목록 - Chrome", ["chrome.exe", "explorer.exe"]),
    "general": ("explorer.exe", "다운로드 - 파일 탐색기", ["explorer.exe"]),
}

# A few cases have an explicitly matching task; these exercise relevance 5 even
# when the notification itself is not urgent.
MATCHED_TASKS = {
    "주문 API 응답 중단": ("Code.exe", "order_api.py - 가상 프로젝트 - Visual Studio Code", ["Code.exe", "WindowsTerminal.exe"], 5),
    "코드 리뷰 요청": ("Code.exe", "search_view.tsx - 가상 프로젝트 - Visual Studio Code", ["Code.exe", "chrome.exe"], 5),
    "회의 5분 전": ("OUTLOOK.EXE", "서비스 점검 회의 - Outlook", ["OUTLOOK.EXE", "ms-teams.exe"], 5),
    "배터리 잔량 매우 낮음": ("SystemSettings.exe", "전원 및 배터리 - Windows 설정", ["SystemSettings.exe"], 5),
    "다운로드 기록 정리": ("explorer.exe", "다운로드 - 파일 탐색기", ["explorer.exe"], 4),
}

def make_sample(index: int, variant: int, case: tuple) -> FilteringSample:
    category, urgency, app, sender, title, body, topic = case
    stamp = BASE_TIME + timedelta(minutes=index * 4)
    if variant == 0:
        process, window, recent = CONTEXTS[topic]
        relevance = {"shopping": 4, "design": 4, "code": 3, "finance": 3,
                     "calendar": 3, "document": 2, "system": 2,
                     "personal": 2, "general": 2}[topic]
        if title in MATCHED_TASKS:
            process, window, recent, relevance = MATCHED_TASKS[title]
        relation = "현재 작업 분야와 관련이 있지만" if relevance >= 3 else "현재 작업과의 직접적인 관련성은 낮지만"
    elif variant == 1:
        other_topic = "document" if topic == "code" else "code"
        process, window, recent = CONTEXTS[other_topic]
        relevance = 1
        relation = "현재 작업과 직접 관련은 없지만"
    else:
        process, window, recent = "", "", []
        relevance = 1
        relation = "현재 작업 정보가 없어 관련성은 알 수 없지만"
    if urgency >= 4:
        reason = f"{title} 알림은 {relation} 빠른 확인이나 대응이 필요합니다."
    elif urgency == 3:
        reason = f"{title} 알림은 {relation} 내용을 비교적 빠르게 확인할 필요가 있습니다."
    else:
        reason = f"{title} 알림은 {relation} 나중에 확인해도 됩니다."
    return FilteringSample.model_validate({
        "notification": {
            "id": f"candidate_{index + 1:03d}_{variant + 1}",
            "app_name": app, "sender": sender, "title": title, "body": body,
            "timestamp": stamp.isoformat().replace("+00:00", "Z"),
        },
        "context": {
            "active_process": process, "window_title": window,
            "last_updated": (stamp - timedelta(seconds=4)).isoformat().replace("+00:00", "Z"),
            "duration_seconds": 0 if variant == 2 else 30 + (index % 6) * 15,
            "recent_processes": recent,
        },
        "label": {
            "urgency_score": urgency, "relevance_score": relevance,
            "category": category, "ai_summary_reason": reason,
        },
    })


def generate(output: Path, training: Path = DATASET_PATH, holdout: Path = HOLDOUT_PATH) -> dict:
    if len(CASES) != 64 or Counter(case[0] for case in CASES) != {name: 8 for name in CATEGORIES}:
        raise ValueError("the first batch needs eight distinct cases per category")
    existing = load_samples(training)
    if holdout.exists():
        existing.extend(load_samples(holdout))
    reserved_ids = {item.notification.id for item in existing}
    reserved_text = {notification_key(item) for item in existing}
    samples = [make_sample(index, variant, case)
               for index, case in enumerate(CASES) for variant in range(3)]
    ids = [item.notification.id for item in samples]
    if len(ids) != len(set(ids)) or set(ids) & reserved_ids:
        raise ValueError("candidate IDs are duplicated or overlap existing data")
    texts = {notification_key(item) for item in samples}
    if len(texts) != len(CASES) or texts & reserved_text:
        raise ValueError("candidate notification text overlaps existing data")
    for item in samples:
        parse_model_output(json.dumps(item.label.model_dump(), ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for item in samples:
            target.write(item.model_dump_json(exclude_none=True) + "\n")
    return {
        "candidate_count": len(samples),
        "distinct_notification_texts": len(texts),
        "category_counts": dict(Counter(item.label.category for item in samples)),
        "status": "candidate only; labels require human review before training",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
