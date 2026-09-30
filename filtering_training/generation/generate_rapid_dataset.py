"""Generate 3,000 Korean filtering candidates from authored scenario patterns.

This is deterministic composition, not an LLM translation. Inputs and labels are
provisional and belong in ignored outputs until review. No private file is read.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from filtering_training import TRAINING_ROOT

from filtering_training.datasets.prepare_dataset import DATASET_PATH, load_samples
from filtering_training.datasets.prepare_holdout import HOLDOUT_PATH, notification_key
from src.filtering.prompt import CATEGORIES, parse_model_output
from src.filtering.schema import FilteringSample

OUTPUT_PATH = TRAINING_ROOT / "outputs" / "candidates" / "rapid_korean_3000.jsonl"
BASE_TIME = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)

POOLS = {
    "services": ["주문 서비스", "결제 서비스", "인증 서비스", "검색 서비스", "알림 서비스",
                 "재고 서비스", "정산 서비스", "예약 서비스", "배송 서비스", "고객 관리 서비스"],
    "features": ["로그인 기능", "결제 화면", "검색 화면", "알림 설정", "주문 목록",
                 "재고 화면", "관리자 화면", "예약 화면", "배송 조회", "고객 대시보드"],
    "meetings": ["배포 점검 회의", "예산 검토 회의", "디자인 리뷰", "주간 계획 회의", "품질 점검 회의",
                 "고객 상담", "프로젝트 회고", "운영 공유 회의", "기획 검토 회의", "일정 조율 회의"],
    "accounts": ["회사 계정", "개인 계정", "메일 계정", "클라우드 계정", "개발 계정",
                 "쇼핑 계정", "업무 계정", "원격 접속 계정", "저장소 계정", "결제 계정"],
    "devices": ["노트북", "업무용 PC", "태블릿", "휴대전화", "외장 드라이브",
                "프린터", "공유기", "무선 이어폰", "보조 배터리", "원격 데스크톱"],
    "files": ["발표 자료", "예산표", "계약서 초안", "사진 폴더", "프로젝트 문서",
              "회의록", "설계 파일", "보고서", "백업 파일", "작업 목록"],
    "appointments": ["치과 진료", "건강 검진", "보험 상담", "차량 점검", "수리 방문",
                     "은행 상담", "서류 제출", "숙소 체크인", "비자 수령", "대학 등록"],
    "transactions": ["카드 결제", "계좌 이체", "정기 결제", "자동 이체", "온라인 주문",
                     "교통비 결제", "구독 결제", "보험료 납부", "예약금 결제", "환불 처리"],
    "deliveries": ["도서 배송", "생활용품 배송", "의류 배송", "전자기기 배송", "식료품 배송",
                   "선물 배송", "서류 배송", "가구 배송", "의약품 배송", "부품 배송"],
    "contacts": ["가족", "친구", "동생", "형제", "부모님", "이웃", "지인", "동료",
                 "보호자", "친척"],
    "content": ["사진 모음", "재생 목록", "읽을 기사", "게임 소식", "동영상",
                "오디오북", "주말 기록", "여행 사진", "독서 목록", "운동 기록"],
    "products": ["무선 이어폰", "노트북 거치대", "커피 원두", "운동화", "책",
                 "휴대전화 케이스", "모니터", "여행 가방", "키보드", "조명"],
    "computers": ["노트북", "업무용 PC", "태블릿", "휴대전화", "원격 데스크톱",
                  "파일 서버", "가정용 PC", "개발 PC", "공용 PC", "가상 머신"],
    "health": ["치과 진료", "건강 검진", "예방 접종", "안과 진료", "정형외과 진료",
               "물리 치료", "피부과 진료", "검사 결과 상담", "처방 상담", "정기 검진"],
    "insurance": ["보험 갱신", "보험금 청구", "보장 내역", "납입 증명", "계약 변경",
                  "실손 청구", "만기 안내", "보험 서류", "피보험자 정보", "청구 심사"],
    "documents": ["신분증 재발급", "여권 수령", "비자 수령", "등록 서류", "계약서 서명",
                  "증명서 발급", "면허증 갱신", "세금 서류", "학적 서류", "보증 서류"],
    "photos": ["가족 사진", "여행 사진", "주말 사진", "산책 사진", "작년 여행 사진",
               "생일 사진", "반려동물 사진", "풍경 사진", "모임 사진", "휴가 사진"],
    "music": ["집중 음악", "재즈 목록", "산책 음악", "운동 음악", "휴식 음악",
              "출근길 음악", "피아노 연주", "새 앨범", "좋아요 목록", "주말 재생 목록"],
    "videos": ["요리 영상", "여행 영상", "기술 강의", "운동 영상", "영화 리뷰",
               "게임 영상", "뉴스 영상", "음악 공연", "독서 영상", "디자인 강의"],
    "games": ["협동 게임", "주말 게임", "신규 게임", "보드게임", "퍼즐 게임",
              "전략 게임", "온라인 게임", "모바일 게임", "친구 게임", "게임 대회"],
    "books": ["추리 소설", "역사책", "에세이", "경제 도서", "과학 책",
              "여행기", "자기계발서", "소설 모음", "기술 서적", "오디오북"],
    "notes": ["주말 계획", "독서 메모", "요리 메모", "운동 계획", "여행 계획",
              "장보기 목록", "공부 메모", "집안일 목록", "영화 목록", "취미 기록"],
    "activities": ["주말 여행", "산책", "운동", "독서", "공부",
                   "요리", "게임", "친구 모임", "휴식", "영화 관람"],    "utilities": ["다운로드 목록", "클립보드 기록", "최근 파일", "메모 목록", "바탕화면",
                  "알림 센터", "파일 정렬", "검색 기록", "창 배치", "작업 표시줄"],
}

# A suffix changes the underlying time, impact, or action requirement, not just punctuation.
DETAILS = {
    "urgent": [("영향 범위를 확인하고 빠르게 조치해야 합니다.", 4),
               ("현재 진행 중인 업무가 중단돼 즉시 대응해야 합니다.", 5),
               ("오늘 마감 작업에 영향이 있어 빠른 확인이 필요합니다.", 4),
               ("후속 작업을 진행할 수 없어 즉시 조치해야 합니다.", 5)],
    "urgent_short": [("오늘 반영 전 수정 필요.", 4),
                     ("핵심 작업 중단. 즉시 복구 필요.", 5),
                     ("남은 작업이 막힘. 마감 전 확인 필요.", 4),
                     ("장애 지속. 담당자 바로 대응 필요.", 5)],
    "work": [("시간 될 때 확인해 주세요.", 2),
             ("다음 작업 전에 확인해 주세요.", 2),
             ("오늘 안에 확인해 주세요.", 3),
             ("오늘 마감 전까지 확인이 필요합니다.", 3)],
    "meeting": [("다음 주에 진행됩니다.", 2),
                ("내일 진행됩니다.", 3),
                ("30분 뒤 시작됩니다.", 3),
                ("5분 뒤 시작됩니다.", 4)],
    "security": [("활동 내역에서 확인할 수 있습니다.", 2),
                 ("오늘 안에 본인 활동인지 확인해 주세요.", 3),
                 ("평소와 다른 접근이 이어져 빠른 확인이 필요합니다.", 4),
                 ("본인 활동이 아니라면 즉시 계정을 보호해 주세요.", 5)],
    "system_problem": [("추가 작업에 영향이 생길 수 있습니다.", 2),
                       ("작업을 계속하려면 상태를 확인해 주세요.", 3),
                       ("현재 작업이 중단돼 빠른 확인이 필요합니다.", 4),
                       ("작업 결과가 손실될 수 있어 즉시 조치해 주세요.", 5)],
    "system_info": [("별도 조치는 필요하지 않습니다.", 1),
                    ("시간 될 때 상태를 확인할 수 있습니다.", 1),
                    ("오늘 중 설정을 확인할 수 있습니다.", 2),
                    ("작업을 마친 뒤 확인해 주세요.", 2)],
    "personal_action": [("다음 주까지 확인해 주세요.", 2),
                        ("내일까지 확인해 주세요.", 3),
                        ("오늘 안에 확인해 주세요.", 3),
                        ("30분 안에 확인해야 합니다.", 4)],
    "personal_risk": [("거래 내역을 확인할 수 있습니다.", 2),
                      ("오늘 안에 거래 내역을 확인해 주세요.", 3),
                      ("승인하지 않은 요청이라면 빠르게 확인해 주세요.", 4),
                      ("본인이 요청하지 않았다면 즉시 거래를 확인해 주세요.", 5)],
    "personal_message": [("시간 될 때 답장을 보내 주세요.", 1),
                         ("이번 주 안에 답장을 보내 주세요.", 2),
                         ("오늘 중에 답장을 보내 주세요.", 2),
                         ("지금 통화가 가능한지 알려 주세요.", 3)],
    "casual": [("나중에 확인해도 됩니다.", 1),
               ("새 내용은 앱에서 확인할 수 있습니다.", 1),
               ("시간 날 때 살펴보세요.", 1),
               ("이번 주 안에 확인할 수 있습니다.", 2)],
    "promotion": [("관심 있으면 상품 페이지에서 자세히 볼 수 있습니다.", 1),
                  ("이번 주 안에 혜택을 확인할 수 있습니다.", 1),
                  ("혜택은 오늘까지만 적용됩니다.", 1),
                  ("혜택은 이번 주말까지 적용됩니다.", 2)],
    "neutral": [("나중에 관련 화면에서 확인할 수 있습니다.", 1),
                ("알림을 열면 세부 내역을 볼 수 있습니다.", 1),
                ("필요할 때 다시 확인할 수 있습니다.", 2),
                ("관련 목록에서 세부 내용을 확인할 수 있습니다.", 2)],
}

# app, title, body, subject pool, detail group, domain. Each row is authored.
SCENARIOS = {
    "긴급 업무": [
        ("운영 알림", "{subject} 연결 장애", "{subject}에서 연결 오류가 발생했습니다.", "services", "urgent", "work"),
        ("PagerDuty", "{subject} 응답 지연", "{subject}의 응답 시간이 급격히 늘었습니다.", "services", "urgent", "work"),
        ("Slack", "{subject} 요청 실패", "{subject}에 들어온 요청이 정상 처리되지 않습니다.", "services", "urgent", "work"),
        ("GitHub Actions", "{subject} 배포 차단", "{subject} 배포 검증 실패.", "services", "urgent_short", "work"),
        ("Jira", "{subject} 인증 오류", "{subject}에 대한 인증 오류가 반복되고 있습니다.", "services", "urgent", "work"),
        ("운영 알림", "{subject} 데이터 조회 중단", "{subject} 조회 결과 누락.", "services", "urgent_short", "work"),
        ("Microsoft Teams", "{subject} 저장 오류", "{subject}에서 새 변경 사항을 저장하지 못했습니다.", "services", "urgent", "work"),
        ("PagerDuty", "{subject} 처리 대기열 정지", "{subject} 신규 요청 처리 안 됨.", "services", "urgent_short", "work"),
        ("PagerDuty", "{subject} 동기화 중단", "{subject}의 최신 데이터 동기화가 멈췄습니다.", "services", "urgent", "work"),
        ("Jira", "{subject} 권한 변경", "{subject} 담당자 접근 차단.", "services", "urgent_short", "work"),
    ],
    "일반 업무": [
        ("GitHub", "{subject} 코드 리뷰 요청", "{subject} 변경 사항을 검토해 달라는 요청이 도착했습니다.", "features", "work", "work"),
        ("Slack", "{subject} 작업 의견", "{subject} 구현 방식에 대한 의견이 등록됐습니다.", "features", "work", "work"),
        ("Jira", "{subject} 담당 작업 배정", "{subject} 개선 작업이 새로 배정됐습니다.", "features", "work", "work"),
        ("Notion", "{subject} 문서 수정", "{subject} 설명 문서가 업데이트됐습니다.", "features", "work", "work"),
        ("Microsoft Teams", "{subject} 시안 공유", "{subject}에 대한 새 화면 시안이 공유됐습니다.", "features", "work", "work"),
        ("GitHub", "{subject} 테스트 결과", "{subject} 관련 테스트 결과가 게시됐습니다.", "features", "work", "work"),
        ("Linear", "{subject} 진행 상태 변경", "{subject} 작업의 진행 상태가 변경됐습니다.", "features", "work", "work"),
        ("Outlook", "{subject} 검토 요청 메일", "{subject} 관련 검토 요청 메일이 도착했습니다.", "features", "work", "work"),
        ("Asana", "{subject} 작업 목록 갱신", "{subject}의 남은 작업 목록이 갱신됐습니다.", "features", "work", "work"),
        ("Google Drive", "{subject} 공유 파일 변경", "{subject} 관련 공유 파일에 변경 사항이 생겼습니다.", "features", "work", "work"),
    ],
    "일정/회의": [
        ("Outlook", "{subject} 일정 알림", "{subject} 일정이 등록돼 있습니다.", "meetings", "meeting", "schedule"),
        ("Microsoft Teams", "{subject} 초대", "{subject} 참석 초대가 도착했습니다.", "meetings", "meeting", "schedule"),
        ("Google Calendar", "{subject} 시작 안내", "{subject} 진행 시간이 안내됐습니다.", "meetings", "meeting", "schedule"),
        ("Zoom", "{subject} 참가 안내", "{subject}에 참여할 수 있습니다.", "meetings", "meeting", "schedule"),
        ("Outlook", "{subject} 자료 준비", "{subject}에서 사용할 자료를 확인해 주세요.", "meetings", "meeting", "schedule"),
        ("Microsoft Teams", "{subject} 참석 확인", "{subject} 참석 여부를 알려 주세요.", "meetings", "meeting", "schedule"),
        ("Google Calendar", "{subject} 시간 확인", "{subject} 진행 시간을 확인해 주세요.", "meetings", "meeting", "schedule"),
        ("Slack", "{subject} 발표 준비", "{subject} 발표 순서를 확인해 주세요.", "meetings", "meeting", "schedule"),
        ("Outlook", "{subject} 회의실 안내", "{subject}의 회의실 정보가 등록됐습니다.", "meetings", "meeting", "schedule"),
        ("Microsoft Teams", "{subject} 접속 정보", "{subject}의 온라인 접속 정보가 준비됐습니다.", "meetings", "meeting", "schedule"),
    ],
    "시스템/보안": [
        ("Windows 보안", "{subject} 로그인 확인", "{subject}에서 새 로그인 활동이 감지됐습니다.", "accounts", "security", "system"),
        ("Google Chrome", "{subject} 접근 알림", "{subject}에 평소와 다른 기기에서 접근했습니다.", "accounts", "security", "system"),
        ("Microsoft Authenticator", "{subject} 인증 요청", "{subject}의 새 인증 요청이 발생했습니다.", "accounts", "security", "system"),
        ("OneDrive", "{subject} 동기화 오류", "{subject}의 동기화가 정상적으로 끝나지 않았습니다.", "files", "system_problem", "system"),
        ("Dropbox", "{subject} 백업 오류", "{subject}의 백업이 완료되지 않았습니다.", "files", "system_problem", "system"),
        ("Windows", "{subject} 저장 공간 부족", "{subject} 관련 파일을 저장할 공간이 부족합니다.", "files", "system_problem", "system"),
        ("원격 접속", "{subject} 연결 해제", "{subject}의 연결이 끊어졌습니다.", "computers", "system_problem", "system"),
        ("Windows 보안", "{subject} 보안 검사 완료", "{subject}의 보안 검사가 끝났으며 위협은 발견되지 않았습니다.", "computers", "system_info", "system"),
        ("Windows 업데이트", "{subject} 업데이트 준비", "{subject}에 적용할 새 업데이트가 준비됐습니다.", "computers", "system_info", "system"),
        ("기기 설정", "{subject} 상태 변경", "{subject}의 설정 상태가 변경됐습니다.", "computers", "system_info", "system"),
    ],
    "개인 중요": [
        ("병원 앱", "{subject} 예약 확인", "{subject} 예약 정보에 확인 요청이 도착했습니다.", "health", "personal_action", "personal"),
        ("문자 메시지", "{subject} 일정 변경", "{subject} 예약 일정이 변경됐습니다.", "appointments", "personal_action", "personal"),
        ("은행 앱", "{subject} 승인 확인", "{subject} 요청을 확인해야 합니다.", "transactions", "personal_risk", "personal"),
        ("카드 앱", "{subject} 거래 확인", "{subject} 내역에 확인이 필요한 항목이 있습니다.", "transactions", "personal_risk", "personal"),
        ("배송 앱", "{subject} 수령 확인", "{subject}의 수령 정보를 확인해 주세요.", "deliveries", "personal_action", "personal"),
        ("정부24", "{subject} 확인 요청", "{subject} 관련 서류를 확인해 주세요.", "documents", "personal_action", "personal"),
        ("보험 앱", "{subject} 서류 확인", "{subject} 관련 서류를 확인해 주세요.", "insurance", "personal_action", "personal"),
        ("은행 앱", "{subject} 처리 보류", "{subject} 처리가 보류돼 확인이 필요합니다.", "transactions", "personal_risk", "personal"),
        ("병원 앱", "{subject} 결과 안내", "{subject} 관련 결과를 확인할 수 있습니다.", "health", "personal_action", "personal"),
        ("문자 메시지", "{subject} 본인 확인", "{subject} 요청에 대한 본인 확인이 필요합니다.", "transactions", "personal_risk", "personal"),
    ],
    "개인 일반": [
        ("KakaoTalk.exe", "{subject} 공유", "앨범에 새 사진이 공유됐다는 메시지가 도착했습니다.", "photos", "casual", "personal"),
        ("Google Photos", "{subject} 백업", "{subject}의 백업이 완료됐습니다.", "photos", "casual", "personal"),
        ("Spotify", "{subject} 추천", "{subject} 관련 음악이 추천됐습니다.", "music", "casual", "personal"),
        ("YouTube", "{subject} 새 항목", "{subject}와 관련된 영상이 추가됐습니다.", "videos", "casual", "personal"),
        ("Discord", "{subject} 이야기", "{subject}에 대한 새 대화가 있습니다.", "games", "casual", "personal"),
        ("오디오북 앱", "{subject} 이용 안내", "{subject} 콘텐츠를 다시 이용할 수 있습니다.", "books", "casual", "personal"),
        ("메모 앱", "{subject} 동기화", "{subject} 항목이 동기화됐습니다.", "notes", "casual", "personal"),
        ("사진 앱", "{subject} 추억 알림", "{subject}에 관한 지난 기록을 볼 수 있습니다.", "photos", "casual", "personal"),
        ("KakaoTalk.exe", "{subject} 근황", "{subject}에 관한 이야기를 나누자는 메시지가 도착했습니다.", "activities", "casual", "personal"),
        ("독서 앱", "{subject} 저장", "{subject} 항목을 나중에 볼 목록에 저장했습니다.", "books", "casual", "personal"),
    ],
    "광고/홍보": [
        ("쇼핑 앱", "{subject} 할인", "{subject} 할인 상품이 준비됐습니다.", "products", "promotion", "promotion"),
        ("이메일", "{subject} 추천", "{subject} 관련 추천 상품을 살펴보세요.", "products", "promotion", "promotion"),
        ("Google Chrome", "{subject} 특가", "{subject} 특가 상품이 등록됐습니다.", "products", "promotion", "promotion"),
        ("쇼핑 앱", "{subject} 행사", "{subject} 구매에 사용할 수 있는 행사가 열렸습니다.", "products", "promotion", "promotion"),
        ("쇼핑 앱", "{subject} 쿠폰", "{subject} 구매에 적용할 쿠폰이 도착했습니다.", "products", "promotion", "promotion"),
        ("KakaoTalk.exe", "{subject} 신상품", "{subject}의 관심 있으면 상품 페이지에서 자세히 볼 수 있습니다.", "products", "promotion", "promotion"),
        ("이메일", "{subject} 회원 혜택", "{subject} 구매를 위한 회원 혜택이 준비됐습니다.", "products", "promotion", "promotion"),
        ("Google Chrome", "{subject} 장바구니 안내", "{subject} 상품을 장바구니에서 다시 확인해 보세요.", "products", "promotion", "promotion"),
        ("쇼핑 앱", "{subject} 무료 배송", "{subject} 주문에 무료 배송 혜택을 적용할 수 있습니다.", "products", "promotion", "promotion"),
        ("이메일", "{subject} 기획전", "{subject} 관련 기획전이 시작됐습니다.", "products", "promotion", "promotion"),
    ],
    "기타": [
        ("파일 탐색기", "{subject} 정보 갱신", "{subject} 정보가 갱신됐습니다.", "utilities", "neutral", "misc"),
        ("Windows", "{subject} 상태 기록", "{subject}의 상태가 기록됐습니다.", "utilities", "neutral", "misc"),
        ("메모 앱", "{subject} 목록 정리", "{subject} 목록이 정리됐습니다.", "utilities", "neutral", "misc"),
        ("캡처 도구", "{subject} 기록 추가", "{subject} 관련 기록이 추가됐습니다.", "utilities", "neutral", "misc"),
        ("파일 탐색기", "{subject} 표시 변경", "{subject}의 표시 방식이 변경됐습니다.", "utilities", "neutral", "misc"),
        ("Windows", "{subject} 사용 팁", "{subject} 정리에 도움이 되는 기능을 사용할 수 있습니다.", "utilities", "neutral", "misc"),
        ("시계", "{subject} 알림 설정", "{subject}의 알림 설정이 저장됐습니다.", "utilities", "neutral", "misc"),
        ("계산기", "{subject} 최근 항목", "{subject}의 최근 기록을 확인할 수 있습니다.", "utilities", "neutral", "misc"),
        ("메모 앱", "{subject} 보관 완료", "{subject} 항목이 보관됐습니다.", "utilities", "neutral", "misc"),
        ("파일 탐색기", "{subject} 정렬 완료", "{subject} 항목이 정렬됐습니다.", "utilities", "neutral", "misc"),
    ],
}

CONTEXTS = {
    "work": (("Code.exe", "{subject} 작업 - Visual Studio Code", ["Code.exe", "chrome.exe", "WindowsTerminal.exe"]),
             ("chrome.exe", "다른 기능 작업 - Jira", ["chrome.exe", "Code.exe"]),
             ("EXCEL.EXE", "개인 가계부 - Excel", ["EXCEL.EXE"])),
    "schedule": (("OUTLOOK.EXE", "{subject} - Outlook", ["OUTLOOK.EXE", "ms-teams.exe", "chrome.exe"]),
                 ("OUTLOOK.EXE", "이번 주 일정 - Outlook", ["OUTLOOK.EXE", "chrome.exe"]),
                 ("Code.exe", "로그인 화면 작업 - Visual Studio Code", ["Code.exe"])),
    "system": (("SystemSettings.exe", "{subject} 상태 - Windows 설정", ["SystemSettings.exe", "explorer.exe", "chrome.exe"]),
               ("SystemSettings.exe", "Windows 설정 - 시스템", ["SystemSettings.exe", "explorer.exe"]),
               ("Figma.exe", "아이콘 시안 - Figma", ["Figma.exe"])),
    "personal": (("chrome.exe", "{subject} 정보 - Chrome", ["chrome.exe", "KakaoTalk.exe", "explorer.exe"]),
                 ("chrome.exe", "개인 일정 정리 - Chrome", ["chrome.exe", "KakaoTalk.exe"]),
                 ("Code.exe", "검색 코드 작업 - Visual Studio Code", ["Code.exe"])),
    "promotion": (("chrome.exe", "{subject} 상품 비교 - Chrome", ["chrome.exe", "explorer.exe", "KakaoTalk.exe"]),
                  ("chrome.exe", "구매 목록 - Chrome", ["chrome.exe", "explorer.exe"]),
                  ("Code.exe", "알림 코드 작업 - Visual Studio Code", ["Code.exe"])),
    "misc": (("explorer.exe", "{subject} - 파일 탐색기", ["explorer.exe", "chrome.exe", "WINWORD.EXE"]),
             ("explorer.exe", "문서 폴더 - 파일 탐색기", ["explorer.exe", "WINWORD.EXE"]),
             ("OUTLOOK.EXE", "다음 주 일정 - Outlook", ["OUTLOOK.EXE"])),
}

UTILITY_APPS = {
    "다운로드 목록": "Google Chrome", "클립보드 기록": "Windows",
    "최근 파일": "파일 탐색기", "메모 목록": "메모 앱",
    "바탕화면": "Windows", "알림 센터": "Windows",
    "파일 정렬": "파일 탐색기", "검색 기록": "Google Chrome",
    "창 배치": "Windows", "작업 표시줄": "Windows",
}
URGENCY_REASON = {
    1: "언제 확인해도 큰 문제가 없습니다",
    2: "나중에 확인해도 됩니다",
    3: "비교적 빠른 확인이 필요합니다",
    4: "현재 작업을 잠시 중단하고 확인할 가치가 있습니다",
    5: "즉시 확인하거나 대응해야 합니다",
}


def _rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _context(domain: str, subject: str, mode: int):
    if mode == 3:
        return "", "", [], 0, 1, "현재 작업 정보가 없어 연관성을 알 수 없으며"
    process, window, recent = CONTEXTS[domain][mode]
    if domain == "system" and mode == 0:
        if subject in POOLS["accounts"]:
            process, window, recent = "chrome.exe", f"{subject} 보안 설정 - Chrome", ["chrome.exe", "SystemSettings.exe", "explorer.exe"]
        elif subject in POOLS["files"]:
            process, window, recent = "explorer.exe", f"{subject} 폴더 - 파일 탐색기", ["explorer.exe", "OneDrive.exe", "chrome.exe"]
        else:
            window = window.format(subject=subject)
    elif domain == "misc" and mode == 0:
        app = UTILITY_APPS[subject]
        process = {"Google Chrome": "chrome.exe", "메모 앱": "Notepad.exe"}.get(app, "explorer.exe")
        window = f"{subject} - {app}"
        recent = [process, "explorer.exe", "chrome.exe"]
    else:
        window = window.format(subject=subject)
    if mode == 0:
        relevance = 5 if domain in {"work", "schedule", "system"} else 4
        relation = "현재 열어 둔 작업과 직접 관련이 있으며"
    elif mode == 1:
        relevance = 3 if domain in {"work", "schedule", "personal"} else (1 if domain == "misc" else 2)
        relation = "현재 작업 분야와 간접적으로 관련이 있으며"
    else:
        relevance = 1
        relation = "현재 작업과 무관하며"
    return process, window, recent, 25 + mode * 20, relevance, relation


def generate(output: Path, target: int = 3000,
             training: Path = DATASET_PATH, holdout: Path = HOLDOUT_PATH) -> dict:
    if target < 8 or target % len(CATEGORIES):
        raise ValueError("target must be at least eight and divisible by eight")
    if set(SCENARIOS) != set(CATEGORIES):
        raise ValueError("category definitions do not match schema")
    reserved = load_samples(training)
    if holdout.exists():
        reserved.extend(load_samples(holdout))
    reserved_ids = {item.notification.id for item in reserved}
    reserved_keys = {notification_key(item) for item in reserved}
    rows_by_category = {}
    per_category = target // len(CATEGORIES)
    for category in CATEGORIES:
        combinations = []
        for scenario_index, scenario in enumerate(SCENARIOS[category]):
            app, title_template, body_template, pool_name, detail_group, domain = scenario
            if len(POOLS[pool_name]) != 10 or len(DETAILS[detail_group]) != 4:
                raise ValueError("each scenario must have ten subjects and four details")
            for subject_index, subject in enumerate(POOLS[pool_name]):
                for detail_index, (detail, urgency) in enumerate(DETAILS[detail_group]):
                    combinations.append((scenario_index, subject_index, detail_index,
                                         app, title_template.format(subject=subject),
                                         body_template.format(subject=subject) + " " + detail,
                                         subject, domain, urgency))
        if len(combinations) < per_category:
            raise ValueError(f"not enough candidates for {category}")
        rows_by_category[category] = sorted(
            combinations, key=lambda value: _rank(f"{category}:{value[0]}:{value[1]}:{value[2]}")
        )[:per_category]
    samples = []
    lineage = []
    for category in CATEGORIES:
        for case in rows_by_category[category]:
            scenario_index, subject_index, detail_index, app, title, body, subject, domain, urgency = case
            if category == "기타":
                app = UTILITY_APPS[subject]
            index = len(samples) + 1
            mode = int(_rank(f"context:{category}:{scenario_index}:{subject_index}:{detail_index}")[:8], 16) % 4
            process, window, recent, duration, relevance, relation = _context(domain, subject, mode)
            timestamp = BASE_TIME + timedelta(seconds=index * 23)
            item = FilteringSample.model_validate({
                "notification": {
                    "id": f"rapid_{index:04d}", "app_name": app, "sender": "",
                    "title": title, "body": body,
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                },
                "context": {
                    "active_process": process, "window_title": window,
                    "last_updated": (timestamp - timedelta(seconds=3)).isoformat().replace("+00:00", "Z"),
                    "duration_seconds": duration, "recent_processes": recent,
                },
                "label": {
                    "urgency_score": urgency, "relevance_score": relevance, "category": category,
                    "ai_summary_reason": f"{title} 알림은 {relation} {URGENCY_REASON[urgency]}.",
                },
            })
            parse_model_output(item.label.model_dump_json())
            samples.append(item)
            lineage.append({"id": item.notification.id, "scenario": f"{category}:{scenario_index}",
                            "subject_index": subject_index, "detail_index": detail_index,
                            "context_mode": mode})
    ids = [item.notification.id for item in samples]
    keys = [notification_key(item) for item in samples]
    text_pairs = [(item.notification.title, item.notification.body) for item in samples]
    if (len(ids) != len(set(ids)) or set(ids) & reserved_ids
            or len(keys) != len(set(keys)) or set(keys) & reserved_keys
            or len(text_pairs) != len(set(text_pairs))):
        raise ValueError("duplicate or reserved notification detected")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in samples:
            stream.write(item.model_dump_json() + "\n")
    output.with_suffix(".lineage.json").write_text(json.dumps({
        "status": "provisional deterministic candidates; independently review before training",
        "method": "authored Korean scenario patterns x ten subject choices x four meaningful detail variants",
        "source_reference": "public NotifAI scenario themes and private real-sample styles; neither is read or copied",
        "items": lineage,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "count": len(samples), "distinct_notification_texts": len(text_pairs),
        "scenario_families": sum(len(group) for group in SCENARIOS.values()),
        "category_counts": dict(Counter(item.label.category for item in samples)),
        "status": "candidate only; labels and naturalness require review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--target", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.target), ensure_ascii=False))


if __name__ == "__main__":
    main()