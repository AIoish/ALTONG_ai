"""Expand the approved filtering seed with independently authored scenarios.

Outputs stay provisional. Eight ambiguous cases are selected for human review;
no training or external upload is performed here.
"""

from filtering_training.common.paths import OUTPUTS_ROOT

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.filtering.policy import POLICY_VERSION, decision_label
from src.filtering.prompt import CATEGORIES, SYSTEM_PROMPT, build_messages, parse_model_output
from src.filtering.schema import FilteringSample


ROOT = OUTPUTS_ROOT
CONTEXTS = {
    "deploy": ("WindowsTerminal.exe", "고객 설치 패키지 배포 - Terminal"),
    "demo": ("Code.exe", "16시 고객 시연 오류 수정 - Visual Studio Code"),
    "records": ("Code.exe", "고객 데이터 복구 - Visual Studio Code"),
    "contract": ("EXCEL.EXE", "납품 계약 금액 확인 - Excel"),
    "server": ("WindowsTerminal.exe", "현장 서버 복구 - Terminal"),
    "release": ("chrome.exe", "오늘 릴리스 승인 - Chrome"),
    "dbnotes": ("Notion.exe", "DB 인덱스 비교 자료 정리 - Notion"),
    "notify": ("Code.exe", "알림 기능 구현 - Visual Studio Code"),
    "spec": ("Notion.exe", "검색 기능 요구사항 정리 - Notion"),
    "slides": ("POWERPNT.EXE", "발표 자료 표 수정 - PowerPoint"),
    "build": ("chrome.exe", "search-ui 브랜치 CI 결과 확인 - GitHub"),
    "names": ("Code.exe", "결제 검증 함수 작성 - Visual Studio Code"),
    "interview": ("chrome.exe", "온라인 면접 참가 준비 - Chrome"),
    "workshop": ("OUTLOOK.EXE", "다음 달 워크숍 일정 조율 - Outlook"),
    "zoom": ("Zoom.exe", "고객 상담 회의 입장 - Zoom"),
    "coffee": ("OUTLOOK.EXE", "팀 커피 모임 일정 정리 - Outlook"),
    "counsel": ("chrome.exe", "내일 면담 일정 확인 - Chrome"),
    "review": ("ms-teams.exe", "설계 리뷰 회의 입장 - Teams"),
    "network": ("SystemSettings.exe", "Wi-Fi 연결 복구 - Windows 설정"),
    "signin": ("chrome.exe", "회사 계정 로그인 - Chrome"),
    "password": ("chrome.exe", "비밀번호 유출 점검 - Chrome"),
    "sync": ("OneDrive.exe", "보고서 파일 동기화 문제 확인 - OneDrive"),
    "update": ("SystemSettings.exe", "Windows 업데이트 확인 - Windows 설정"),
    "battery": ("SystemSettings.exe", "배터리 사용량 확인 - Windows 설정"),
    "hospitalpay": ("chrome.exe", "가족 병원비 납부 - Chrome"),
    "bank": ("chrome.exe", "카드 이상거래 확인 - Chrome"),
    "application": ("chrome.exe", "지원 서류 제출 - Chrome"),
    "leak": ("chrome.exe", "집 누수 수리 접수 - Chrome"),
    "appointment": ("OUTLOOK.EXE", "내일 진료 예약 일정 - Outlook"),
    "insurance": ("chrome.exe", "보험 갱신 조건 비교 - Chrome"),
    "meal": ("ONENOTE.EXE", "내일 저녁 메뉴와 장보기 - OneNote"),
    "game": ("chrome.exe", "협동 게임 보스 공략 - Chrome"),
    "lunch": ("KakaoTalk.exe", "오늘 점심 메뉴 정하기 - KakaoTalk"),
    "music": ("Spotify.exe", "집중 음악 재생목록 비교 - Spotify"),
    "film": ("chrome.exe", "주말에 볼 영화 고르기 - Chrome"),
    "photo": ("Photos.exe", "지난 주말 여행 사진 편집 - Photos"),
    "ssd": ("chrome.exe", "SSD 상품 가격 비교 - Chrome"),
    "shoes": ("chrome.exe", "운동화 상품 비교 - Chrome"),
    "storage": ("chrome.exe", "팀 저장 공간 요금제 비교 - Chrome"),
    "charger": ("chrome.exe", "노트북 충전기 구매 비교 - Chrome"),
    "clip": ("ONENOTE.EXE", "인용문을 메모에 붙이기 - OneNote"),
    "archive": ("explorer.exe", "발표 파일 압축하기 - 파일 탐색기"),
    "download": ("msedge.exe", "학습 자료 다운로드 확인 - Microsoft Edge"),
    "drawing": ("krita.exe", "캐릭터 스케치 - Krita"),
    "timer": ("Clock.exe", "공부용 타이머 설정 - 시계"),
    "print": ("explorer.exe", "강의자료 인쇄 대기열 - 파일 탐색기"),
}

# Six new messages/category. Each tuple contains app, sender, displayed title,
# body, urgency, primary context, relevance, urgency evidence, relevance evidence.
SCENARIOS = {
    "긴급 업무": [
        ("KakaoTalk.exe", "지현", "지현", "배포 인증서 만료됐어. 고객 설치가 전부 막히는데 재서명 좀 부탁해", 5, "deploy", 5, "고객 설치 실패가 계속되고 있어 즉시 대응이 필요합니다", "현재 고객 설치 패키지 배포 문제와 직접 연결됩니다"),
        ("Slack", "서연", "#고객시연", "오늘 16시 시연 화면이 실행 안 됩니다. 시작 전에 수정 담당자 확인 부탁드려요", 4, "demo", 5, "오늘 고객 시연을 진행하지 못하는 문제로 확인이 필요합니다", "현재 고객 시연 오류를 수정하는 작업과 연결됩니다"),
        ("Microsoft Teams", "운영 담당", "운영팀", "고객 데이터가 잘못 덮어써지고 있습니다. 지금 쓰기 중지 판단 부탁드립니다", 5, "records", 5, "고객 데이터 손상이 진행 중이라 즉시 대응이 필요합니다", "현재 고객 데이터 복구에 필요한 정보입니다"),
        ("Jira", "검토 담당", "납품 계약 검토", "오늘 발송할 계약서의 금액이 견적과 다릅니다. 발송 전 수정해 주세요", 4, "contract", 5, "잘못된 금액의 계약서가 발송되기 전 확인이 필요합니다", "현재 확인 중인 납품 계약 금액에 관한 요청입니다"),
        ("Outlook", "현장 담당", "현장 서버 중단", "현장 서버가 꺼져 업무 접수가 모두 멈췄습니다. 복구 담당자 연락 바랍니다", 5, "server", 5, "업무 접수가 중단된 상태라 즉시 복구 대응이 필요합니다", "현재 현장 서버를 복구하는 작업과 직접 연결됩니다"),
        ("Discord", "배포 담당", "#release", "릴리스 10분 남았는데 승인 하나 비었어요. 승인 없으면 오늘 배포 못 합니다", 4, "release", 5, "임박한 릴리스가 승인 대기로 막혀 확인이 필요합니다", "현재 진행 중인 오늘 릴리스 승인과 연결됩니다"),
    ],
    "일반 업무": [
        ("Slack", "도윤", "#db-study", "DB 인덱스 비교 자료 정리했어요. 다음 주에 시간 될 때 의견 부탁드립니다", 2, "dbnotes", 5, "다음 주에 확인해도 되는 자료 검토 요청입니다", "현재 정리 중인 DB 인덱스 비교 자료와 직접 연결됩니다"),
        ("KakaoTalk.exe", "준호", "준호", "알림 기능 README 띄어쓰기만 고쳤어ㅋㅋ 코드 바뀐 건 없음", 1, "notify", 3, "문서의 띄어쓰기 수정이고 기능 변경이나 행동 요청이 없습니다", "같은 알림 기능이지만 현재 구현에 주는 도움은 불분명합니다"),
        ("Notion", "문서 댓글", "검색 기능 요구사항", "이번 주 안에 검색 필터 조건에 의견 남겨 주세요. 아직 구현 일정은 정하지 않았습니다", 2, "spec", 4, "이번 주에 의견을 남길 수 있는 요구사항 논의입니다", "현재 검색 기능 요구사항을 정리하는 데 도움이 됩니다"),
        ("Google Drive", "서연", "발표 자료에 댓글이 달렸습니다", "3페이지 표의 합계가 다른 것 같아요. 오늘 중으로 숫자 확인 부탁드립니다", 3, "slides", 5, "오늘 중 숫자 확인 요청이고 즉각적인 장애는 명시되지 않았습니다", "현재 수정 중인 발표 자료 표의 숫자와 직접 연결됩니다"),
        ("GitHub", "CI", "search-ui · checks passed", "Build succeeded. 24 tests passed on search-ui.", 1, "build", 5, "빌드와 테스트 성공을 알리는 완료 안내입니다", "현재 확인 중인 search-ui 브랜치 CI 결과입니다"),
        ("Microsoft Teams", "하린", "개발팀", "급함ㅋㅋ 다음 프로젝트 이름 추천 좀. 나는 감자 한 표", 1, "names", 2, "농담을 섞은 이름 추천으로 긴급한 피해나 마감이 없습니다", "개발팀 대화지만 현재 결제 검증 함수 작성과 거의 무관합니다"),
    ],
    "일정/회의": [
        ("KakaoTalk.exe", "채용 담당", "채용 담당", "면접 10분 뒤에 시작합니다. 참가 링크를 새로 보냈으니 확인해 주세요", 4, "interview", 5, "면접 시작이 임박했고 참가 링크 확인이 필요합니다", "현재 온라인 면접 참가 준비에 필요한 정보입니다"),
        ("Microsoft Teams", "행사 담당", "워크숍 일정", "다음 달 워크숍 날짜 후보를 올렸습니다. 다음 주까지 가능한 날 표시해 주세요", 2, "workshop", 4, "다음 주까지 응답할 수 있는 다음 달 일정 조율입니다", "현재 다음 달 워크숍 일정을 정하는 데 도움이 됩니다"),
        ("Zoom", "상담 진행자", "대기실 입장 안내", "고객 상담이 5분 뒤 시작됩니다. 대기실에 입장해 주세요", 4, "zoom", 5, "5분 뒤 시작하는 상담의 입장 요청입니다", "현재 입장 중인 고객 상담 회의에 관한 안내입니다"),
        ("Slack", "민서", "#팀잡담", "내일 오후에 커피 마실 사람? 참석은 자유고 메뉴만 미리 알려줘요", 2, "coffee", 4, "내일의 선택적인 모임으로 나중에 답할 수 있습니다", "현재 정리 중인 팀 커피 모임 일정에 도움이 됩니다"),
        ("Outlook", "면담 담당", "내일 면담 시간 변경", "내일 면담이 14시에서 15시로 변경되었습니다. 참석 전에 확인 부탁드립니다", 3, "counsel", 5, "내일 참석할 시간 변경으로 비교적 빠른 확인이 필요합니다", "현재 확인 중인 내일 면담 일정과 직접 연결됩니다"),
        ("KakaoTalk.exe", "진행 담당", "진행 담당", "5분 뒤 설계 리뷰 취소됐어요. 회의실로 안 오셔도 됩니다", 4, "review", 5, "곧 시작할 회의가 취소되어 바로 일정 변경을 확인해야 합니다", "현재 참가 준비 중인 설계 리뷰 회의에 관한 변경입니다"),
    ],
    "시스템/보안": [
        ("Windows", "", "네트워크 연결 끊김", "Wi-Fi 연결이 해제되었습니다. 진행 중인 온라인 통화가 중단되었습니다.", 4, "network", 5, "진행 중인 통화가 중단되어 연결 상태 확인이 필요합니다", "현재 Wi-Fi 연결을 복구하는 작업과 직접 연결됩니다"),
        ("Microsoft Authenticator", "", "로그인 승인 요청", "회사 계정 로그인을 승인하시겠습니까? 본인이 요청하지 않았다면 거부하세요.", 3, "signin", 5, "로그인 승인 여부를 확인해야 하지만 침해가 발생했다고 단정할 수 없습니다", "현재 회사 계정 로그인에 필요한 승인 요청입니다"),
        ("Google Chrome", "", "유출된 비밀번호", "저장된 계정 비밀번호가 공개된 유출 목록에서 발견되었습니다. 비밀번호를 변경하세요.", 4, "password", 5, "사용 중인 비밀번호 노출이 명시되어 변경이 필요합니다", "현재 비밀번호 유출 점검과 직접 연결됩니다"),
        ("OneDrive", "", "동기화 실패", "저장 공간이 가득 차 보고서 변경 내용이 업로드되지 않았습니다. 동기화 상태를 확인하세요.", 3, "sync", 5, "변경 내용 업로드가 실패했지만 즉각적인 손실은 명시되지 않았습니다", "현재 확인 중인 보고서 동기화 문제에 관한 안내입니다"),
        ("Windows 업데이트", "", "업데이트 사용 가능", "새 업데이트를 설치할 수 있습니다. 편한 시간에 설치하세요.", 1, "update", 4, "설치 시점을 선택할 수 있는 업데이트 안내입니다", "현재 업데이트를 확인하는 목적에 도움이 됩니다"),
        ("Windows", "", "배터리 절약 모드", "배터리 잔량 25%. 배터리 절약 모드가 켜졌습니다.", 2, "battery", 4, "배터리 상태 안내이며 즉시 종료 위험은 명시되지 않았습니다", "현재 배터리 사용량을 확인하는 데 도움이 됩니다"),
    ],
    "개인 중요": [
        ("KakaoTalk.exe", "엄마", "엄마", "병원 수납 창구야. 10분 안에 병원비 결제 승인해 줘야 퇴원 처리된대", 4, "hospitalpay", 5, "임박한 결제 승인이 퇴원 처리를 막고 있어 확인이 필요합니다", "현재 가족 병원비 납부에 필요한 요청입니다"),
        ("카드 앱", "이상거래 알림", "해외 결제 확인", "평소와 다른 해외 결제가 승인되었습니다. 본인 거래가 아니면 즉시 신고해 주세요.", 4, "bank", 5, "평소와 다른 승인 거래로 빠른 확인이 필요합니다", "현재 카드 이상거래를 확인하는 목적과 연결됩니다"),
        ("문자 메시지", "접수센터", "접수센터", "오늘 17시 지원서 접수 마감입니다. 첨부 서류가 누락되어 보완이 필요합니다.", 4, "application", 5, "오늘 마감하는 지원서에 필수 서류가 빠져 보완이 필요합니다", "현재 제출 중인 지원 서류에 관한 요청입니다"),
        ("KakaoTalk.exe", "룸메이트", "룸메이트", "천장에서 물 계속 떨어져. 전등 쪽까지 번지는데 관리실에 지금 연락해야 할 듯", 5, "leak", 5, "전등 쪽으로 누수가 번지고 있어 즉시 대응이 필요합니다", "현재 집 누수 수리 접수와 직접 연결됩니다"),
        ("병원 앱", "예약 안내", "진료 시간 변경", "내일 진료가 10시에서 11시로 변경되었습니다. 방문 전에 확인해 주세요.", 3, "appointment", 5, "내일 진료 시간 변경으로 비교적 빠른 확인이 필요합니다", "현재 확인 중인 내일 진료 예약 일정입니다"),
        ("Gmail", "보험 안내", "보험 갱신 조건 안내", "다음 주 갱신할 보험의 조건을 보냈습니다. 이번 주 안에 선택하신 내용을 회신해 주세요.", 2, "insurance", 4, "이번 주에 회신할 수 있는 다음 주 갱신 안내입니다", "현재 보험 갱신 조건을 비교하는 데 도움이 됩니다"),
    ],
    "개인 일반": [
        ("KakaoTalk.exe", "엄마", "엄마", "내일 저녁은 집에서 먹자. 냉장고에 반찬 있으니까 또 배달시키지 마ㅋㅋ", 2, "meal", 4, "내일 저녁 계획으로 당장 답하지 않아도 됩니다", "현재 내일 저녁 메뉴와 장보기를 정하는 데 도움이 됩니다"),
        ("Discord", "재훈", "#게임잡담", "보스한테 한 대 맞고 누웠음ㅋㅋ 스샷 표정 봐", 1, "game", 3, "게임 플레이를 두고 농담하는 내용으로 긴급한 요청이 없습니다", "같은 게임 이야기지만 현재 보스 공략에 주는 도움은 불분명합니다"),
        ("KakaoTalk.exe", "민수", "민수", "급함!! 오늘 점심 짜장 vs 짬뽕 뭐 먹냐ㅋㅋ", 1, "lunch", 4, "점심 메뉴 선택으로 급함이라는 표현만으로 긴급도를 높일 수 없습니다", "현재 점심 메뉴를 정하는 대화에 도움이 되는 선택 질문입니다"),
        ("Spotify", "", "새 재생목록 추천", "조용한 피아노와 함께하는 집중 시간 — 추천 재생목록을 들어보세요", 1, "music", 4, "선택적으로 들을 수 있는 음악 추천입니다", "현재 집중 음악 재생목록을 비교하는 목적에 도움이 됩니다"),
        ("YouTube", "영화 이야기", "이번 주 영화 잡담", "이번 주 본 영화 얘기하면서 수다 떨었어요ㅋㅋ 스포는 영상에 표시해 뒀습니다", 1, "film", 3, "선택적인 영상 소개로 즉시 확인할 필요가 없습니다", "영화라는 주제는 같지만 현재 영화 선택에 도움이 되는지는 불분명합니다"),
        ("KakaoTalk.exe", "수빈", "수빈", "지난 주말 여행 사진 원본 보내줄게. 편집할 때 이걸로 써", 1, "photo", 5, "사진 원본 공유로 긴급한 마감이나 요청이 없습니다", "현재 편집 중인 지난 주말 여행 사진의 원본입니다"),
    ],
    "광고/홍보": [
        ("Gmail", "쇼핑몰", "SSD 할인 안내", "오늘 자정까지 SSD 전 품목 15% 할인. 장바구니에서 쿠폰을 적용하세요.", 2, "ssd", 4, "할인을 놓쳐도 큰 피해가 명시되지 않은 선택적 구매 안내입니다", "현재 SSD 가격 비교와 구매 목적에 도움이 됩니다"),
        ("KakaoTalk.exe", "패션 브랜드 채널", "패션 브랜드 채널", "신상 손목시계 출시! 이번 주 무료 배송⌚", 1, "shoes", 2, "선택적인 신상품 광고입니다", "패션 상품이지만 현재 운동화 구매와 거의 무관합니다"),
        ("Slack", "서비스 소식", "#서비스-안내", "팀 저장 공간이 필요하신가요? 새 유료 요금제에 추가 용량이 포함됩니다.", 1, "storage", 4, "선택적인 유료 요금제 홍보입니다", "현재 팀 저장 공간 요금제를 비교하는 데 도움이 됩니다"),
        ("Google Chrome", "전자기기 쇼핑몰", "충전기 할인", "노트북 충전기 20% 할인! 단품과 케이블 세트를 비교해 보세요.", 1, "charger", 4, "선택적인 상품 할인 안내입니다", "현재 노트북 충전기 구매 비교에 도움이 됩니다"),
        ("문자 메시지", "신발 쇼핑몰", "신발 쇼핑몰", "[광고] 긴급! 운동화 쿠폰 만료까지 5분. 지금 받으세요", 2, "shoes", 4, "쿠폰 만료가 임박해도 놓쳤을 때 큰 피해가 명시되지 않았습니다", "현재 운동화 상품 비교와 구매 목적에 도움이 됩니다"),
        ("Discord", "게임 소식", "#이벤트", "협동 게임 신규 패키지 판매 시작! 추가 의상과 프로필 배지를 만나보세요", 1, "game", 3, "선택적인 게임 부가 상품 홍보입니다", "같은 게임이지만 현재 보스 공략에 직접 도움이 되는 내용은 아닙니다"),
    ],
    "기타": [
        ("Windows", "", "클립보드", "선택한 텍스트가 클립보드에 복사되었습니다.", 1, "clip", 5, "텍스트 복사 완료 안내로 긴급한 대응이 없습니다", "현재 메모에 인용문을 붙이는 작업에 필요한 결과입니다"),
        ("파일 탐색기", "", "압축 완료", "발표자료.zip 만들기가 완료되었습니다.", 1, "archive", 5, "파일 압축 완료 안내입니다", "현재 발표 파일을 압축하는 작업의 결과입니다"),
        ("Microsoft Edge", "", "다운로드 완료", "연습문제.pdf 다운로드가 완료되었습니다.", 1, "download", 5, "다운로드 완료 안내로 긴급한 대응이 없습니다", "현재 확인 중인 학습 자료 다운로드 결과입니다"),
        ("Windows", "", "창 정렬 팁", "창을 화면 가장자리로 끌어 여러 창을 나란히 배치해 보세요.", 1, "drawing", 2, "선택적으로 참고할 수 있는 사용 팁입니다", "같은 PC의 사용 팁이지만 현재 캐릭터 스케치와 거의 무관합니다"),
        ("시계", "", "타이머 저장됨", "공부용 타이머 설정이 저장되었습니다.", 1, "timer", 5, "타이머 설정 저장 완료 안내입니다", "현재 설정 중인 공부용 타이머의 결과입니다"),
        ("프린터", "", "인쇄 완료", "강의자료 8페이지 인쇄가 완료되었습니다.", 1, "print", 5, "인쇄 완료 안내로 긴급한 대응이 없습니다", "현재 확인 중인 강의자료 인쇄 결과입니다"),
    ],
}

# Only new ambiguities require a second human review, not all 48 new messages.
REVIEW_INDICES = {10, 12, 20, 22, 32, 33, 38, 42}  # 1-based expansion indices.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_review(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    return list(csv.DictReader(text.splitlines()))


def expand(seed_dir, output_dir, approval_text, approval_date):
    source = seed_dir / "candidates.jsonl"
    seed = [FilteringSample.model_validate_json(line) for line in source.read_text(encoding="utf-8").splitlines()]
    rows = decode_review(seed_dir / "review.csv")
    if len(seed) != 48 or len(rows) != 16 or any(row.get("feedback", "").strip() for row in rows):
        raise ValueError("seed review changed or contains feedback; resolve it before expansion")
    seed_manifest = json.loads((seed_dir / "manifest.json").read_text(encoding="utf-8"))
    if seed_manifest["dataset_sha256"] != digest(source) or seed_manifest["policy_version"] != POLICY_VERSION:
        raise ValueError("approved seed provenance or policy mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("choose an empty output directory; never overwrite review feedback")
    authored = [(category, item) for category in CATEGORIES for item in SCENARIOS[category]]
    if len(authored) != 48 or any(len(SCENARIOS[c]) != 6 for c in CATEGORIES):
        raise ValueError("need six independent authored messages per category")
    samples = list(seed)
    review, all_rows, lineage = [], [], []
    for sample in seed:
        lineage.append({"id": sample.notification.id, "message_id": sample.notification.id.rsplit("_", 1)[0],
                        "review_status": "human_approved", "source": "approved_seed"})
    start = datetime(2026, 10, 3, 4, 0, tzinfo=timezone.utc)
    for index, (category, item) in enumerate(authored):
        app, sender, title, body, urgency, context_key, score, urgent_reason, relevance_reason = item
        other_key = authored[(index + 24) % len(authored)][1][5]
        if context_key == other_key:
            raise ValueError("identical primary and unrelated contexts")
        message_id = f"v3_{index + 17:02d}"
        when = start + timedelta(minutes=index * 3)
        row = {"message_id": message_id, "app_name": app, "sender": sender, "title": title,
               "body": body, "category": category, "urgency_score": urgency,
               "related_window": CONTEXTS[context_key][1], "related_relevance": score,
               "related_decision": decision_label(urgency, score),
               "unrelated_window": CONTEXTS[other_key][1], "unrelated_relevance": 1,
               "unrelated_decision": decision_label(urgency, 1), "empty_relevance": 1,
               "empty_decision": decision_label(urgency, 1),
               "urgency_reason": urgent_reason, "relevance_reason": relevance_reason, "feedback": ""}
        all_rows.append(row)
        if index + 1 in REVIEW_INDICES:
            review.append(row)
        for role, key, relevance in (("related", context_key, score), ("unrelated", other_key, 1), ("empty", None, 1)):
            process, window = CONTEXTS[key] if key else ("", "")
            relation = relevance_reason if role == "related" else (
                "현재 창의 작업과 다른 주제입니다" if role == "unrelated"
                else "현재 작업 정보가 없어 관련성을 확인할 수 없습니다")
            sample = FilteringSample.model_validate({
                "notification": {"id": f"{message_id}_{role}", "app_name": app, "sender": sender,
                                 "title": title, "body": body, "timestamp": when.isoformat().replace("+00:00", "Z")},
                "context": {"active_process": process, "window_title": window,
                            "last_updated": (when - timedelta(seconds=3)).isoformat().replace("+00:00", "Z"),
                            "duration_seconds": (25, 65, 90, 180)[index % 4] if process else 0,
                            "recent_processes": [process] if process else []},
                "label": {"urgency_score": urgency, "relevance_score": relevance, "category": category,
                          "ai_summary_reason": f"{relation}; {urgent_reason}."},
            })
            parse_model_output(sample.label.model_dump_json())
            samples.append(sample)
            lineage.append({"id": sample.notification.id, "message_id": message_id,
                            "review_status": "authored_provisional", "context_role": role,
                            "source": "new_authored_scenario", "boundary_review": index + 1 in REVIEW_INDICES})
    by_context = defaultdict(set)
    by_message = defaultdict(list)
    for sample in samples:
        by_message[sample.notification.id.rsplit("_", 1)[0]].append(sample)
        if sample.context.active_process:
            by_context[(sample.context.active_process, sample.context.window_title)].add(sample.label.relevance_score)
    if len(by_message) != 64 or len({s.notification.id for s in samples}) != 192:
        raise ValueError("incorrect message or sample count")
    if any(len(group) != 3 or len({(s.label.urgency_score, s.label.category) for s in group}) != 1 for group in by_message.values()):
        raise ValueError("context variants must preserve urgency and category")
    if any(1 not in values or len(values) < 2 for values in by_context.values()):
        raise ValueError("a populated window alone predicts relevance")
    if len({(s.notification.app_name, s.notification.title, s.notification.body) for s in samples}) != 64:
        raise ValueError("duplicate notification content")
    output_dir.mkdir(parents=True, exist_ok=True)
    approval = {"approval_date": approval_date, "user_statement": approval_text,
                "approved_notification_count": 16, "approved_sample_count": 48,
                "seed_sha256": digest(source), "review_csv_sha256": digest(seed_dir / "review.csv"),
                "policy_version": POLICY_VERSION}
    (output_dir / "seed_approval.json").write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "candidates.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(sample.model_dump_json() + "\n")
    with (output_dir / "model_io.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps({"id": sample.notification.id,
                                   "messages": build_messages(sample.notification, sample.context),
                                   "expected_output": sample.label.model_dump(),
                                   "policy_decision": decision_label(sample.label.urgency_score, sample.label.relevance_score)}, ensure_ascii=False) + "\n")
    for name, records in (("review.csv", review), ("new_notifications.csv", all_rows)):
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = ["# 2차 검수: 새 경계 사례 8개", "",
             "첫 16개는 승인본으로 보존했습니다. 이 문서는 추가한 48개 중 점수 경계가 있는 8개만 보여줍니다.",
             "원문과 문맥별 예상 정답을 읽고 바꾸고 싶은 항목만 review.csv의 feedback 열에 적어 주세요.",
             "모두 합성 예시이며 집중 모드 ON입니다. 긴급도 또는 관련도 4 이상이면 PASS입니다.",
             "검수 완료를 알려주시면 빈 의견은 승인으로 처리합니다. 현재는 확장 후보이며 학습은 아직 진행하지 않습니다.", ""]
    for row in review:
        lines.extend([f"## {row['message_id']} · {row['category']}", "",
                      f"앱: {row['app_name']} / 발신자: {row['sender'] or '(없음)'} / 제목: {row['title']}", "",
                      f"> {row['body']}", "", f"긴급도 **{row['urgency_score']}** — {row['urgency_reason']}", "",
                      "| 문맥 | 관련도 | 결과 |", "|---|---:|---|",
                      f"| {row['related_window']} | {row['related_relevance']} | {row['related_decision']} |",
                      f"| {row['unrelated_window']} | 1 | {row['unrelated_decision']} |",
                      f"| 빈 창 · 집중 모드 ON | 1 | {row['empty_decision']} |", "",
                      f"기본 문맥 판단: {row['relevance_reason']}.", ""])
    (output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"status": "first expansion; awaiting boundary review; not for training",
                "notification_count": 64, "sample_count": len(samples), "human_approved_notifications": 16,
                "new_provisional_notifications": 48, "review_notifications": len(review),
                "policy_version": POLICY_VERSION, "focus_mode": "ON assumed",
                "seed_sha256": digest(source), "dataset_sha256": digest(output_dir / "candidates.jsonl"),
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "categories": dict(Counter(group[0].label.category for group in by_message.values())),
                "apps": dict(Counter(group[0].notification.app_name for group in by_message.values())),
                "urgency_counts": dict(Counter(group[0].label.urgency_score for group in by_message.values())),
                "relevance_counts": dict(Counter(s.label.relevance_score for s in samples)),
                "policy_counts": dict(Counter(decision_label(s.label.urgency_score, s.label.relevance_score) for s in samples)),
                "populated_context_count": len(by_context), "constant_relevance_populated_contexts": 0,
                "split_status": "unassigned; isolate scenario families and context variants before evaluation",
                "source": "approved fictional seed plus authored synthetic scenarios; no private messages"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "lineage.json").write_text(json.dumps({"items": lineage}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, default=ROOT / "v3_seed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "v3_expansion_01")
    parser.add_argument("--seed-approval", required=True, help="explicit human approval text")
    parser.add_argument("--approval-date", required=True, help="human approval date YYYY-MM-DD")
    args = parser.parse_args()
    print(json.dumps(expand(args.seed_dir, args.output_dir, args.seed_approval, args.approval_date), ensure_ascii=False))


if __name__ == "__main__":
    main()
