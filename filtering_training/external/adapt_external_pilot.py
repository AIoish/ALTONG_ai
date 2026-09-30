"""Build a hand-rewritten Korean pilot from selected NotifAI notification ideas.

The source priorities are never used as ALTONG labels. Output stays Git-ignored
until an independent review approves the rewritten text and all four labels.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from filtering_training import TRAINING_ROOT

from filtering_training.datasets.prepare_dataset import DATASET_PATH, load_samples
from filtering_training.datasets.prepare_holdout import HOLDOUT_PATH, notification_key
from src.filtering.prompt import parse_model_output
from src.filtering.schema import FilteringSample

from filtering_training.external.select_external_candidates import OUTPUT_PATH as SOURCE_POOL


OUTPUT_PATH = TRAINING_ROOT / "outputs" / "candidates" / "external_adapted_pilot.jsonl"
CONTEXTS = {
    "code": ("Code.exe", "auth_service.py - 가상 프로젝트 - Visual Studio Code", ["Code.exe", "chrome.exe"], 45),
    "cdn": ("Code.exe", "cdn_config.yml - 가상 프로젝트 - Visual Studio Code", ["Code.exe", "WindowsTerminal.exe"], 65),
    "build": ("Code.exe", "test_results.py - 가상 프로젝트 - Visual Studio Code", ["Code.exe", "WindowsTerminal.exe"], 35),
    "calendar": ("OUTLOOK.EXE", "다음 주 일정 - Outlook", ["OUTLOOK.EXE", "ms-teams.exe"], 25),
    "design": ("Figma.exe", "대시보드 시안 - Figma", ["Figma.exe", "chrome.exe"], 85),
    "document": ("WINWORD.EXE", "발표 자료 - Word", ["WINWORD.EXE", "chrome.exe"], 50),
    "finance": ("EXCEL.EXE", "예산 정리 - Excel", ["EXCEL.EXE", "chrome.exe"], 75),
    "sync": ("Dropbox.exe", "동기화 중인 제안서 폴더 - Dropbox", ["Dropbox.exe", "WINWORD.EXE"], 40),
    "headphones": ("chrome.exe", "무선 헤드폰 비교 - Chrome", ["chrome.exe", "explorer.exe"], 55),
    "phone": ("chrome.exe", "휴대전화 구매 비교 - Chrome", ["chrome.exe", "explorer.exe"], 60),
    "migration": ("Code.exe", "database_migration.sql - 가상 프로젝트 - Visual Studio Code", ["Code.exe", "WindowsTerminal.exe"], 70),
    "review": ("chrome.exe", "열린 코드 리뷰 목록 - GitHub - Chrome", ["chrome.exe", "Code.exe"], 45),
    "redesign": ("Figma.exe", "웹사이트 개편 시안 - Figma", ["Figma.exe", "chrome.exe"], 80),
    "empty": ("", "", [], 0),
    "files": ("explorer.exe", "다운로드 - 파일 탐색기", ["explorer.exe"], 25),
    "release": ("Code.exe", "release_notes.md - 가상 프로젝트 - Visual Studio Code", ["Code.exe", "chrome.exe", "WindowsTerminal.exe"], 60),
    "meeting": ("ms-teams.exe", "프로젝트 주간 회의 - Microsoft Teams", ["ms-teams.exe", "OUTLOOK.EXE", "chrome.exe"], 40),
    "vpn": ("mstsc.exe", "원격 작업 환경", ["mstsc.exe", "chrome.exe", "Code.exe"], 70),
}

# Fictional role labels exercise the optional sender field without personal names.
SENDERS = {
    "04057": "일정 알림",
    "11646": "회의 일정",
    "00437": "팀 일정",
    "10585": "계정 보안",
}
# source ID, rewritten app/title/body, category, urgency, relevance, context, reason.
# Source ideas are from NotifAI; all Korean wording and labels below are new.
PILOT = [
    ("12399", "GitHub", "후원 기능 안내", "개발자 후원 프로필을 개설할 수 있습니다.", "광고/홍보", 1, 1, "code", "후원 기능을 소개하는 안내로 현재 인증 코드 작업과 관계가 없습니다."),
    ("14955", "Slack", "다음 주 면담 일정", "다음 주에 예정된 팀 면담 시간을 확인해 주세요.", "일정/회의", 2, 3, "calendar", "일정 확인과 관련 있지만 면담이 다음 주라 즉시 볼 필요는 없습니다."),
    ("02727", "Microsoft Teams", "연간 평가 결과", "올해의 업무 평가 결과가 게시됐습니다. 편할 때 확인해 주세요.", "일반 업무", 2, 1, "design", "업무 결과를 확인하는 알림이지만 현재 디자인 작업과 직접 관련은 없습니다."),
    ("09613", "Jira", "CDN 설정 작업 배정", "이번 주 금요일까지 CDN 설정을 개선하는 작업이 배정됐습니다.", "일반 업무", 2, 5, "cdn", "현재 수정 중인 CDN 설정의 작업 배정이지만 마감이 임박했다는 근거는 없습니다."),
    ("14401", "Slack", "운영 데이터베이스 연결 장애", "데이터베이스 연결이 부족해 여러 서비스 응답이 느려졌습니다. 운영 담당자의 확인이 필요합니다.", "긴급 업무", 5, 1, "document", "현재 문서 작업과는 무관해도 서비스 장애가 진행 중이므로 즉시 확인해야 합니다."),
    ("04957", "GitHub", "기본 브랜치 빌드 실패", "테스트 두 건이 실패해 기본 브랜치의 빌드가 완료되지 않았습니다.", "일반 업무", 3, 5, "build", "현재 확인 중인 테스트와 직접 관련 있지만 배포 마감이나 서비스 장애는 확인되지 않습니다."),
    ("05017", "KakaoTalk.exe", "다음 주 시간 문의", "다음 주에 시간 되는 날이 있는지 알려줘.", "개인 일반", 1, 1, "code", "다음 주 개인 일정 문의로 현재 코드 작업과 무관하고 답변을 미뤄도 됩니다."),
    ("02586", "KakaoTalk.exe", "이사 도와줄 수 있어?", "다음 주 토요일에 이사하는데 시간이 되면 도와줄 수 있을까?", "개인 일반", 2, 1, "finance", "다음 주 개인적인 부탁으로 현재 예산 작업과 관련이 없습니다."),
    ("12950", "병원 앱", "치과 예약 안내", "내일 오후 치과 예약이 있습니다. 시간을 확인해 주세요.", "일정/회의", 3, 1, "code", "내일 방문 일정을 확인해야 하지만 현재 코드 작업과 관련은 없습니다."),
    ("15367", "이동 서비스", "차량 도착 2분 전", "예약한 차량이 곧 도착합니다. 탑승 장소로 이동해 주세요.", "개인 중요", 4, 1, "code", "현재 작업과 무관해도 차량이 곧 도착하므로 지금 확인할 필요가 있습니다."),
    ("00384", "이메일", "여권 수령 안내", "신청한 비자의 처리가 끝났습니다. 내일부터 여권을 받을 수 있습니다.", "개인 중요", 3, 1, "design", "개인 서류 수령 안내로 현재 디자인 작업과는 무관하며 즉시 대응할 일은 아닙니다."),
    ("03577", "배송 앱", "수령 서명 필요", "배송 예정 물품을 받으려면 서명이 필요합니다. 수령 시간을 확인해 주세요.", "개인 중요", 3, 1, "code", "물품 수령 시간을 정해야 하지만 현재 코드 작업과 관련은 없습니다."),
    ("11964", "Dropbox", "저장 공간 가득 참", "저장 공간이 부족해 파일 동기화가 멈췄습니다.", "시스템/보안", 3, 5, "sync", "현재 동기화 중인 폴더에 영향을 주므로 작업을 이어가기 전에 확인해야 합니다."),
    ("08044", "배송 앱", "택배 배송 완료", "주문한 물품이 현관 앞에 도착했습니다.", "개인 일반", 2, 1, "code", "배송 완료 알림으로 현재 코드 작업과 관련이 없고 나중에 확인할 수 있습니다."),
    ("10102", "Google Chrome", "계정 활동 확인", "평소와 다른 계정 활동이 감지됐습니다. 로그인 내역을 확인해 주세요.", "시스템/보안", 4, 1, "document", "현재 문서 작업과 관계없이 계정 보안을 위해 빠른 확인이 필요합니다."),
    ("03604", "은행 앱", "알 수 없는 결제", "직접 승인하지 않은 결제 요청이 발생했습니다. 거래 내역을 확인해 주세요.", "개인 중요", 5, 1, "document", "현재 작업과 무관해도 승인하지 않은 결제이므로 즉시 확인해야 합니다."),
    ("14433", "쇼핑 앱", "회원 할인 행사", "회원 대상 할인 행사가 시작됐습니다. 원하는 상품을 둘러보세요.", "광고/홍보", 1, 1, "code", "할인 행사 홍보로 현재 코드 작업과 관계가 없습니다."),
    ("02403", "쇼핑 앱", "헤드폰 할인", "살펴보던 무선 헤드폰을 할인 중입니다. 상품을 확인해 보세요.", "광고/홍보", 1, 4, "headphones", "현재 비교 중인 헤드폰과 관련 있지만 할인 광고라 즉시 대응할 필요는 없습니다."),
    ("00831", "동영상 서비스", "새 시즌 공개", "구독 중인 프로그램의 새 시즌을 시청할 수 있습니다.", "광고/홍보", 1, 1, "calendar", "영상 시청 홍보로 현재 일정 확인 작업과 무관합니다."),
    ("03519", "쇼핑 앱", "휴대전화 특가", "살펴보던 휴대전화의 할인 상품이 등록됐습니다.", "광고/홍보", 1, 4, "phone", "현재 휴대전화 구매 비교와 관련 있지만 할인 광고의 긴급도는 낮습니다."),
    ("14034", "Linear", "데이터베이스 이전 작업 대기", "배포 설정 변경이 끝나야 데이터베이스 이전 작업을 시작할 수 있습니다.", "일반 업무", 3, 5, "migration", "현재 진행 중인 이전 작업이 선행 작업 때문에 멈춰 있어 확인이 필요합니다."),
    ("04113", "Notion", "API 변경 문서 공개", "새 API 문서에 호환되지 않는 변경 내용이 정리됐습니다.", "일반 업무", 3, 4, "code", "현재 API 코드 작업에 영향을 줄 수 있는 변경 문서라 확인이 필요합니다."),
    ("14458", "OneDrive", "클라우드 저장 공간 부족", "저장 공간이 거의 가득 찼습니다. 동기화 상태를 확인해 주세요.", "시스템/보안", 2, 1, "code", "저장 공간 안내지만 현재 코드 작업이 중단됐다는 근거는 없습니다."),
    ("00499", "GitHub", "저장소에 인증 정보 노출", "새로 올린 코드에서 서비스 인증 정보가 발견됐습니다. 즉시 제거해 주세요.", "시스템/보안", 5, 4, "code", "코드에 인증 정보가 노출돼 빠른 제거와 확인이 필요합니다."),
    ("08130", "GitHub", "라이브러리 보안 업데이트", "현재 사용하는 라이브러리에 높은 위험의 취약점이 보고됐습니다.", "시스템/보안", 4, 3, "code", "현재 개발 환경에서 쓰는 라이브러리의 취약점이므로 빠른 점검이 필요합니다."),
    ("04911", "KakaoTalk.exe", "회의 중 웃긴 장면", "아까 회의에서 있었던 일 아직도 웃겨.", "개인 일반", 1, 1, "code", "가벼운 개인 대화로 현재 코드 작업과 무관합니다."),
    ("12233", "GitHub", "코드 리뷰 대기", "검토를 기다리는 변경 요청이 네 건 있습니다. 순서를 정해 확인해 주세요.", "일반 업무", 3, 5, "review", "현재 보고 있는 코드 리뷰 목록과 직접 관련 있지만 즉시 처리할 마감은 제시되지 않았습니다."),
    ("04847", "Asana", "웹사이트 개편 마감 안내", "내일 마감인 웹사이트 개편 작업에 미완료 항목이 두 개 남았습니다.", "긴급 업무", 4, 5, "redesign", "현재 작업 중인 개편안의 마감이 내일이고 남은 항목이 있어 빠른 확인이 필요합니다."),
    ("08010", "KakaoTalk.exe", "다음 달 여행 예약", "다음 달 여행 교통편 예약했어. 일정은 나중에 같이 보자.", "개인 일반", 2, 1, "finance", "다음 달 개인 여행 안내로 현재 예산 작업과 관련이 없습니다."),
    ("09688", "배달 앱", "음식 주문 접수", "주문한 음식이 접수됐습니다. 예상 도착까지 약 20분입니다.", "개인 일반", 2, 1, "code", "음식 주문 상태를 알려 주지만 현재 코드 작업을 중단할 일은 아닙니다."),
    ("01137", "운영 알림", "데이터 백업 실패", "예약된 데이터 백업이 실패해 복구 작업을 진행 중입니다.", "긴급 업무", 4, 1, "document", "현재 문서 작업과 무관해도 백업 실패 상태를 빠르게 확인해야 합니다."),
    ("08007", "쇼핑 앱", "쿠폰 만료 예정", "보유한 할인 쿠폰이 오늘 만료됩니다. 사용 여부를 확인해 보세요.", "광고/홍보", 2, 1, "finance", "구매를 유도하는 쿠폰 알림으로 현재 예산 작업과 직접 관련이 없습니다."),
    ("00350", "Microsoft Teams", "직무 변경 안내", "새 직무가 확정됐습니다. 상세 안내는 인사 페이지에서 확인할 수 있습니다.", "개인 중요", 2, 1, "code", "개인에게 중요한 업무 소식이지만 현재 코드 작업과 무관하고 즉시 대응할 내용은 없습니다."),
    ("09625", "GitHub", "인증 코드 리뷰 의견", "인증 구조 변경에 수정 요청 의견 세 건이 달렸습니다.", "일반 업무", 3, 5, "code", "현재 수정 중인 인증 코드에 대한 검토 의견이라 작업과 직접 관련이 있습니다."),
    ("00521", "Microsoft Office", "발표 자료 동기화 진행", "발표 자료의 클라우드 동기화가 아직 진행 중입니다.", "시스템/보안", 2, 4, "document", "현재 열어 둔 발표 자료의 저장 상태와 관련 있지만 동기화 실패는 확인되지 않았습니다."),
    ("10326", "GitHub", "오래된 이슈 정리 예정", "한동안 활동이 없던 이슈 다섯 건이 다음 주에 닫힐 예정입니다.", "일반 업무", 2, 2, "release", "저장소 관리 소식이지만 현재 작성 중인 릴리스 문서와 직접 연결되지는 않습니다."),
    ("04057", "Microsoft Teams", "일대일 면담 30분 전", "예약된 일대일 면담이 30분 뒤 시작됩니다.", "일정/회의", 3, 1, "empty", "회의가 곧 시작되지만 현재 작업 맥락이 없어 작업 연관도를 판단할 근거가 없습니다."),
    ("15468", "일정 앱", "내일 주차 제한", "내일 오전 도로 청소가 예정돼 있습니다. 해당 구역에 주차한 차량은 이동해 주세요.", "개인 중요", 3, 1, "code", "차량 이동이 필요할 수 있지만 현재 코드 작업과 무관하고 즉시 조치할 상황은 아닙니다."),
    ("07200", "GitHub", "예전 저장소 보관 완료", "사용하지 않는 예전 저장소가 읽기 전용으로 전환됐습니다.", "기타", 1, 1, "empty", "단순 보관 완료 상태이며 현재 작업 맥락이 없어 연관성을 확인할 수 없습니다."),
    ("11646", "Outlook", "예산 검토 회의 15분 전", "예산 검토 회의가 15분 뒤 시작됩니다. 회의 자료를 확인해 주세요.", "일정/회의", 4, 5, "finance", "현재 예산 자료를 열어 두었고 회의가 곧 시작돼 준비가 필요합니다."),
    ("15393", "GitHub", "도구 업데이트 안내", "사용 중인 개발 도구의 새 버전이 공개됐습니다. 변경 내용을 확인할 수 있습니다.", "시스템/보안", 1, 1, "empty", "업데이트 공개 안내일 뿐 즉시 적용할 필요가 없고 현재 작업 정보도 없습니다."),
    ("05060", "GitHub", "프로젝트 참여자 증가", "공개 저장소에 새로운 참여자가 세 명 추가됐습니다.", "기타", 1, 1, "files", "참여자 수를 알려 주는 통계성 소식으로 현재 파일 탐색 작업과 무관합니다."),
    ("00437", "Outlook", "팀 회의 15분 전", "정기 팀 회의가 15분 뒤 시작됩니다.", "일정/회의", 4, 4, "meeting", "현재 팀 회의 창을 열어 둔 상태에서 회의 시작이 임박했습니다."),
    ("01748", "GitHub", "새 버전 배포 완료", "담당 저장소의 새 버전 배포가 완료됐습니다. 배포 기록을 확인할 수 있습니다.", "일반 업무", 2, 5, "release", "현재 작성 중인 릴리스 문서와 직접 관련 있지만 배포 실패나 즉시 조치할 내용은 없습니다."),
    ("05088", "GitHub", "이전 저장소 보관됨", "이전 저장소가 보관 처리돼 수정하려면 보관을 해제해야 합니다.", "일반 업무", 2, 1, "files", "저장소 수정이 필요할 때 확인하면 되는 정보이며 현재 파일 탐색과는 연결되지 않습니다."),
    ("15876", "GitHub", "다운로드 기록 갱신", "공개 저장소의 누적 다운로드 수가 새로운 기록에 도달했습니다.", "기타", 1, 1, "empty", "통계 기록을 알려 주는 소식이고 현재 작업 맥락도 확인할 수 없습니다."),
    ("01107", "GitHub", "활동 없는 이슈 종료", "오래된 이슈가 정리 규칙에 따라 자동으로 닫혔습니다.", "일반 업무", 1, 2, "release", "저장소 관리 정보지만 현재 릴리스 문서에 영향을 준다는 근거는 없습니다."),
    ("04316", "GitHub", "저장소 이전 완료", "코드 저장소를 새 위치로 옮기는 작업이 완료됐습니다.", "일반 업무", 2, 3, "release", "현재 릴리스 문서와 같은 저장소의 변경이지만 문서 작업을 즉시 멈출 필요는 없습니다."),
    ("00356", "GitHub", "사용하지 않는 저장소 보관", "장기간 변경이 없던 저장소가 자동으로 보관 처리됐습니다.", "기타", 1, 1, "files", "오래된 저장소의 관리 상태만 알려 주며 현재 파일 탐색과 무관합니다."),
    ("08273", "게임 앱", "게임 콘텐츠 업데이트", "설치한 게임에 새 콘텐츠가 추가됐습니다.", "광고/홍보", 1, 1, "meeting", "게임 이용을 유도하는 소식으로 현재 회의와 관계가 없습니다."),
    ("04088", "오디오북 앱", "도서 다운로드 준비 완료", "선택한 오디오북을 이제 기기에 내려받을 수 있습니다.", "개인 일반", 1, 1, "empty", "개인 콘텐츠 이용 안내이며 현재 작업 맥락을 알 수 없습니다."),
    ("02460", "원격 접속", "원격 연결 종료", "원격 작업 환경과의 연결이 끊어졌습니다. 작업을 계속하려면 다시 연결해 주세요.", "시스템/보안", 4, 5, "vpn", "현재 원격 작업 환경의 연결이 끊겨 작업을 계속하려면 재연결이 필요합니다."),
    ("15650", "Spotify", "앱 업데이트 가능", "재생 안정성을 개선한 새 버전이 준비됐습니다.", "시스템/보안", 1, 1, "meeting", "음악 앱 업데이트 안내로 현재 회의와 무관하고 즉시 설치할 필요는 없습니다."),
    ("14273", "보안 앱", "보안 검사 완료", "기기 검사가 끝났으며 발견된 위협은 없습니다.", "시스템/보안", 1, 1, "files", "위협이 없다는 검사 결과로 현재 파일 탐색을 중단할 이유는 없습니다."),
    ("04069", "VPN 앱", "보안 연결 완료", "VPN 연결이 정상적으로 설정됐습니다.", "시스템/보안", 1, 4, "vpn", "현재 원격 작업 환경과 관련된 연결 상태지만 별도 대응이 필요하지 않습니다."),
    ("10585", "쇼핑 앱", "새 기기 로그인 확인", "계정에 새 기기에서의 로그인이 감지됐습니다. 본인 활동인지 확인해 주세요.", "시스템/보안", 4, 1, "empty", "계정 접근을 확인해야 하지만 현재 작업 맥락이 없어 연관성은 판단할 수 없습니다."),
    ("15671", "기기 설정", "배터리 성능 저하", "배터리 최대 성능이 낮아져 사용 시간이 줄어들 수 있습니다.", "시스템/보안", 2, 1, "meeting", "기기 상태 안내지만 현재 회의를 즉시 중단할 상황은 아닙니다."),
    ("15711", "오디오북 앱", "추천 도서 안내", "최근 감상한 도서와 비슷한 오디오북을 살펴보세요.", "광고/홍보", 1, 1, "files", "오디오북 이용을 권하는 추천으로 현재 파일 탐색과 무관합니다."),
]


def build_pilot(source_pool: Path, output: Path, training: Path = DATASET_PATH,
                holdout: Path = HOLDOUT_PATH) -> dict:
    pool = {json.loads(line)["source_id"] for line in source_pool.read_text(encoding="utf-8").splitlines() if line.strip()}
    source_ids = [case[0] for case in PILOT]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("pilot repeats a source ID")
    missing = set(source_ids) - pool
    if missing:
        raise ValueError(f"pilot source IDs absent from selected pool: {sorted(missing)}")
    existing = load_samples(training)
    if holdout.exists():
        existing.extend(load_samples(holdout))
    reserved = {notification_key(item) for item in existing}
    stamp = datetime(2026, 9, 26, 10, 0, tzinfo=timezone.utc)
    samples = []
    provenance = []
    for index, (source_id, app, title, body, category, urgency, relevance, topic, reason) in enumerate(PILOT, 1):
        process, window, recent, duration = CONTEXTS[topic]
        current = stamp + timedelta(minutes=index * 7)
        item = FilteringSample.model_validate({
            "notification": {
                "id": f"external_pilot_{index:03d}", "app_name": app, "sender": SENDERS.get(source_id, ""),
                "title": title, "body": body,
                "timestamp": current.isoformat().replace("+00:00", "Z"),
            },
            "context": {
                "active_process": process, "window_title": window,
                "last_updated": (current - timedelta(seconds=4)).isoformat().replace("+00:00", "Z"),
                "duration_seconds": duration, "recent_processes": recent,
            },
            "label": {
                "urgency_score": urgency, "relevance_score": relevance,
                "category": category, "ai_summary_reason": reason,
            },
        })
        parse_model_output(item.label.model_dump_json())
        samples.append(item)
        provenance.append({"candidate_id": item.notification.id, "source_id": source_id})
    keys = [notification_key(item) for item in samples]
    if len(keys) != len(set(keys)) or set(keys) & reserved:
        raise ValueError("pilot notification text overlaps existing data")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in samples:
            stream.write(item.model_dump_json() + "\n")
    output.with_suffix(".provenance.json").write_text(
        json.dumps({"status": "provisional relabeling; review before training",
                    "source_pool": str(source_pool), "items": provenance},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"count": len(samples), "distinct_notification_texts": len(set(keys)),
            "status": "provisional Korean adaptation; independent label review pending"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, default=SOURCE_POOL)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(build_pilot(args.source_pool, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
