"""Build 2,000 local candidates: 1,000 new notifications with context pairs.

Five families/category x five subjects x five semantic states x two contexts.
Labels are authored, provisional, and need independent review. No private file
or development-evaluation wording is read for generation.
"""

from filtering_training.common.paths import LEGACY_OUTPUTS_ROOT
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from filtering_training.quality.audit_dataset import audit_samples
from filtering_training.legacy.generation.generate_rapid_dataset import OUTPUT_PATH as BASE_DATASET
from filtering_training.common.dataset import load_samples
from filtering_training.preparation.prepare_holdout import HOLDOUT_PATH, notification_key
from src.filtering.prompt import CATEGORIES, parse_model_output
from src.filtering.schema import FilteringSample

ROOT = LEGACY_OUTPUTS_ROOT
OUTPUT_PATH = ROOT / "candidates" / "targeted_korean_2000.jsonl"
COMBINED_PATH = ROOT / "candidates" / "combined_korean_5000.jsonl"

SUBJECTS = {
    "work": ["상품 API", "프로필 API", "댓글 API", "쿠폰 API", "통계 API"],
    "schedule": ["캐시 설계 리뷰", "데이터 이전 점검", "접근성 검토", "성능 튜닝 회의", "릴리스 회고"],
    "system": ["개발 노트북", "테스트 PC", "업무 노트북", "원격 작업 PC", "공용 개발 PC"],
    "personal": ["주거비", "교육비", "관리비", "통신비", "보험료"],
    "casual": ["보드게임 모임", "영화 모임", "산책 모임", "요리 모임", "독서 모임"],
    "promotion": ["SSD", "웹캠", "USB 허브", "마우스 패드", "충전 케이블"],
    "misc": ["작업 메모", "설정 메모", "정리 메모", "참고 메모", "아이디어 메모"],
}

PERSONAL_SUBJECTS = {
    "debit_return": ["인터넷 요금", "전기 요금", "수도 요금", "가스 요금", "휴대전화 요금"],
    "home_access": ["인터넷 설치", "가스 점검", "수도 점검", "난방 점검", "정수기 점검"],
    "duplicate_charge": ["구독료", "온라인 구매 대금", "예약금", "교통비", "배송비"],
    "support_deadline": ["주거비", "교육비", "의료비", "양육비", "난방비"],
    "family_payment": ["생활비", "교재비", "병원비", "교통비", "식비"],
}
# Each state changes the event or impact, not just a generic urgency suffix.
# app, domain, family ID, title, five (body, urgency, evidence) states.
FAMILIES = {
    "긴급 업무": [
        ("Sentry", "work", "quota", "{s} 운영 할당량 초과", [
            ("{s} 호출 한도 소진으로 실제 사용자 요청이 거절됩니다.", 5, "사용자 요청이 거절되는 운영 장애"),
            ("{s} 운영 제한에 걸려 오늘 출시 검증을 진행하지 못합니다. 담당자 조치 부탁.", 4, "오늘 출시 검증이 막혀 조치 필요"),
            ("{s} 요청 제한 때문에 유료 고객 사용 불가. 복구 담당 확인 요망.", 5, "유료 고객이 서비스를 사용할 수 없음"),
            ("{s} 운영 한도 증액 승인을 30분 안에 끝내야 예정된 출시가 가능합니다.", 4, "출시를 위한 한도 승인 마감이 임박"),
            ("{s} 한도 초과가 다른 운영 서비스로 확산돼 후속 요청도 실패합니다.", 5, "운영 장애의 영향이 다른 서비스로 확산"),
        ]),
        ("GitLab", "work", "migration", "{s} DB 변경 복구 필요", [
            ("{s} 운영 DB 변경 후 새 레코드가 손상되고 있습니다. 원복 담당 호출.", 5, "운영 데이터 손상이 진행 중"),
            ("{s} DB 변경을 되돌리지 않으면 20분 뒤 릴리스를 진행할 수 없습니다.", 4, "임박한 릴리스 전에 원복이 필요"),
            ("{s} 이전 작업이 잠금을 유지해 사용자 수정이 모두 멈췄습니다.", 5, "DB 잠금으로 사용자 수정이 멈춤"),
            ("{s} DB 변경 원복 절차 승인이 남았습니다. 오늘 릴리스 전에 확인 바랍니다.", 4, "오늘 릴리스 전에 원복 절차 승인 필요"),
            ("{s} DB 변환 실패로 잘못된 값이 계속 저장됩니다. 쓰기 중지 판단 필요.", 5, "잘못된 운영 데이터 저장이 계속됨"),
        ]),
        ("Azure DevOps", "work", "dependency", "{s} 연동 계약 불일치", [
            ("{s} 운영 연동 규격이 바뀌어 외부 고객 요청이 모두 거절되고 있습니다.", 5, "운영 연동 불일치로 고객 요청 실패"),
            ("{s} 연동 응답 구조가 바뀌었습니다. 1시간 뒤 릴리스 전 어댑터 수정 필요.", 4, "릴리스 전에 연동 코드 수정이 필요"),
            ("{s} 연동 버전 불일치로 결제 후 결과 전달이 중단됐습니다.", 5, "결제 이후 결과 전달이 중단됨"),
            ("{s} 호환 라이브러리 교체가 승인 대기입니다. 오늘 반영 전에 확인 바랍니다.", 4, "오늘 반영을 위한 호환 변경 승인 필요"),
            ("{s} 운영 응답 형식 불일치로 고객 데이터가 잘못 표시됩니다. 원복 판단 필요.", 5, "운영 고객 데이터가 잘못 표시됨"),
        ]),
        ("Slack", "work", "release_signing", "{s} 릴리스 서명 문제", [
            ("{s} 배포 서명이 만료돼 고객 설치가 실패합니다. 재서명 담당 확인 부탁.", 5, "고객 설치가 실제 실패하는 상태"),
            ("{s} 배포 서명 갱신을 15분 안에 완료해야 예약된 출시를 진행할 수 있습니다.", 4, "예약된 출시 전에 서명 갱신 필요"),
            ("{s} 서명 불일치로 자동 업데이트가 중단됐습니다. 운영 담당 호출 바랍니다.", 5, "고객 자동 업데이트가 중단됨"),
            ("{s} 새 인증서 적용 승인 대기. 오늘 외부 공개 전 담당 검토 필요.", 4, "오늘 외부 공개 전에 인증서 승인 필요"),
            ("{s} 고객 배포 패키지 검증 실패가 계속 발생합니다. 정상 서명으로 재발행 필요.", 5, "배포 패키지 검증 실패가 계속 발생"),
        ]),
        ("Datadog", "work", "lost_events", "{s} 운영 이벤트 유실", [
            ("{s} 운영 이벤트가 소비 전에 삭제되고 있습니다. 유실 방지 조치 필요.", 5, "운영 이벤트가 실제 삭제되는 중"),
            ("{s} 이벤트 재처리 승인을 30분 안에 해야 오늘 집계를 마칠 수 있습니다.", 4, "오늘 집계를 위한 재처리 승인 마감이 임박"),
            ("{s} 중복 소비로 고객 상태가 계속 잘못 변경됩니다. 소비 중지 판단 필요.", 5, "고객 상태의 잘못된 변경이 진행 중"),
            ("{s} 재처리 대상 확인이 남았습니다. 오늘 외부 전달 전에 담당 검토 부탁.", 4, "오늘 외부 전달 전 재처리 대상 확인 필요"),
            ("{s} 이벤트 누락으로 사용자 변경이 반영되지 않습니다. 복구 담당 확인 요망.", 5, "사용자 변경이 운영 서비스에 반영되지 않음"),
        ]),
    ],
    "일반 업무": [
        ("GitHub", "work", "benchmark", "{s} 성능 비교 자료", [
            ("{s} 로컬 벤치마크 표를 올렸습니다. 다음 주 최적화 참고용입니다.", 2, "다음 주 최적화에 참고할 자료"),
            ("{s} 이전 버전과 비교한 실행 시간을 공유합니다. 당장 수정할 항목은 없습니다.", 1, "실행 시간 공유이며 수정 요청 없음"),
            ("{s} 벤치마크 설정 검토를 오늘 중 부탁드립니다. 운영 영향은 없습니다.", 3, "오늘 중 검토 요청이지만 운영 영향 없음"),
            ("{s} 실험용 성능 결과를 저장했습니다. 급한 확인은 필요 없습니다.", 1, "실험 결과 보관이며 급한 확인 불필요"),
            ("{s} 다음 스프린트 성능 목표에 의견 부탁드립니다. 이번 주에 확인해 주세요.", 2, "이번 주 의견 수렴이며 즉시 대응 불필요"),
        ]),
        ("Jira", "work", "test_fixture", "{s} 테스트 자료 정리", [
            ("{s} 테스트용 응답 예제를 정리했습니다. 시간 될 때 검토 부탁.", 2, "테스트 자료 검토를 나중에 진행 가능"),
            ("{s} 테스트 데이터 사용법을 문서에 추가했습니다. 참고만 해 주세요.", 1, "사용법 공유이며 조치 요청 없음"),
            ("{s} 새 테스트 예제의 누락 항목을 오늘 안에 확인해 주세요. 배포와 무관합니다.", 3, "오늘 확인 요청이며 배포를 막는 상황은 아님"),
            ("{s} 오래된 테스트 예제를 보관했습니다. 제품 동작은 바뀌지 않습니다.", 1, "예제 보관이며 제품 영향 없음"),
            ("{s} 다음 주부터 쓸 테스트 예제에 코멘트 부탁드립니다.", 2, "다음 주 테스트 준비를 위한 의견 요청"),
        ]),
        ("Linear", "work", "tech_debt", "{s} 기술 부채 정리", [
            ("{s} 중복 함수 정리 제안을 등록했습니다. 다음 스프린트에 논의합니다.", 2, "다음 스프린트 개선 제안"),
            ("{s} 사용하지 않는 코드 목록 공유. 현재 동작에는 영향 없습니다.", 1, "미사용 코드 목록 공유이며 현재 영향 없음"),
            ("{s} 리팩터링 범위를 오늘 계획 회의 전 확인해 주세요. 운영 장애는 아닙니다.", 3, "오늘 계획 논의 전에 범위 확인 필요"),
            ("{s} 개선 아이디어를 백로그에 보관했습니다. 이번 릴리스에는 포함되지 않습니다.", 1, "백로그 보관이며 이번 릴리스와 무관"),
            ("{s} 이름 변경 제안을 검토 부탁드립니다. 내주 중 답변이면 됩니다.", 2, "다음 주까지 답변 가능한 개선 제안"),
        ]),
        ("GitLab", "work", "build_success", "{s} 개발 빌드 결과", [
            ("{s} 개발 빌드 성공. 변경 검토는 다음 작업 때 진행해 주세요.", 2, "개발 빌드가 성공했고 검토를 미룰 수 있음"),
            ("{s} Build succeeded. 변경 사항이 없으며 참고용 결과입니다.", 1, "단순 빌드 성공 보고이며 조치 없음"),
            ("{s} 빌드 산출물 검토를 오늘 중 부탁드립니다. 운영 배포는 예정돼 있지 않습니다.", 3, "오늘 산출물 검토 요청이지만 운영 배포 없음"),
            ("{s} 테스트 브랜치 빌드 완료. 성공 로그를 보관했습니다.", 1, "테스트 빌드 완료 기록"),
            ("{s} 개발 빌드 산출물을 공유했습니다. 다음 주 기능 검증에 사용합니다.", 2, "다음 주 검증용 산출물 공유"),
        ]),
        ("Notion", "work", "ownership", "{s} 담당 범위 안내", [
            ("{s} 문서 관리 담당을 다음 주부터 바꿀 예정입니다. 의견 부탁.", 2, "다음 주 담당 변경에 대한 의견 요청"),
            ("{s} 연락 담당 목록을 갱신했습니다. 작업 권한은 그대로입니다.", 1, "연락 목록만 갱신돼 작업 영향 없음"),
            ("{s} 다음 스프린트 담당 범위를 오늘 중 확인해 주세요.", 3, "오늘 중 다음 스프린트 담당 확인 요청"),
            ("{s} 과거 담당 이력을 문서에 남겼습니다. 추가 조치는 없습니다.", 1, "과거 담당 기록이며 조치 없음"),
            ("{s} 협업 연락 경로를 정리했습니다. 이번 주 중 살펴봐 주세요.", 2, "이번 주에 확인할 협업 정보"),
        ]),
    ],
    "일정/회의": [
        ("Microsoft Teams", "schedule", "room_change", "{s} 참석 장소 변경", [
            ("{s}: 3분 후 시작. 온라인 대신 회의실 B로 와 주세요.", 4, "3분 뒤 참석할 장소가 바뀜"),
            ("{s} 내일 회의실을 B로 변경했습니다. 참석 전에 확인해 주세요.", 3, "내일 참석 장소 변경 확인 필요"),
            ("{s} 다음 주 참석 장소를 정했습니다. 일정표 참고 바랍니다.", 2, "다음 주 장소 안내"),
            ("{s} 10분 후 시작할 장소가 2층으로 바뀌었습니다.", 4, "10분 뒤 회의 참석 장소 변경"),
            ("{s} 이번 주 회의 장소 후보를 공유합니다. 아직 확정 전입니다.", 2, "확정 전 회의 장소 후보 공유"),
        ]),
        ("Outlook", "schedule", "presenter", "{s} 진행 담당 변경", [
            ("{s} 5분 뒤 시작하며 진행을 맡아 주셔야 합니다. 시작 전 준비 부탁.", 4, "5분 뒤 회의 진행을 준비해야 함"),
            ("{s} 내일 진행 담당을 확인해 주세요. 발표 자료는 준비돼 있습니다.", 3, "내일 진행 담당 확인 필요"),
            ("{s} 다음 주 진행 담당 후보를 받습니다. 금요일까지 답변 부탁.", 2, "다음 주 진행 담당 의견 수렴"),
            ("{s} 잠시 후 시작. 원래 진행자 부재로 대신 진행 부탁드립니다.", 4, "곧 시작할 회의의 진행자 대체 필요"),
            ("{s} 지난 회의 진행 기록을 일정에 추가했습니다. 참석 요청은 아닙니다.", 1, "지난 회의 기록이며 참석 요청 없음"),
        ]),
        ("Google Calendar", "schedule", "cancelled", "{s} 일정 취소 안내", [
            ("{s} 4분 뒤 시작 예정이던 회의 취소. 이동하지 않아도 됩니다.", 4, "곧 시작할 회의가 취소돼 즉시 일정 변경 필요"),
            ("{s} 내일 회의를 취소했습니다. 일정에서 제거해 주세요.", 3, "내일 일정 취소 확인 필요"),
            ("{s} 다음 주 회의를 취소합니다. 다음 일정은 별도 공지 예정입니다.", 2, "다음 주 일정 취소"),
            ("{s} 8분 뒤 예정된 회의가 연기됐습니다. 오늘은 참석하지 않습니다.", 4, "8분 뒤 참석 예정인 회의가 연기됨"),
            ("{s} 지난달 취소 이력을 정리했습니다. 현재 일정에는 변경 없습니다.", 1, "과거 취소 기록이며 현재 변경 없음"),
        ]),
        ("Slack", "schedule", "preparation", "{s} 참석 준비 확인", [
            ("{s} 7분 후 시작. 참석용 사전 질문을 지금 제출해 주세요.", 4, "7분 뒤 회의 전에 제출이 필요"),
            ("{s} 내일 논의할 질문을 오늘 안에 보내 주세요.", 3, "내일 회의 준비를 오늘 요청"),
            ("{s} 다음 주 논의할 질문을 모읍니다. 이번 주 중 제출 부탁.", 2, "다음 주 회의를 위한 질문 수집"),
            ("{s} 곧 시작합니다. 공유 자료 접근 확인을 시작 전에 끝내 주세요.", 4, "곧 시작할 회의 전에 자료 접근 확인 필요"),
            ("{s} 지난 회의 질문 모음을 공유합니다. 새 제출 요청은 없습니다.", 1, "지난 질문 모음 공유이며 제출 불필요"),
        ]),
        ("Zoom", "schedule", "access_code", "{s} 참석 인증 안내", [
            ("{s} 6분 뒤 시작. 참가 암호가 재발급됐으니 새 암호로 입장하세요.", 4, "곧 입장할 회의의 참가 암호 변경"),
            ("{s} 내일 참석용 인증 정보를 보냈습니다. 입장 전 확인 바랍니다.", 3, "내일 참석 인증 확인 필요"),
            ("{s} 다음 주 회의의 참가 정보를 미리 안내합니다.", 2, "다음 주 참가 정보 사전 안내"),
            ("{s} 지금 시작합니다. 대기실 승인 담당에게 참석 확인 부탁.", 4, "현재 시작하는 회의 입장 확인 필요"),
            ("{s} 종료된 회의의 참가 정보를 보관했습니다. 다시 참석할 필요는 없습니다.", 1, "종료된 회의 참가 정보 보관"),
        ]),
    ],
    "시스템/보안": [
        ("Windows 보안", "system", "quarantine", "{s} 의심 파일 격리", [
            ("{s}에서 악성 파일 실행 시도가 계속됩니다. 격리 실패로 보호 조치가 필요합니다.", 5, "악성 파일 실행이 이어지고 격리도 실패"),
            ("{s}에서 의심 파일을 차단했습니다. 같은 경로의 실행 시도가 반복돼 확인이 필요합니다.", 4, "차단 뒤에도 의심 실행이 반복됨"),
            ("{s}에서 의심 파일을 격리했습니다. 추가 감염은 없으며 오늘 검사 결과를 확인해 주세요.", 3, "위협은 격리됐고 오늘 결과 확인 요청"),
            ("{s} 정기 검사가 끝났습니다. 격리 내역을 나중에 살펴볼 수 있습니다.", 2, "정기 검사 기록을 나중에 확인 가능"),
            ("{s} 격리 기록을 보관했습니다. 활성 위협과 추가 조치는 없습니다.", 1, "활성 위협 없이 기록만 보관"),
        ]),
        ("Windows", "system", "protection_expiry", "{s} 보호 정책 상태", [
            ("{s} 보안 정책 적용 실패로 위험한 실행을 막지 못합니다. 정책 복구 필요.", 5, "보안 정책 실패로 위험 실행을 막지 못함"),
            ("{s} 보호 정책이 10분 뒤 만료됩니다. 위험한 실행 차단을 유지하려면 갱신해 주세요.", 4, "보호 만료가 임박해 갱신 필요"),
            ("{s} 보호 정책 갱신을 오늘 완료해 주세요. 현재 차단 기능은 정상입니다.", 3, "현재 정상이나 오늘 정책 갱신 요청"),
            ("{s} 보호 정책을 내주에 갱신할 예정입니다. 설정을 미리 확인할 수 있습니다.", 2, "다음 주 정책 갱신 사전 안내"),
            ("{s} 보호 정책 갱신 완료. 차단 기능 정상이며 별도 조치는 없습니다.", 1, "보호 정책 갱신 완료이며 조치 없음"),
        ]),
        ("Windows", "system", "thermal", "{s} 온도 감지", [
            ("{s} 과열로 강제 종료가 반복됩니다. 작업 손실 방지를 위한 조치가 필요합니다.", 5, "반복 강제 종료로 작업 손실 위험"),
            ("{s} 고온 상태가 이어져 종료 경고가 떴습니다. 저장하고 냉각 상태를 확인해 주세요.", 4, "고온 종료 경고로 저장과 냉각 확인 필요"),
            ("{s} 온도가 평소보다 높습니다. 오늘 냉각 상태를 확인해 주세요. 작업은 계속 가능합니다.", 3, "작업 가능하지만 오늘 냉각 확인 요청"),
            ("{s} 온도 추세 기록을 보관했습니다. 당장 조치할 경고는 없습니다.", 2, "온도 추세 참고 기록"),
            ("{s} 온도가 정상 범위로 돌아왔습니다. 추가 조치는 필요 없습니다.", 1, "온도가 정상으로 복구됨"),
        ]),
        ("Microsoft Defender", "system", "credential_leak", "{s} 자격 증명 노출 점검", [
            ("{s}에 저장된 인증 정보로 비인가 접속이 발생했습니다. 인증 정보 폐기 필요.", 5, "노출 자격 증명으로 실제 비인가 접속 발생"),
            ("{s}의 인증 정보가 공개 로그에 노출됐습니다. 사용 중인 정보를 교체해 주세요.", 4, "사용 중인 인증 정보가 공개돼 교체 필요"),
            ("{s} 이전 인증 정보가 로그에 남아 있습니다. 이미 폐기했지만 오늘 기록을 확인해 주세요.", 3, "폐기된 정보지만 오늘 노출 기록 확인 요청"),
            ("{s} 인증 정보 점검 예약을 다음 주로 잡았습니다. 활성 노출 경고는 없습니다.", 2, "활성 노출 없이 다음 주 점검 예정"),
            ("{s} 인증 정보 검사 결과 노출 없음. 추가 조치가 필요하지 않습니다.", 1, "노출 없는 정상 검사 결과"),
        ]),
        ("Windows", "system", "disk_read", "{s} 디스크 읽기 점검", [
            ("{s} 디스크 읽기 오류가 늘어나 파일이 손상됩니다. 작업 중단과 복구 판단 필요.", 5, "파일 손상이 진행되는 디스크 오류"),
            ("{s} 디스크 오류로 현재 파일을 열지 못합니다. 작업을 재개하려면 복구가 필요합니다.", 4, "파일 접근 불가로 현재 작업이 막힘"),
            ("{s} 디스크 검사에서 경고가 발견됐습니다. 파일 접근은 가능하며 오늘 점검해 주세요.", 3, "파일 접근 가능하지만 오늘 점검 요청"),
            ("{s} 정기 디스크 검사를 다음 주에 예약했습니다. 작업 영향은 없습니다.", 2, "작업 영향 없이 다음 주 검사 예정"),
            ("{s} 디스크 검사 완료. 오류 없음.", 1, "디스크 검사에서 오류 없음"),
        ]),
    ],
    "개인 중요": [
        ("은행 앱", "personal", "debit_return", "{s} 납부 반환 안내", [
            ("{s} 납부가 반환됐습니다. 20분 뒤 납부 마감이며 미납 시 서비스가 정지됩니다.", 4, "납부 마감이 임박하고 서비스 정지 위험"),
            ("{s} 결제가 거절돼 필수 서비스가 이미 정지됐습니다. 재납부 확인 바랍니다.", 5, "납부 실패로 필수 서비스가 이미 정지됨"),
            ("{s} 납부가 반환됐습니다. 내일까지 재납부해야 연체가 발생하지 않습니다.", 3, "내일까지 재납부해야 연체를 방지 가능"),
            ("{s} 자동 납부 방식을 다음 주까지 확인해 주세요. 현재 미납은 없습니다.", 2, "현재 미납 없이 다음 주 납부 방식 확인"),
            ("{s} 반환 금액 재납부가 30분 뒤 마감됩니다. 지금 납부 수단을 확인해 주세요.", 4, "재납부 마감이 30분 뒤"),
        ]),
        ("생활 알림", "personal", "home_access", "{s} 방문 확인", [
            ("{s} 담당 기사가 10분 뒤 방문합니다. 지금 출입 안내를 확인해 주세요.", 4, "10분 뒤 방문에 출입 안내 필요"),
            ("{s} 확인을 위한 방문 담당이 건물 앞에서 대기 중입니다. 연락 부탁드립니다.", 4, "방문 담당이 현재 대기 중"),
            ("{s} 담당 기사 방문은 내일입니다. 오늘 중 가능 시간을 알려 주세요.", 3, "내일 방문을 위해 오늘 시간 확인 요청"),
            ("{s} 방문 일정을 다음 주 중 정하려고 합니다. 시간 될 때 답변 주세요.", 2, "다음 주 방문 일정 조율"),
            ("{s} 확인 방문이 15분 앞당겨져 곧 도착합니다. 출입 방법 전달 부탁.", 4, "방문이 앞당겨져 출입 안내가 임박"),
        ]),
        ("카드 앱", "personal", "duplicate_charge", "{s} 중복 청구 확인", [
            ("{s} 중복 청구 취소 요청이 25분 뒤 마감됩니다. 내역 확인 바랍니다.", 4, "중복 청구 취소 마감이 임박"),
            ("{s} 중복 청구가 반복되고 잔액이 계속 감소합니다. 추가 결제 중지 확인 필요.", 5, "반복 중복 청구로 금전 피해 진행"),
            ("{s} 중복 청구 의심 내역을 오늘 안에 확인해 주세요. 추가 청구는 멈췄습니다.", 3, "추가 피해는 멈췄으나 오늘 내역 확인 필요"),
            ("{s} 지난 중복 청구의 환불 내역이 준비됐습니다. 이번 주 중 확인해 주세요.", 2, "환불 결과를 이번 주에 확인 가능"),
            ("{s} 중복 청구 이의 신청 기한이 40분 남았습니다. 확인 후 신청 바랍니다.", 4, "이의 신청 기한이 임박"),
        ]),
        ("공공 서비스", "personal", "support_deadline", "{s} 지원 서류 보완", [
            ("{s} 지원 신청 보완 제출이 20분 뒤 마감됩니다. 누락 서류를 제출해 주세요.", 4, "지원 서류 보완 제출이 임박"),
            ("{s} 지원 서류 오류로 필수 지원이 중단됐습니다. 긴급 재확인 요청이 왔습니다.", 5, "필수 지원 중단으로 긴급 재확인 필요"),
            ("{s} 지원 서류 보완을 내일까지 완료해 주세요.", 3, "내일까지 보완 제출 필요"),
            ("{s} 지원 서류 점검 안내입니다. 다음 주까지 확인하면 됩니다.", 2, "다음 주까지 서류 확인 가능"),
            ("{s} 지원 서류의 최종 승인 확인이 30분 뒤 마감됩니다.", 4, "지원 최종 승인 확인 마감이 임박"),
        ]),
        ("생활 알림", "personal", "family_payment", "{s} 가족 확인 요청", [
            ("{s} 납부를 대신 처리하려면 10분 안에 가족 확인이 필요합니다.", 4, "10분 안에 가족 확인 필요"),
            ("{s} 관련 사칭 요청으로 가족이 계속 송금 중입니다. 지금 연락해 중지 확인 부탁.", 5, "가족의 사칭 피해 송금이 진행 중"),
            ("{s} 가족 공동 납부 동의를 오늘 안에 확인해 주세요.", 3, "오늘 안에 공동 납부 동의 확인 요청"),
            ("{s} 가족 분담 내역을 다음 주까지 확인하면 됩니다.", 2, "다음 주까지 분담 내역 확인 가능"),
            ("{s} 가족 대리 납부 승인 요청이 15분 뒤 종료됩니다.", 4, "가족 대리 납부 승인 마감이 임박"),
        ]),
    ],
    "개인 일반": [
        ("KakaoTalk", "casual", "hobby_poll", "{s} 취향 투표", [
            ("{s} 다음 달 주제 투표를 열었어요. 편할 때 골라 주세요.", 1, "다음 달 취미 주제를 편할 때 선택 가능"),
            ("{s} 취향 설문은 이번 주말까지예요. 참여는 자유입니다.", 2, "이번 주말 선택 참여 설문"),
            ("{s} 투표가 곧 닫혀요. 참여하지 않아도 모임 참석에 영향은 없습니다.", 1, "마감 표현이 있지만 선택 참여이며 불이익 없음"),
            ("{s} 지난 투표 결과를 공유합니다. 답장 필요 없어요.", 1, "취미 투표 결과 공유"),
            ("{s} 후보를 하나 더 추가했어요. 다음 주에 같이 골라 봐요.", 2, "다음 주 취미 후보 선택"),
        ]),
        ("Discord", "casual", "hobby_record", "{s} 기록 공유", [
            ("{s} 지난 활동 후기를 올렸어요. 나중에 읽어 주세요.", 1, "지난 취미 활동 후기 공유"),
            ("{s} 활동 기록 정리에 이번 주 중 의견 부탁드려요.", 2, "이번 주 취미 기록 의견 요청"),
            ("{s} 오늘 사진 공유 창이 곧 닫힙니다. 업로드는 선택이고 기존 사진은 남습니다.", 1, "선택 업로드 마감이며 기존 사진 손실 없음"),
            ("{s} 작년 기록을 다시 올렸어요. 새 일정 안내는 아닙니다.", 1, "과거 기록 재공유이며 일정 변경 없음"),
            ("{s} 기록 태그를 다음 주에 정리해요. 편할 때 제안해 주세요.", 2, "다음 주 취미 기록 정리 제안"),
        ]),
        ("KakaoTalk", "casual", "hobby_chat", "{s} 잡담", [
            ("{s} 관련 재밌는 글을 보냈어요. 답장은 없어도 괜찮아요.", 1, "답장 불필요한 취미 잡담"),
            ("{s} 지난주 얘기 이어서 하고 싶어요. 이번 주에 편할 때 답해 주세요.", 2, "이번 주에 답할 수 있는 잡담"),
            ("{s} 지금 답장해도 되고 나중에 봐도 돼요. 급한 일은 아닙니다.", 1, "지금이라는 표현이 있지만 급한 요청 아님"),
            ("{s} 관련 밈을 공유했어요. 참고만 하세요.", 1, "취미 밈 공유"),
            ("{s} 다음 주에 이야기할 주제를 생각해 주세요. 일정 확정 요청은 아닙니다.", 2, "다음 주 잡담 주제 제안"),
        ]),
        ("네이버 카페", "casual", "community_badge", "{s} 커뮤니티 배지", [
            ("{s} 활동 배지를 받았습니다. 프로필에서 볼 수 있어요.", 1, "취미 활동 배지 알림"),
            ("{s} 배지 설명을 이번 주 중 확인할 수 있습니다. 필수 조치는 아닙니다.", 2, "선택적으로 확인할 배지 설명"),
            ("{s} 배지 이벤트가 5분 뒤 끝납니다. 참여 보상만 종료되며 불이익은 없습니다.", 1, "선택 보상 종료이며 실제 불이익 없음"),
            ("{s} 지난 배지 기록을 다시 표시했습니다.", 1, "과거 취미 배지 기록 표시"),
            ("{s} 다음 주 배지 이름 공모에 의견을 남겨 주세요.", 2, "다음 주 배지 이름 의견 요청"),
        ]),
        ("Discord", "casual", "recommendation", "{s} 추천 목록", [
            ("{s} 관련 추천 콘텐츠를 모았어요. 시간 날 때 보세요.", 1, "취미 콘텐츠를 편할 때 확인 가능"),
            ("{s} 이번 주 추천 목록에 의견을 남겨 주세요.", 2, "이번 주 추천 목록 의견 요청"),
            ("{s} 새 추천을 지금 확인할 수 있어요. 예약이나 신청 마감은 없습니다.", 1, "지금 볼 수 있지만 마감 없는 콘텐츠"),
            ("{s} 저장한 추천 목록을 다시 공유했습니다.", 1, "저장된 추천 목록 공유"),
            ("{s} 다음 주 추천 주제를 함께 골라 주세요.", 2, "다음 주 추천 주제 선택"),
        ]),
    ],
    "광고/홍보": [
        ("쇼핑 앱", "promotion", "flash_sale", "{s} 타임 세일", [
            ("{s} 할인 3분 남음! 선택 구매 이벤트이며 기존 주문에는 영향 없습니다.", 1, "할인 마감이며 실제 의무나 기존 주문 영향 없음"),
            ("{s} 오늘만 할인합니다. 관심 있으면 상품을 둘러보세요.", 1, "선택 구매를 권하는 당일 할인"),
            ("{s} 이번 주 할인 목록입니다. 필요할 때 확인해 주세요.", 2, "이번 주 선택 구매 정보"),
            ("{s} 재고 얼마 안 남았어요! 구매하지 않아도 불이익은 없습니다.", 1, "구매를 유도하는 재고 표현"),
            ("{s} 사전 할인 안내. 다음 주부터 행사합니다.", 2, "다음 주 할인 행사 사전 안내"),
        ]),
        ("이메일", "promotion", "sponsored_review", "{s} 협찬 추천", [
            ("{s} 협찬 리뷰를 지금 확인하세요. 광고 콘텐츠입니다.", 1, "광고 리뷰 확인을 유도함"),
            ("{s} 인기 상품 리뷰 모음. 구매는 선택입니다.", 1, "선택 구매용 상품 광고"),
            ("{s} 이번 주 협찬 콘텐츠 모음을 보내 드립니다.", 2, "이번 주 광고 콘텐츠 안내"),
            ("{s} 리뷰 시청 이벤트 10분 후 종료! 참여하지 않아도 됩니다.", 1, "선택 광고 이벤트 마감"),
            ("{s} 다음 주 신제품 리뷰를 공개합니다. 관심 있으면 구독하세요.", 2, "다음 주 광고 리뷰 구독 권유"),
        ]),
        ("Google Chrome", "promotion", "upgrade_offer", "{s} 업그레이드 제안", [
            ("{s} 새 모델 특가 제안. 지금 바꾸세요! 기존 제품 고장 알림은 아닙니다.", 1, "기존 제품 문제가 아닌 구매 유도"),
            ("{s} 업그레이드 할인 오늘 종료. 교체 의무는 없습니다.", 1, "교체 의무 없는 광고 마감"),
            ("{s} 업그레이드 혜택은 이번 주까지입니다. 필요하면 살펴보세요.", 2, "이번 주 선택 구매 혜택"),
            ("{s} 한정 수량 교체 할인! 기존 서비스 사용에는 영향 없습니다.", 1, "기존 서비스 영향 없는 한정 구매 광고"),
            ("{s} 다음 주 업그레이드 행사를 예고합니다.", 2, "다음 주 구매 행사 안내"),
        ]),
        ("쇼핑 앱", "promotion", "review_reward", "{s} 후기 작성 보상", [
            ("{s} 후기 작성하면 포인트! 참여는 선택입니다.", 1, "선택 후기 작성 보상 광고"),
            ("{s} 포인트 이벤트 2분 후 끝납니다. 미참여로 주문이 취소되지는 않습니다.", 1, "마감은 보상에만 적용되고 주문 영향 없음"),
            ("{s} 이번 주 후기 보상 이벤트를 안내합니다.", 2, "이번 주 선택 보상 이벤트"),
            ("{s} 지금 후기 쓰고 추첨에 참여하세요. 추가 결제는 없습니다.", 1, "추첨 참여를 권하는 홍보"),
            ("{s} 다음 주부터 후기 포인트 이벤트가 시작됩니다.", 2, "다음 주 포인트 행사 사전 안내"),
        ]),
        ("이메일", "promotion", "bundle", "{s} 묶음 구매 혜택", [
            ("{s} 묶음 상품 특별가. 관심 있는 고객만 선택 구매하세요.", 1, "선택 구매용 묶음 할인"),
            ("{s} 묶음 할인 5분 남았습니다. 기존 결제나 납부와 무관합니다.", 1, "기존 납부와 무관한 광고 마감"),
            ("{s} 이번 주 묶음 구매 혜택을 안내합니다.", 2, "이번 주 구매 혜택 광고"),
            ("{s} 지금 함께 구매하면 할인! 필수 구매는 아닙니다.", 1, "지금이라는 표현을 쓰는 선택 구매 광고"),
            ("{s} 다음 주 묶음 상품 행사를 준비 중입니다.", 2, "다음 주 상품 행사 예고"),
        ]),
    ],
    "기타": [
        ("메모장", "misc", "export", "{s} 내보내기 완료", [
            ("{s} 내보내기를 완료했습니다. 원본도 남아 있습니다.", 1, "정상 내보내기 완료이며 원본 유지"),
            ("{s} 내보내기 내역을 이번 주 중 확인할 수 있습니다.", 2, "필요할 때 확인할 내보내기 내역"),
            ("{s} 내보내기 완료 알림을 지금 표시합니다. 오류는 없습니다.", 1, "지금 표시된 정상 완료 상태"),
            ("{s} 이전 내보내기 기록을 다시 보여 줍니다.", 1, "과거 내보내기 기록 표시"),
            ("{s} 내보내기 목록 정리는 다음 주에도 가능합니다.", 2, "다음 주에도 가능한 목록 정리"),
        ]),
        ("Windows", "misc", "theme", "{s} 표시 테마 저장", [
            ("{s} 표시 테마를 저장했습니다. 작업 내용은 바뀌지 않습니다.", 1, "테마 저장이며 작업 내용 영향 없음"),
            ("{s} 테마 설정을 이번 주에 확인할 수 있습니다.", 2, "이번 주 선택적 표시 설정 확인"),
            ("{s} 테마 변경을 지금 적용했습니다. 재시작은 필요 없습니다.", 1, "즉시 적용됐지만 재시작 불필요"),
            ("{s} 이전 테마 기록을 보관했습니다.", 1, "과거 표시 테마 기록 보관"),
            ("{s} 표시 테마는 다음 주에 다시 바꿀 수 있습니다.", 2, "다음 주에도 바꿀 수 있는 표시 옵션"),
        ]),
        ("메모장", "misc", "find", "{s} 검색 완료", [
            ("{s} 검색을 마쳤습니다. 결과 목록이 준비됐습니다.", 1, "정상 검색 완료 상태"),
            ("{s} 저장된 검색 결과를 이번 주 중 확인할 수 있습니다.", 2, "저장된 검색 결과의 선택적 확인"),
            ("{s} 지금 검색 완료. 누락이나 오류 경고는 없습니다.", 1, "정상 검색 완료이며 오류 없음"),
            ("{s} 지난 검색 결과를 다시 표시했습니다.", 1, "지난 검색 결과 재표시"),
            ("{s} 검색 결과 정리는 다음 주에도 가능합니다.", 2, "다음 주에도 가능한 결과 정리"),
        ]),
        ("메모장", "misc", "zoom", "{s} 확대 비율 기록", [
            ("{s} 확대 비율을 저장했습니다. 본문 변경은 없습니다.", 1, "보기 비율 저장이며 본문 변경 없음"),
            ("{s} 보기 설정을 이번 주 중 확인할 수 있습니다.", 2, "이번 주 선택적 보기 설정 확인"),
            ("{s} 확대 비율을 지금 적용했습니다. 파일 손상은 없습니다.", 1, "보기 설정 적용이며 파일 손상 없음"),
            ("{s} 이전 보기 비율 기록을 불러왔습니다.", 1, "이전 보기 비율 기록 표시"),
            ("{s} 확대 설정은 다음 주에 다시 조정할 수 있습니다.", 2, "다음 주에도 조정 가능한 보기 설정"),
        ]),
        ("Windows", "misc", "pin", "{s} 즐겨찾기 등록", [
            ("{s} 즐겨찾기 등록 완료. 추가 요청은 없습니다.", 1, "정상 즐겨찾기 등록 완료"),
            ("{s} 즐겨찾기 목록을 이번 주 중 확인할 수 있습니다.", 2, "즐겨찾기 목록의 선택적 확인"),
            ("{s} 즐겨찾기를 지금 등록했습니다. 작업 중단은 필요 없습니다.", 1, "지금 완료됐지만 작업 중단 불필요"),
            ("{s} 이전 즐겨찾기 기록을 표시했습니다.", 1, "과거 즐겨찾기 기록 표시"),
            ("{s} 즐겨찾기 정리는 다음 주에도 가능합니다.", 2, "다음 주에도 가능한 목록 정리"),
        ]),
    ],
}

CONTEXT_PAIRS = [(0, 2), (0, 4), (1, 2), (0, 3), (0, 2)]
RELATIONS = {
    0: "현재 창 제목에 같은 대상이 있어 직접 관련됨",
    1: "같은 분야의 작업이지만 해당 대상 작업인지는 확인되지 않음",
    2: "현재 다른 분야의 작업을 하고 있어 관련성이 낮음",
    3: "현재 작업 정보가 없어 관련성을 확인할 수 없음",
    4: "최근 앱 사용 정보만으로 같은 대상 작업이라고 단정할 수 없음",
}


def context_for(domain, subject, mode, timestamp):
    direct = {
        "work": ("Code.exe", f"{subject} 핸들러 수정 - Visual Studio Code", ["Code.exe", "WindowsTerminal.exe", "chrome.exe"]),
        "schedule": ("OUTLOOK.EXE", f"{subject} 참석 준비 - Outlook", ["OUTLOOK.EXE", "ms-teams.exe"]),
        "system": ("SystemSettings.exe", f"{subject} 장치 점검 - 설정", ["SystemSettings.exe"]),
        "personal": ("chrome.exe", f"{subject} 내역 확인 - Chrome", ["chrome.exe", "KakaoTalk.exe"]),
        "casual": ("KakaoTalk.exe", f"{subject} 대화방 - KakaoTalk", ["KakaoTalk.exe", "chrome.exe"]),
        "promotion": ("chrome.exe", f"{subject} 구매 비교 - Chrome", ["chrome.exe", "explorer.exe"]),
        "misc": ("Notepad.exe", f"{subject} - 메모장", ["Notepad.exe"]),
    }
    indirect = {
        "work": ("Code.exe", "웹 클라이언트 개발 - Visual Studio Code", ["Code.exe", "chrome.exe"]),
        "schedule": ("OUTLOOK.EXE", "이번 주 회의 일정 - Outlook", ["OUTLOOK.EXE"]),
        "system": ("SystemSettings.exe", "장치 목록 - 설정", ["SystemSettings.exe", "explorer.exe"]),
        "personal": ("chrome.exe", "개인 납부 일정 정리 - Chrome", ["chrome.exe"]),
        "casual": ("KakaoTalk.exe", "취미 모임 목록 - KakaoTalk", ["KakaoTalk.exe"]),
        "promotion": ("chrome.exe", "주변기기 구매 목록 - Chrome", ["chrome.exe"]),
        "misc": ("Notepad.exe", "새 메모 - 메모장", ["Notepad.exe"]),
    }
    if mode == 0:
        process, window, recent = direct[domain]
        relevance = 5 if domain in {"work", "schedule", "system"} else 4
    elif mode == 1:
        process, window, recent = indirect[domain]
        relevance = 3 if domain != "misc" else 2
    elif mode == 2:
        process, window, recent = (("EXCEL.EXE", "가계부 정리 - Excel", ["EXCEL.EXE"])
                                   if domain in {"work", "schedule", "system"}
                                   else ("Code.exe", "그래프 렌더링 개발 - Visual Studio Code", ["Code.exe"]))
        relevance = 1
    elif mode == 3:
        process, window, recent, relevance = "", "", [], 1
    else:
        process, window, recent, relevance = "chrome.exe", "새 탭 - Chrome", ["chrome.exe", "Code.exe", "KakaoTalk.exe"], 2
    return {
        "active_process": process, "window_title": window,
        "last_updated": (timestamp - timedelta(seconds=2)).isoformat().replace("+00:00", "Z"),
        "duration_seconds": 0 if mode == 3 else 15 + mode * 23,
        "recent_processes": recent,
    }, relevance


def write_samples(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8")


def generate(base=BASE_DATASET, output=OUTPUT_PATH, combined=COMBINED_PATH,
             holdout=HOLDOUT_PATH, report_path=ROOT / "audit" / "targeted_5000_audit.json"):
    if output.resolve() == base.resolve() or combined.resolve() in {base.resolve(), output.resolve()}:
        raise ValueError("output paths must not overwrite the source dataset")
    original = load_samples(base)
    original_lineage = json.loads(base.with_suffix(".lineage.json").read_text(encoding="utf-8"))["items"]
    reserved = original + (load_samples(holdout) if holdout.exists() else [])
    reserved_keys = {notification_key(sample) for sample in reserved}
    reserved_texts = {(s.notification.title, s.notification.body) for s in reserved}
    rows, lineage = [], []
    for category in CATEGORIES:
        if len(FAMILIES[category]) != 5:
            raise ValueError("five new families required per category")
        for app, domain, family, title_template, states in FAMILIES[category]:
            if len(states) != 5 or len(SUBJECTS[domain]) != 5:
                raise ValueError("five semantic states and subjects required")
            subjects = PERSONAL_SUBJECTS[family] if domain == "personal" else SUBJECTS[domain]
            for subject_index, subject in enumerate(subjects):
                for state_index, (body_template, urgency, evidence) in enumerate(states):
                    title, body = title_template.format(s=subject), body_template.format(s=subject)
                    if (title, body) in reserved_texts:
                        raise ValueError("new notification overlaps reserved text")
                    pair_id = f"targeted:{category}:{family}:{subject_index}:{state_index}"
                    for mode in CONTEXT_PAIRS[(state_index + subject_index) % len(CONTEXT_PAIRS)]:
                        timestamp = datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=len(rows)*17)
                        context, relevance = context_for(domain, subject, mode, timestamp)
                        diversity = int(hashlib.sha256(f"{pair_id}:{mode}".encode()).hexdigest()[:8], 16)
                        if mode != 3:
                            context["duration_seconds"] = [12, 43, 95, 180, 600][diversity % 5]
                            context["recent_processes"] = context["recent_processes"][:diversity % 4]
                        sample = FilteringSample.model_validate({
                            "notification": {"id": f"targeted_{len(rows)+1:04d}", "app_name": app,
                                             "sender": "", "title": title, "body": body,
                                             "timestamp": timestamp.isoformat().replace("+00:00", "Z")},
                            "context": context,
                            "label": {"urgency_score": urgency, "relevance_score": relevance,
                                      "category": category,
                                      "ai_summary_reason": f"{evidence}. {RELATIONS[mode]}.",},
                        })
                        parse_model_output(sample.label.model_dump_json())
                        if notification_key(sample) in reserved_keys:
                            raise ValueError("notification overlaps reserved data")
                        rows.append(sample)
                        lineage.append({"id": sample.notification.id, "scenario": f"{category}:targeted_{family}",
                                        "subject_index": subject_index, "detail_index": state_index,
                                        "context_mode": mode, "pair_id": pair_id})
    all_rows = original + rows
    all_lineage = original_lineage + lineage
    ids = [s.notification.id for s in all_rows]
    if len(ids) != len(set(ids)) or {s.notification.id for s in rows} & {s.notification.id for s in reserved}:
        raise ValueError("duplicate sample ID")
    if {e["id"] for e in all_lineage} != set(ids) or len(all_lineage) != len(ids):
        raise ValueError("source lineage does not match dataset")
    audit = audit_samples(all_rows)
    if audit["contradictory_notification_groups"] or audit["conflicting_identical_inputs"]:
        raise ValueError("contradictory labels detected")
    write_samples(output, rows)
    write_samples(combined, all_rows)
    metadata = {
        "status": "provisional authored labels; independent review pending",
        "method": "40 new families x five subjects x five semantic states x two contrasting contexts",
        "source_reference": "Codex-authored scenarios; public themes and private sample style only; no raw text copied",
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
    }
    for path, items in ((output, lineage), (combined, all_lineage)):
        path.with_suffix(".lineage.json").write_text(json.dumps({**metadata, "items": items}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    report = {
        "added_count": len(rows), "added_unique_notification_texts": len({(s.notification.title,s.notification.body) for s in rows}),
        "added_families": len({e["scenario"] for e in lineage}),
        "combined_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
        "combined_families": len({e["scenario"] for e in all_lineage}),
        "added_category_counts": dict(Counter(s.label.category for s in rows)), "combined_audit": audit,
        "review_status": "automatic checks passed; semantic labels require independent review",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--combined", type=Path, default=COMBINED_PATH)
    args = parser.parse_args()
    print(json.dumps(generate(args.base,args.output,args.combined),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
