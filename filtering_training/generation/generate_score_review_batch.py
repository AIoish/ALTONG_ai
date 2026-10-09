"""용도: 필터링 점수 보강 검수표와 다양한 원문 후보 생성·수정.
생성일: 2026-10-03
"""

import argparse
import hashlib
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
import random

from filtering_training.common.dataset import load_samples
from filtering_training.common.score_tasks import unique_urgency_samples
from filtering_training.common.score_tasks import HistorySample, HistoryContext
from filtering_training.preparation.prepare_score_experiment import digest, write_json, write_lines
from src.filtering.schema import FilteringSample
from src.filtering.policy import POLICY_VERSION, decision_label
from src.filtering.prompt import parse_model_output


# Candidate cases are authored here; links describe an explicitly inspected family,
# not all possible semantic similarities. All linked old originals must be train-only.
CASES = [
    {"app":"Slack","sender":"보안 담당","title":"#키-관리",
     "body":"운영 API 비밀 키가 공개 저장소에 올라간 것을 확인했습니다. 외부에서 사용할 수 있으니 지금 키를 폐기하고 새 키로 바꿔 주세요.",
     "category":"시스템/보안","urgency":5,"relevance":5,
     "context":("Code.exe","운영 API 키 폐기와 재발급 - Visual Studio Code"),
     "urgency_reason":"비밀 키 공개가 확인됐고 즉시 폐기·교체 요청이 있습니다",
     "relevance_reason":"현재 수행하는 운영 API 키 폐기·재발급에 정확히 연결됩니다",
     "family":"credential_exposure","train_links":["v3_37"]},
    {"app":"KakaoTalk.exe","sender":"배우자","title":"배우자",
     "body":"아이가 숨쉬기 힘들어해서 119 불렀어. 지금 통화 받아 줘. 구급대가 보호자랑 확인할 게 있대.",
     "category":"개인 중요","urgency":5,"relevance":5,
     "context":("chrome.exe","아이 응급 상황 보호자 연락 확인 - Chrome"),
     "urgency_reason":"호흡 곤란과 구급대의 보호자 연락 요청이 명시됐습니다",
     "relevance_reason":"현재 아이 응급 상황의 보호자 연락을 확인하는 행동과 연결됩니다",
     "family":"family_emergency_contact","train_links":["v3_09"]},
    {"app":"Microsoft Teams","sender":"채용 담당","title":"면접 입장 확인",
     "body":"면접관이 입장해 기다리고 있습니다. 접속 오류가 있었으니 지금 새 링크로 들어와 주세요. 3분 뒤에는 이번 면접 입장을 마감합니다.",
     "category":"일정/회의","urgency":4,"relevance":5,
     "context":("ms-teams.exe","오늘 채용 면접 대기실 입장 - Teams"),
     "urgency_reason":"이미 시작한 면접이며 3분 뒤 입장이 마감됩니다",
     "relevance_reason":"현재 면접 대기실에 입장하는 행동에 필요한 안내입니다",
     "family":"meeting_access","train_links":["v3_29"]},
    {"app":"KakaoTalk.exe","sender":"스터디 동료","title":"스터디 동료",
     "body":"스페인어 듣기에서 동사 끝이 잘 안 들릴 때, 앞에 나온 주어랑 같이 확인하면 구분하기 쉬워요. 연습 방법 정리해 뒀으니 필요할 때 봐요.",
     "category":"개인 일반","urgency":2,"relevance":4,
     "context":("chrome.exe","스페인어 동사 변화 듣기 연습 - Chrome"),
     "urgency_reason":"기한이나 즉시 대응 요청이 없는 참고 학습 방법입니다",
     "relevance_reason":"현재 듣기 연습에 직접 도움을 주지만 정확한 문제나 녹음을 지정하지 않습니다",
     "family":"language_learning_support","train_links":[]},
    {"app":"Slack","sender":"설계 동료","title":"#모터-조립",
     "body":"M7 모터 조립 순서를 확인할 수 있는 분해 뷰를 올렸어요. 부품 이름도 표시해 뒀으니 조립도 작업할 때 참고하세요. 급하게 확인할 건 아니에요.",
     "category":"일반 업무","urgency":2,"relevance":5,
     "context":("SLDWORKS.exe","M7 모터 조립 순서 도면 작성 - SOLIDWORKS"),
     "urgency_reason":"즉시 확인이 필요하지 않다고 명시한 참고 자료 공유입니다",
     "relevance_reason":"현재 도면을 작성하는 정확한 M7 모터의 조립 순서 자료입니다",
     "family":"assembly_reference","train_links":[]},
    {"app":"KakaoTalk.exe","sender":"문구몰 채널","title":"문구몰 채널",
     "body":"스페인어 단어가 적힌 머그컵 출시 ☕ 책상 위에 올려 두고 분위기 내 보세요. 이번 주 무료 배송!",
     "category":"광고/홍보","urgency":1,"relevance":3,
     "context":("chrome.exe","스페인어 동사 변화 듣기 연습 - Chrome"),
     "urgency_reason":"선택적인 상품 홍보로 대응할 필요가 없습니다",
     "relevance_reason":"언어 주제는 같지만 현재 동사 듣기 연습에 직접 주는 도움은 불분명합니다",
     "family":"language_merchandise","train_links":[]},
    {"app":"KakaoTalk.exe","sender":"유빈","title":"유빈",
     "body":"스페인어 동사 변화 외우다 내 표정도 시시각각 변하는 중ㅋㅋ",
     "category":"개인 일반","urgency":1,"relevance":3,
     "context":("chrome.exe","스페인어 동사 변화 듣기 연습 - Chrome"),
     "urgency_reason":"학습에 관한 농담이며 대응 요청이 없습니다",
     "relevance_reason":"현재 연습과 주제는 같지만 문제 풀이에 주는 도움은 불분명합니다",
     "family":"language_learning_joke","train_links":[]},
    {"app":"KakaoTalk.exe","sender":"러닝샵 채널","title":"러닝샵 채널",
     "body":"긴급 쿠폰 알림! 러닝화 전 제품 10% 쿠폰이 5분 뒤 끝나요. 구매하실 분만 적용해 주세요 👟",
     "category":"광고/홍보","urgency":1,"relevance":4,
     "context":("chrome.exe","러닝화 무게와 최종 구매 가격 비교 - Chrome"),
     "urgency_reason":"짧은 쿠폰 기한이지만 선택적인 구매이며 피해나 필수 대응이 없습니다",
     "relevance_reason":"현재 러닝화 구매 비교의 최종 가격을 확인하는 목적에 도움이 됩니다",
     "family":"footwear_promotion","train_links":["v3_13","v3_14","v3_57"]},
]


EXPANSION_CASES = [
    {"app":"KakaoTalk.exe","sender":"가공 스터디","title":"가공 스터디",
     "body":"절삭 조건 공부하다 내 집중력도 같이 깎이는 중ㅋㅋ",
     "category":"개인 일반","urgency":1,"relevance":3,"context":("FreeCAD.exe","알루미늄 절삭 조건과 공구 경로 검토 - FreeCAD"),
     "urgency_reason":"농담이며 대응 요청이나 기한이 없습니다","relevance_reason":"가공 주제는 같지만 현재 절삭 조건을 결정하는 데 도움은 불분명합니다","family":"machining_topic_joke","train_links":[]},
    {"app":"KakaoTalk.exe","sender":"제조 장비몰","title":"장비몰 소식",
     "body":"3D 프린터용 PLA 필라멘트 새 색상 출시! 이번 주 묶음 구매 할인합니다.",
     "category":"광고/홍보","urgency":1,"relevance":2,"context":("FreeCAD.exe","알루미늄 절삭 조건과 공구 경로 검토 - FreeCAD"),
     "urgency_reason":"선택적인 구매 홍보입니다","relevance_reason":"제조 분야는 같지만 알루미늄 절삭 조건 검토에 직접 도움이 되지 않습니다","family":"manufacturing_adjacent_ad","train_links":[]},
    {"app":"Slack","sender":"가공 동료","title":"#공정-자료",
     "body":"알루미늄 가공에서 칩 배출이 막힐 때 절삭 조건을 조정하는 순서를 정리했어요. 시간 날 때 검토해 보세요.",
     "category":"일반 업무","urgency":2,"relevance":4,"context":("FreeCAD.exe","알루미늄 절삭 조건과 공구 경로 검토 - FreeCAD"),
     "urgency_reason":"나중에 확인할 참고 자료입니다","relevance_reason":"현재 절삭 조건을 검토하는 목적에 직접 도움이 되는 절차입니다","family":"machining_reference","train_links":[]},
    {"app":"Slack","sender":"치구 설계 담당","title":"#X9-치구",
     "body":"X9 치구의 기준핀 위치가 표시된 조립도를 올렸습니다. 기준핀 배치할 때 이 도면을 참고하세요. 급한 수정은 아닙니다.",
     "category":"일반 업무","urgency":2,"relevance":5,"context":("SLDWORKS.exe","X9 치구 기준핀 위치 배치 - SOLIDWORKS"),
     "urgency_reason":"급하지 않은 작업 참고 자료입니다","relevance_reason":"현재 배치 중인 X9 치구의 정확한 기준핀 위치 자료입니다","family":"fixture_datum_reference","train_links":[]},
    {"app":"Slack","sender":"자재 담당","title":"#자재-확인",
     "body":"오늘 오후 출고분에 넣을 고정 볼트 수량이 아직 확인되지 않았어요. 점심 이후 창고 재고를 확인해 답해주세요.",
     "category":"일반 업무","urgency":3,"relevance":1,"context":("chrome.exe","사진 편집 강의 학습 - Chrome"),
     "urgency_reason":"당일 출고 전에 확인해야 하지만 즉시 대응 시점은 아닙니다","relevance_reason":"현재 사진 편집 학습과 무관합니다","family":"inventory_confirmation","train_links":[]},
    {"app":"Slack","sender":"검사 담당","title":"#R8-검사",
     "body":"R8 부품의 검사 기록에서 측정 단위가 빠져 있습니다. 내일 오전 제출 전에 단위 표기를 확인해 회신해주세요.",
     "category":"일반 업무","urgency":3,"relevance":5,"context":("EXCEL.EXE","R8 부품 검사 기록 단위 표기 검토 - Excel"),
     "urgency_reason":"제출 전 확인이 필요하지만 즉시 작업을 중단할 요청은 아닙니다","relevance_reason":"현재 검토하는 정확한 R8 검사 기록의 단위 표기 요청입니다","family":"inspection_unit_confirmation","train_links":[]},
    {"app":"Microsoft Teams","sender":"면담 담당","title":"면담 연결 확인",
     "body":"예정된 면담을 시작했습니다. 접속이 끊겨 새 입장 링크를 보냅니다. 지금 들어와 주세요. 5분 뒤에는 다음 면담을 시작합니다.",
     "category":"일정/회의","urgency":4,"relevance":5,"context":("ms-teams.exe","예약 면담 재접속 - Teams"),
     "urgency_reason":"현재 진행 중인 면담에 즉시 입장할 필요가 있습니다","relevance_reason":"현재 면담에 재접속하는 정확한 행동과 연결됩니다","family":"meeting_access","train_links":["v3_29"]},
    {"app":"Slack","sender":"보안 담당","title":"#키-교체",
     "body":"외부 공개 저장소에 배포용 서명 비밀키가 올라간 것을 확인했습니다. 해당 키를 즉시 폐기하고 교체해야 합니다.",
     "category":"시스템/보안","urgency":5,"relevance":5,"context":("Code.exe","배포 서명 비밀키 폐기와 교체 - Visual Studio Code"),
     "urgency_reason":"공개된 비밀키를 즉시 폐기하고 교체할 대응이 필요합니다","relevance_reason":"현재 작업하는 배포 서명 키의 정확한 폐기·교체 행동과 연결됩니다","family":"credential_exposure","train_links":["v3_37"]},
]


URGENT_CASES = [
    {"app":"KakaoTalk.exe","sender":"가족","title":"응급 보호자 연락",
     "body":"할머니가 갑자기 의식을 잃어서 구급대가 도착했어. 보호자 확인이 필요하다고 하니 지금 전화 받아 줘.",
     "category":"개인 중요","urgency":5,"relevance":1,"context":("Code.exe","사진 업로드 확장자 검사 수정 - Visual Studio Code"),
     "urgency_reason":"의식 소실로 구급대가 현장에 있고 즉시 보호자 연락이 필요합니다","relevance_reason":"현재 사진 업로드 검사 수정과 무관합니다","family":"family_emergency_contact","train_links":["v3_09"]},
    {"app":"건물 알림","sender":"안전 담당","title":"실제 화재 대피 안내",
     "body":"현재 입주 건물 지하에서 실제 화재가 확인됐습니다. 훈련이 아닙니다. 건물에 있는 분들은 즉시 대피 안내를 따라 주세요.",
     "category":"개인 중요","urgency":5,"relevance":1,"context":("",""),
     "urgency_reason":"실제 화재가 확인됐고 건물에 있는 사람의 즉시 대피가 필요합니다","relevance_reason":"빈 창에는 현재 작업을 확인할 정보가 없습니다","family":"confirmed_building_fire","train_links":[]},
    {"app":"은행 앱","sender":"이상 거래 감시","title":"연속 이체 확인",
     "body":"등록하지 않은 수취인에게 80만원씩 이체가 연속 실행되고 있습니다. 본인 요청이 아니라면 지금 거래 중지 접수를 해주세요.",
     "category":"개인 중요","urgency":5,"relevance":1,"context":("Premiere.exe","공연 영상 장면 연결 편집 - Premiere"),
     "urgency_reason":"예상하지 못한 수취인에게 금전 이체가 현재 반복 실행되고 있습니다","relevance_reason":"현재 공연 영상 편집과 무관합니다","family":"unrecognized_financial_transaction","train_links":["v3_42"]},
    {"app":"학교 포털","sender":"장학 접수 담당","title":"장학 신청 보완 마감",
     "body":"신청한 장학금의 필수 동의서가 빠졌습니다. 보완 마감까지 8분 남았으며 이 시간이 지나면 이번 신청은 접수되지 않습니다.",
     "category":"개인 중요","urgency":5,"relevance":5,"context":("chrome.exe","장학 신청 필수 동의서 제출 - 학교 포털"),
     "urgency_reason":"이미 신청한 필수 서류가 누락됐고 8분 뒤 접수 기회를 잃습니다","relevance_reason":"현재 제출하는 정확한 장학 신청 동의서와 연결됩니다","family":"application_deadline","train_links":["v3_43"]},
    {"app":"시험 안내","sender":"감독관","title":"실기시험 입실 확인",
     "body":"예약한 실기시험의 본인 확인이 시작됐습니다. 지금 확인 링크에 접속해 주세요. 3분 뒤 입실이 마감됩니다.",
     "category":"일정/회의","urgency":5,"relevance":1,"context":("",""),
     "urgency_reason":"예약한 시험의 확인이 이미 시작됐고 3분 뒤 입실 기회를 잃습니다","relevance_reason":"빈 창에는 현재 작업을 확인할 정보가 없습니다","family":"mandatory_event_access","train_links":["v3_29","v3_31"]},
    {"app":"KakaoTalk.exe","sender":"가족","title":"검진 접수 준비",
     "body":"내일 검진 접수할 때 가져갈 서류 목록을 보냈어. 오늘 저녁까지 빠진 게 있는지만 확인해 줘. 지금 당장 볼 필요는 없어.",
     "category":"개인 중요","urgency":3,"relevance":4,"context":("Notion.exe","가족 검진 준비물 점검 - Notion"),
     "urgency_reason":"오늘 저녁까지 확인할 요청이고 즉시 대응은 필요하지 않습니다","relevance_reason":"현재 검진 준비물 점검에 직접 필요한 서류 목록입니다","family":"health_preparation_checklist","train_links":[]},
    {"app":"건물 알림","sender":"시설 담당","title":"소화 설비 점검 일정",
     "body":"내일 오후 소화 설비 정기 점검이 예정돼 있습니다. 오늘 저녁까지 출입 가능한 시간을 알려주세요. 현재 고장이나 화재는 없습니다.",
     "category":"개인 중요","urgency":3,"relevance":1,"context":("EXCEL.EXE","자전거 주행 거리 기록 정리 - Excel"),
     "urgency_reason":"예정된 점검을 위한 당일 회신이며 현재 사고나 고장은 없습니다","relevance_reason":"현재 자전거 기록 정리와 무관합니다","family":"scheduled_facility_inspection","train_links":["v3_10"]},
    {"app":"Slack","sender":"현장 담당","title":"대여 장비 인계 확인",
     "body":"대여 장비 기사님이 지금 인계 장소에 도착해 기다리고 있습니다. 인수 담당자가 누구인지 바로 답해주세요. 장비 고장이나 안전 사고는 없습니다.",
     "category":"긴급 업무","urgency":4,"relevance":1,"context":("zotero.exe","논문 인용 형식 정리 - Zotero"),
     "urgency_reason":"현재 진행 중인 인계가 답변을 기다리지만 사고나 임박한 마감은 없습니다","relevance_reason":"현재 논문 인용 형식 정리와 무관합니다","family":"active_equipment_handoff","train_links":[]},
    {"app":"쇼핑 앱","sender":"캠핑용품몰","title":"구매 쿠폰 종료",
     "body":"긴급 할인! 캠핑용 랜턴 10% 쿠폰이 5분 뒤 종료됩니다. 구매는 선택이며 기존 주문에는 영향이 없습니다.",
     "category":"광고/홍보","urgency":2,"relevance":4,"context":("chrome.exe","캠핑 랜턴 구매 예산과 최종 가격 비교 - Chrome"),
     "urgency_reason":"할인 종료가 임박했지만 선택적 구매이고 기존 주문 피해는 없습니다","relevance_reason":"현재 구매 비용 비교에 할인 조건이 직접 도움이 됩니다","family":"optional_purchase_coupon","train_links":["v3_13","v3_14","v3_53","v3_57"]},
    {"app":"KakaoTalk.exe","sender":"스터디 친구","title":"시험 공부 잡담",
     "body":"긴급ㅋㅋ 실기시험 공부하다가 커피부터 다 마셔 버림. 집중력 충전 쿠폰 어디 없냐",
     "category":"개인 일반","urgency":1,"relevance":3,"context":("chrome.exe","실기시험 평가 항목 복습 - Chrome"),
     "urgency_reason":"긴급이라는 표현이 들어간 농담이며 실제 대응 요청이 없습니다","relevance_reason":"시험 공부 주제는 같지만 현재 평가 항목 복습에 직접 도움은 없습니다","family":"exam_topic_joke","train_links":["v3_12","v3_28","v3_49"]},
]


def build_candidates(expansion=False, urgent=False):
    samples = []
    cases = URGENT_CASES if urgent else EXPANSION_CASES if expansion else CASES
    prefix = "sep04" if urgent else "sep03" if expansion else "sep02"
    timestamp = "2026-10-04T12:00:00Z" if expansion or urgent else "2026-10-03T12:00:00Z"
    for index, case in enumerate(cases):
        # Sharing a window inside this batch creates a train component deliberately;
        # no populated source-dataset window is reused.
        other = cases[(index+1)%len(cases)]["context"] if index not in (3,5,6) else cases[4]["context"]
        variants = (
            ("related",case["context"],case["relevance"]),
            ("unrelated",other,1), ("empty",("",""),1))
        if expansion:
            variants = [variants[0]]
            if index == 6: variants.append(("empty",("",""),1))
            if index == 7: variants.append(("unrelated",cases[4]["context"],1))
        if urgent:
            variants = [("review",case["context"],case["relevance"])]
            if index in (0,2,3):
                variants.append(("empty",("",""),1))
        for role, (process,window), score in variants:
            reason = case["relevance_reason"] if role in ("related","review") else (
                "현재 창의 작업과 무관합니다" if role=="unrelated"
                else "현재 작업 정보가 없어 관련성을 확인할 수 없습니다")
            samples.append(FilteringSample.model_validate({
                "notification":{"id":f"{prefix}_{index+1:02d}_{role}","app_name":case["app"],
                    "sender":case["sender"],"title":case["title"],"body":case["body"],"timestamp":timestamp},
                "context":{"active_process":process,"window_title":window,"last_updated":timestamp,
                    "duration_seconds":60 if process else 0,"recent_processes":[]},
                "label":{"urgency_score":case["urgency"],"relevance_score":score,
                    "category":case["category"],"ai_summary_reason":case["urgency_reason"]+"; "+reason+"."}}))
    unique_urgency_samples(samples)
    return samples


def generate(source,output,expansion=False,urgent=False):
    if output.exists():
        raise ValueError("do not overwrite an earlier candidate batch")
    manifest=json.loads((source/"prepared/manifest.json").read_text(encoding="utf-8"))
    if digest(source/"dataset.jsonl")!=manifest["source_sha256"] or manifest["policy_version"]!=POLICY_VERSION:
        raise ValueError("source dataset or policy changed")
    original=load_samples(source/"dataset.jsonl")
    source_windows={(s.context.active_process,s.context.window_title) for s in original if s.context.active_process}
    normalize=lambda text:re.sub(r"\s+","",text).casefold()
    source_bodies={normalize(s.notification.body) for s in original}
    cases = URGENT_CASES if urgent else EXPANSION_CASES if expansion else CASES
    prefix = "sep04" if urgent else "sep03" if expansion else "sep02"
    created_date = "2026-10-04" if expansion or urgent else "2026-10-03"
    samples=build_candidates(expansion,urgent)
    approved_path = source.parent/"v3_score_review_02_approved_01/dataset.jsonl"
    if (expansion or urgent) and approved_path.exists():
        approved = load_samples(approved_path)
        if {normalize(s.notification.body) for s in samples} & {normalize(s.notification.body) for s in approved}:
            raise ValueError("candidate body exactly overlaps approved supplement")
    additional_sources = []
    if urgent:
        for path in (source.parent/"v3_expansion_review_01_approved/dataset.jsonl",
                     source.parent/"v4_diverse_5000_candidates_03/candidates.jsonl"):
            if path.exists():
                known = load_samples(path)
                if {normalize(s.notification.body) for s in samples} & {normalize(s.notification.body) for s in known}:
                    raise ValueError("urgent candidate duplicates an earlier original")
                source_windows.update((s.context.active_process,s.context.window_title) for s in known if s.context.active_process)
                additional_sources.append({"path":str(path),"sha256":digest(path)})
    if any((s.context.active_process,s.context.window_title) in source_windows for s in samples if s.context.active_process):
        raise ValueError("candidate window overlaps an existing split")
    if any(normalize(s.notification.body) in source_bodies for s in samples):
        raise ValueError("candidate body exactly overlaps source")
    train_messages=set(manifest["splits"]["train"]["message_ids"])
    if any(not set(case["train_links"]).issubset(train_messages) for case in cases):
        raise ValueError("an explicitly linked family is not train-only")
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl",[s.model_dump(mode="json") for s in samples])
    lines=[f"<!-- 용도: 기존 보류 자료와 창·지정 유사 그룹을 분리한 보강 후보 검수.\n생성일: {created_date} (Asia/Seoul)\n상태: 사용자 미승인, 학습 미사용. -->\n",
           "# 건강·안전·금전·필수 일정 긴급 경계 검수\n" if urgent else "# 5천 원문 확장 전 생성 기준 검수\n" if expansion else "# 두 번째 보강 검수 묶음\n",
           f"원문 {len(cases)}개·문맥 {len(samples)}건입니다. 집중 모드 ON, 빈 창은 쉬는 상태로 추정하지 않습니다.\n",
           "긴급도와 카테고리는 문맥이 달라져도 유지합니다. 숫자와 설명은 제안이며 승인 전에는 확정하지 않습니다.\n"]
    for index,case in enumerate(cases,1):
        lines += [f"## {index}. {case['category']}\n",f"앱: {case['app']} / 발신자: {case['sender']} / 제목: {case['title']}\n",
                  f"원문: {case['body']}\n",f"긴급도 제안: **{case['urgency']}** — {case['urgency_reason']}.\n",
                  "| 문맥 | 프로세스 | 창 제목 | 관련도 | 정책 |","| --- | --- | --- | --- | --- |"]
        for sample in [s for s in samples if s.notification.id.startswith(f"{prefix}_{index:02d}_")]:
            role=sample.notification.id.rsplit("_",1)[-1]
            lines.append(f"| {role} | {sample.context.active_process or '(빈 값)'} | {sample.context.window_title or '(빈 값)'} | {sample.label.relevance_score} | {decision_label(sample.label.urgency_score,sample.label.relevance_score)} |")
        lines += ["",case["relevance_reason"]+".\n"]
    lines += ["## 격리 확인\n","기존 원문과 정규화 본문 완전 일치 없음, 정보 창 완전 일치 없음. 명시한 유사 계열은 기존 학습 분할에만 연결됩니다. 모든 의미 유사성의 자동 검증을 주장하지 않습니다.\n",
              ("계열 연결은 manifest의 train_links에 기록합니다. 검증 오류 원문을 복사한 사례가 아닙니다.\n" if urgent else "7: 회의 입장, 8: 자격 정보 노출은 기존 학습 계열에 연결합니다. 나머지는 별도 작성 계열입니다.\n" if expansion else "1: 자격 정보 노출, 2: 가족 응급 연락, 3: 회의 입장, 8: 운동화 홍보 계열은 기존 학습 자료와 연결해 관리합니다. 나머지는 별도 작성 계열입니다.\n"),
              "승인 후에도 후보 묶음 전체를 학습 분할에만 추가하고 기존 검증·테스트 ID를 유지합니다. 평가용 독립 사례로 사용하지 않습니다.\n"]
    (output/"review.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"manifest.json",{"purpose":"small targeted training-candidate review; not evaluation data",
        "created_date":created_date,"status":"unapproved_synthetic_candidates","originals":len(cases),"rows":len(samples),
        "source_dataset_sha256":manifest["source_sha256"],"source_manifest_sha256":digest(source/"prepared/manifest.json"),
        "candidate_sha256":digest(output/"candidates.jsonl"),"review_sha256":digest(output/"review.md"),
        "policy_version":POLICY_VERSION,"exact_body_overlap":0,"populated_window_overlap":0,
        "additional_overlap_sources":additional_sources,"human_reviewed_originals":0,"training_used":False,"test_inference_used":False,
        "families":{f"{prefix}_{i:02d}":{"family":c["family"],"train_links":c["train_links"]} for i,c in enumerate(cases,1)},
        "urgency_counts":dict(Counter(s.label.urgency_score for s in samples)),
        "relevance_counts":dict(Counter(s.label.relevance_score for s in samples)),
        "note":("independently authored non-digital urgent boundaries; do not mass-expand or train before review" if urgent else "scale target: 5000 diverse notification originals; context rows counted separately; pending human review" if expansion else "counterfactual triples still contain two relevance=1 rows per original; balancing is a separate experiment"),
        "files":{"candidates.jsonl":"provisional notification/context/label rows","review.md":"complete human review table"}})
    return samples


# Domain, process, tool, personal/work, and five different task/artifact/field triples.
# These are authored scenario ingredients, not 5000 independently authored texts.
BULK_DOMAINS = [
    ("소프트웨어", "Code.exe", "Visual Studio Code", False, "검색 캐시 점검|검색 응답 설정|캐시 만료 값;로그 수집 정책 검토|수집 필터 설정|제외 경로;파일 업로드 화면 수정|업로드 검증 규칙|허용 확장자;지도 검색 기능 개발|위치 검색 설정|검색 반경;접근성 단축키 구현|키보드 이동 규칙|포커스 순서"),
    ("데이터 분석", "EXCEL.EXE", "Excel", False, "배송 시간 분석|배송 소요 시간표|시간대 코드;설문 응답 집계|응답 집계표|복수 선택 구분;도서 대출 추세 분석|대출 통계표|집계 기간;관람객 동선 분석|입장 구역표|구역 분류;생산 불량 유형 분석|불량 유형표|검사 기준"),
    ("화면 디자인", "Figma.exe", "Figma", False, "예약 화면 디자인|예약 입력 시안|버튼 배치;도서 검색 화면 개선|검색 결과 시안|정보 우선순위;전시 안내 화면 제작|안내 화면 시안|안내 색상;회원 가입 흐름 설계|가입 단계 시안|필수 입력 표시;재고 관리 화면 설계|재고 목록 시안|상태 배지"),
    ("영상 편집", "Premiere.exe", "Premiere", False, "제품 설명 영상 편집|제품 소개 편집본|장면 연결;인터뷰 자막 교정|인터뷰 자막 파일|발화 시점;공연 기록 영상 편집|공연 기록 편집본|카메라 전환;행사 안내 영상 제작|행사 안내 편집본|오프닝 길이;온라인 강의 장면 정리|강의 편집본|챕터 구간"),
    ("음향 제작", "audacity.exe", "Audacity", False, "강의 녹음 잡음 제거|강의 음성 파일|잡음 구간;팟캐스트 배경음 조정|팟캐스트 믹스 파일|음량 균형;공연 음원 트랙 정리|공연 트랙 목록|트랙 순서;안내 방송 음성 편집|안내 음성 파일|문장 간격;낭독 녹음 구간 편집|낭독 녹음 파일|숨소리 구간"),
    ("문서 작성", "WINWORD.EXE", "Word", False, "신입 직원 안내서 작성|입사 안내 초안|담당 연락처;장비 사용 설명서 교정|사용 설명서 초안|주의 문구;회의 안건 정리|회의 안건 초안|발표 순서;채용 평가 기준 정리|평가 기준 초안|평가 항목;프로그램 운영 지침 정리|운영 지침 초안|승인 절차"),
    ("기계 설계", "FreeCAD.exe", "FreeCAD", False, "치구 기준핀 배치|기준핀 배치 도면|기준면;프레임 연결 구조 설계|프레임 연결 도면|연결 방향;절삭 공구 경로 검토|공구 경로 파일|진입 위치;보호 덮개 구조 설계|덮개 구조 도면|개폐 방향;브래킷 조립 순서 검토|브래킷 조립도|부품 순서"),
    ("학습", "chrome.exe", "Chrome", True, "영어 관계대명사 복습|관계대명사 연습 노트|문장 구조;함수 그래프 학습|함수 그래프 노트|축 표기;정렬 알고리즘 학습|정렬 과정 노트|비교 순서;지형도 읽기 연습|지형도 연습 노트|등고선 표시;프랑스어 듣기 연습|듣기 연습 노트|발음 구분"),
    ("연구", "zotero.exe", "Zotero", False, "논문 참고 문헌 정리|참고 문헌 목록|인용 형식;실험 관측 항목 정리|관측 항목 목록|측정 간격;문헌 검색 결과 분류|문헌 분류표|분류 기준;설문 문항 설계|설문 문항 초안|선택지 순서;연구 발표 개요 작성|발표 개요 초안|설명 순서"),
    ("물류", "EXCEL.EXE", "Excel", False, "창고 피킹 순서 설계|피킹 순서표|선반 구역;포장 자재 소요 정리|포장 소요표|자재 종류;반품 분류 절차 검토|반품 분류표|상태 구분;운송 경로 비교|운송 경로표|경유 지점;출고 라벨 양식 점검|출고 라벨 양식|취급 표시"),
    ("행사 운영", "Notion.exe", "Notion", False, "자원봉사 역할 배치|봉사 역할표|담당 구역;리허설 진행 순서 정리|리허설 진행표|입장 순서;전시 설명 일정 조정|전시 설명표|설명 구간;부스 운영 준비|부스 준비 목록|필수 장비;행사 참가 안내 작성|참가 안내 초안|집합 위치"),
    ("개인 재무", "EXCEL.EXE", "Excel", True, "월간 생활비 계획|생활비 계획표|지출 분류;정기 구독 목록 정리|구독 목록표|갱신 주기;저축 목표 점검|저축 목표표|목표 기간;교통비 지출 정리|교통비 기록표|이용 구간;공동 생활비 정산 준비|공동 지출 목록|부담 구분"),
    ("여행 준비", "chrome.exe", "Chrome", True, "기차 여행 일정 구성|기차 여행 일정표|환승 구간;캠핑 장비 준비|캠핑 준비 목록|장비 무게;박물관 방문 동선 정리|관람 동선표|예약 시간;도보 여행 코스 비교|도보 코스표|이동 거리;가족 여행 짐 정리|여행 짐 목록|휴대 구분"),
    ("집안 관리", "Notion.exe", "Notion", True, "세탁 품목 분류|세탁 분류 목록|소재 표시;식재료 사용 순서 정리|식재료 목록|보관 위치;가구 배치 검토|가구 배치안|통행 공간;계절 의류 정리|계절 의류 목록|보관 구역;청소 도구 정리|청소 도구 목록|사용 구역"),
    ("운동 기록", "EXCEL.EXE", "Excel", True, "자전거 주행 기록 정리|주행 기록표|휴식 구간;수영 연습 기록 정리|수영 연습표|연습 종류;걷기 일정 구성|걷기 일정표|경유 지점;스트레칭 기록 정리|스트레칭 기록표|운동 구분;등산 코스 준비|등산 준비표|코스 고도"),
    ("취미", "Notion.exe", "Notion", True, "뜨개질 무늬 기록|뜨개질 무늬 노트|코 순서;자수 도안 정리|자수 도안 노트|실 색상;씨앗 파종 기록|파종 기록표|파종 구역;우표 수집 목록 정리|우표 수집표|발행 지역;모형 조립 순서 기록|모형 조립 노트|연결 위치"),
    ("구매 비교", "chrome.exe", "Chrome", True, "텐트 구매 조건 비교|텐트 비교표|설치 크기;의자 구매 조건 비교|의자 비교표|높이 조절;여행 가방 구매 비교|가방 비교표|수납 구조;주방 용기 구매 비교|용기 비교표|뚜껑 형태;책장 구매 조건 비교|책장 비교표|선반 간격"),
    ("지역 모임", "KakaoTalk.exe", "KakaoTalk", True, "지역 청소 모임 준비|청소 모임 안내|집합 구역;독서 모임 준비|독서 모임 안내|읽을 범위;동네 장터 참여 준비|장터 준비 목록|부스 위치;산책 모임 일정 정리|산책 모임 일정표|출발 지점;공유 정원 역할 정리|정원 역할표|관리 구역"),
    ("개인 연락", "Notion.exe", "Notion", True, "동창 모임 연락 정리|동창 연락 목록|참석 응답;가족 행사 안내 작성|가족 행사 안내|집합 시간;학부모 안내 정리|학부모 안내 초안|준비 항목;생일 모임 준비|생일 모임 목록|역할 구분;동아리 가입 안내 정리|가입 안내 초안|참가 조건"),
    ("시스템 관리", "SystemSettings.exe", "Windows 설정", False, "공유 폴더 권한 정리|공유 권한 목록|접근 범위;백업 복원 절차 검토|복원 절차 노트|복원 순서;방화벽 예외 점검|예외 규칙 목록|허용 경로;암호화 보관함 정리|보관함 목록|보관 기간;기기 동기화 설정 점검|동기화 설정 목록|동기화 범위"),
]

# urgency, event key, category override, action, message, and urgency evidence.
BULK_EVENTS = [
    (1,"joke_focus","개인 일반","주제 대화 확인","{goal} 하다가 내 집중력도 같이 접히는 중ㅋㅋ", "응답 요청 없는 주제 농담"),
    (1,"joke_break","개인 일반","주제 대화 확인","오늘의 소감: {goal}보다 간식 고르는 게 더 어렵네ㅎㅎ", "대응할 필요 없는 잡담"),
    (1,"archive",None,"보관 위치 확인","지난 작업의 {item}를 보관함으로 옮겼습니다. 지금 확인할 일은 없습니다.","완료된 보관 내역 안내"),
    (1,"color_option",None,"표시 색상 비교","{item}에 쓸 수 있는 새 표시 색상이 추가됐어요. 바꾸지 않고 계속 사용해도 됩니다.","선택적인 꾸미기 옵션"),
    (1,"course_ad","광고/홍보","강좌 구매 조건 비교","{goal} 참고 강좌는 월 18000원입니다. 무료 참고 자료도 계속 제공하며 유료 강좌 구매는 선택입니다.","필수 대응 없는 강좌 홍보"),
    (1,"coupon_ad","광고/홍보","참고 자료 구매 가격 비교","긴급 할인 알림! {goal} 참고 자료 10% 쿠폰이 5분 뒤 끝나요. 구매하실 분만 적용하세요.","광고 마감은 선택적인 구매 조건"),
    (1,"history",None,"이전 기록 찾아보기","{item}의 지난 변경 기록을 월별로 찾아볼 수 있게 정리했습니다. 필요할 때 이용하세요.","기한 없는 이전 기록 안내"),
    (1,"theme",None,"표시 방식 선택","{item}의 화면 표시 방식에 밝은 테마가 추가됐어요. 기존 표시 방식은 그대로 쓸 수 있어요.","사용 여부를 선택할 수 있는 테마 안내"),
    (1,"community_news","개인 일반","커뮤니티 소식 읽기","{goal} 관련 모임의 지난달 활동 사진을 올려뒀어요. 구경하고 싶으면 봐요.","지난 활동 소식이며 행동 요청 없음"),
    (1,"bookmark",None,"참고 위치 정리","{goal}에 관한 공개 참고 페이지를 북마크 목록에 추가했습니다. 확인 기한은 없습니다.","기한 없는 참고 위치 저장"),
    (2,"field_comment",None,"항목 설명 검토","{item}의 {field} 설명에 의견을 남겼어요. 급한 내용은 아니니 다음 작업 때 참고해 주세요.","다음 작업 때 확인할 수 있는 의견"),
    (2,"ordering",None,"항목 순서 검토","{item}에서 {field} 항목을 먼저 보여 주는 편이 읽기 좋을 것 같아요. 편할 때 의견 주세요.","즉시 반영할 필요 없는 제안"),
    (2,"reference",None,"참고 절차 확인","{goal}에서 {field}를 정리할 때 참고할 절차를 올렸습니다. 나중에 검토해 주세요.","나중에 확인 가능한 참고 절차"),
    (2,"sample",None,"예시 자료 검토","{item}의 {field}를 설명하는 예시를 준비했어요. 시간 될 때 적절한지 봐주세요.","기한 없이 요청한 예시 검토"),
    (2,"layout",None,"배치 설명 검토","{item}의 {field} 배치 이유를 메모해 뒀습니다. 당장 바꿀 필요는 없어요.","긴급하지 않은 배치 설명 공유"),
    (2,"annotation",None,"주석 보완","{item}의 {field} 주석을 더 알아보기 쉽게 적어 보면 어떨까요? 다음 정리 때 해도 됩니다.","다음 정리 때 가능한 보완 제안"),
    (2,"formats",None,"공유 형식 비교","{item}를 공유할 두 가지 파일 형식의 차이를 정리했어요. 공유 준비할 때 참고하세요.","시간이 정해지지 않은 공유 참고 정보"),
    (2,"faq",None,"질문 목록 검토","{goal}에서 자주 나오는 질문을 모았습니다. 필요할 때 답변 작성에 활용해 보세요.","급하지 않은 질문 자료"),
    (2,"week_invite","일정/회의","다음 주 점검 일정 검토","다음 주에 {goal} 진행 상황을 점검하려고 합니다. 가능한 시간을 이번 주 안에 알려주세요.","일주일 안에 답할 수 있는 일정 조율"),
    (2,"draft",None,"초안 의견 작성","{item}의 {field} 초안을 공유했습니다. 다음 정리 전에 여유 있을 때 의견 남겨 주세요.","여유 있게 검토할 초안 공유"),
    (3,"afternoon",None,"당일 검토 회신","오늘 오후에 {item}를 정리합니다. 점심 이후 {field} 항목을 확인해 회신해 주세요.","당일 확인 필요하나 즉시 중단 요청 없음"),
    (3,"tomorrow",None,"제출 전 항목 확인","{item}의 {field} 표기가 빠져 있습니다. 내일 오전 제출 전까지 확인해서 알려주세요.","다음 날 제출 전에 확인할 사항"),
    (3,"queue",None,"처리 순서 확인","{item} 관련 확인 요청이 순서대로 쌓이고 있어요. 오늘 중 {field} 처리 순서를 정해주세요.","당일 정리가 필요한 누적 요청"),
    (3,"sync_delay",None,"동기화 상태 점검","{item}의 변경 사항이 늦게 동기화되고 있습니다. 원본은 남아 있으니 오늘 중 상태를 확인해 주세요.","원본 손실 없이 지연되는 동기화"),
    (3,"capacity",None,"정리할 파일 선택","{item} 작업 폴더의 여유 공간이 줄고 있습니다. 저장은 아직 가능하니 다음 작업 전 불필요한 파일을 정리해 주세요.","현재 저장 가능하지만 가까운 시일 내 정리 필요"),
    (3,"review_slot",None,"검토 항목 회신","오늘 오후 검토 시간에 {item}를 다룹니다. 그 전에 {field} 의견을 정리해 보내주세요.","몇 시간 뒤 검토 전에 필요한 회신"),
    (3,"retry",None,"공유 실패 재확인","{item} 공유가 한 번 실패했습니다. 원본은 안전하고 오늘 저녁까지 다시 공유하면 됩니다.","재시도 필요하나 대응까지 여유가 있음"),
    (3,"missing_attachment",None,"참고 첨부 보완","내일 {goal} 점검에 쓸 {item}의 참고 첨부가 빠졌어요. 오늘 중 보완해 주세요.","다음 날 점검 준비를 위한 당일 보완"),
    (3,"callback",None,"확인 요청 답변","{item}의 {field}에 대해 문의가 왔습니다. 상대방이 오늘 오후 답변을 기다리니 확인해 주세요.","당일 답변을 기다리는 문의"),
    (3,"disagreement",None,"의견 차이 정리","{item}의 {field}에 서로 다른 의견이 남아 있어요. 내일 작업 시작 전 기준을 정리해 주세요.","다음 작업 시작 전에 정리할 의견 차이"),
    (4,"live_access","일정/회의","진행 중 점검 입장","{goal} 점검이 이미 시작됐어요. 참가 연결이 끊겨 새 입장 링크를 보냅니다. 지금 들어와 주세요.","이미 진행 중인 약속에 지금 입장 필요"),
    (4,"near_close",None,"마감 전 공유 확인","{item} 공유 마감이 10분 남았는데 아직 제출되지 않았습니다. 지금 확인해 주세요.","10분 뒤 마감이어서 작업 중단 가치 있음"),
    (4,"waiting",None,"실시간 확인 답변","지금 진행 중인 {goal} 검토가 {field} 확인 답변을 기다리며 멈췄습니다. 바로 답해주세요.","다른 참여자의 진행이 현재 답변에 막혀 있음"),
    (4,"presentation_turn","일정/회의","발표 자료 열기","{item}를 설명할 차례가 앞당겨졌습니다. 잠시 뒤 호출하니 지금 자료를 열어 준비해 주세요.","바로 앞당겨진 발표 준비 요청"),
    (4,"closed_window",None,"접수 창 확인","예약해 둔 {item} 접수 창이 열렸고 5분 후 닫힙니다. 지금 접수 상태를 확인해 주세요.","예약된 필수 접수 창이 곧 닫힘"),
    (4,"handoff",None,"인계 대상 확인","{item}를 전달할 담당자가 지금 인계 장소에서 기다립니다. {field} 확인 후 바로 답해주세요.","현재 진행 중인 인계에 즉시 확인 필요"),
    (4,"wrong_version",None,"사용 버전 교체","진행 중인 {goal} 설명에 이전 {item}가 열려 있습니다. 지금 최신 버전으로 바꿔주세요.","현재 사용 중인 잘못된 버전 교체 요청"),
    (4,"reservation_release",None,"예약 유지 응답","{goal} 점검 시간을 예약해 뒀는데 5분 안에 참가 확인을 하지 않으면 예약이 해제됩니다. 지금 답해주세요.","곧 예약을 잃을 수 있는 참가 확인"),
    (4,"today_slot",None,"현재 확인 시간 참여","지금만 {item}를 함께 확인할 수 있습니다. 다음 일정으로 이동하기 전 {field} 질문을 바로 확인해 주세요.","현재만 가능한 확인 시간에 참여 필요"),
    (4,"live_field",None,"설명 항목 확인","지금 {item} 설명을 진행하는데 {field} 설명이 비어 있어 진행을 못 하고 있습니다. 확인 가능한 내용을 바로 보내주세요.","진행 중인 설명이 항목 누락으로 멈춤"),
    (5,"only_copy_delete","시스템/보안","원본 삭제 중지","{item}의 유일한 원본이 자동 작업으로 계속 삭제되고 있습니다. 복사본이 없으니 즉시 삭제 작업을 중지해야 합니다.","유일한 원본이 현재 삭제 중인 손실 위험"),
    (5,"public_contacts","시스템/보안","개인정보 공개 차단","{item}에 포함된 개인 연락처가 외부 공개 주소에서 누구나 열람 가능합니다. 지금 공개를 차단해야 합니다.","현재 개인정보가 외부에 공개된 상태"),
    (5,"encrypting","시스템/보안","파일 변경 차단","알 수 없는 프로그램이 {item} 원본을 암호화하고 있습니다. 아직 변하지 않은 파일이 있어 즉시 차단이 필요합니다.","정체 불명 프로그램의 현재 파일 암호화"),
    (5,"mass_external","시스템/보안","외부 전송 중단","{item}의 비공개 내용이 외부 수신자들에게 계속 전송되고 있습니다. 바로 전송을 중단해야 합니다.","현재 비공개 내용의 반복 외부 전송"),
    (5,"destructive_restore","시스템/보안","덮어쓰기 중단","복구 작업이 {item} 원본을 빈 파일로 덮어쓰는 중입니다. 남은 원본을 보존하려면 즉시 중지해야 합니다.","복구 중 원본을 현재 훼손하는 작업"),
    (5,"published_private","시스템/보안","비공개 자료 게시 중지","공개 게시 작업에 비공개 {item}가 섞였고 지금 게시되고 있습니다. 공개된 내용을 내리고 작업을 즉시 중단해야 합니다.","현재 진행 중인 비공개 자료 게시"),
    (5,"personal_export","시스템/보안","개인정보 전송 차단","승인된 범위를 벗어난 자동 내보내기가 {item}의 개인 정보를 외부로 보내고 있습니다. 즉시 전송 경로를 막아야 합니다.","승인 범위 밖 개인정보가 현재 전송 중"),
    (5,"overwriting","시스템/보안","잘못된 저장 중지","자동 저장이 {item}의 정상 원본을 깨진 내용으로 계속 덮어쓰고 있습니다. 원본이 더 손상되기 전에 지금 저장을 중지해야 합니다.","정상 원본을 현재 계속 손상시키는 저장"),
    (5,"permanent_cleanup","시스템/보안","영구 삭제 취소","{item}의 보존해야 할 원본이 영구 삭제 대기열에 들어갔고 지금 실행 중입니다. 복구 가능한 사본이 없어 즉시 취소해야 합니다.","복구 사본 없는 원본의 현재 영구 삭제"),
    (5,"broad_share","시스템/보안","공개 공유 해제","비공개 {item}의 공유 범위가 전체 공개로 바뀌었고 외부 접속이 확인됐습니다. 지금 전체 공개를 해제해야 합니다.","비공개 원본의 실제 외부 접근이 확인됨"),
]


def bulk_recipes():
    recipes = []
    for domain,process,tool,personal,text in BULK_DOMAINS:
        tasks = [tuple(part.split("|")) for part in text.split(";")]
        for task_index,(goal,item,field) in enumerate(tasks):
            for event in BULK_EVENTS:
                recipes.append({"domain":domain,"process":process,"tool":tool,"personal":personal,
                    "goal":goal,"item":item,"field":field,"neighbor":tasks[(task_index+1)%len(tasks)][0],
                    "neighbor_item":tasks[(task_index+1)%len(tasks)][1],"event":event})
    return recipes


def bulk_context(recipe, relevance, rng, other):
    process,tool = recipe["process"],recipe["tool"]
    goal,item = recipe["goal"],recipe["item"]
    action = recipe["event"][3]
    if relevance == 1:
        if rng.random()<.5:
            return "","","현재 작업 정보가 없어 관련성을 확인할 수 없습니다"
        title = rng.choice([other["goal"],other["item"]+" - "+other["event"][3],other["goal"]+" 배경 읽기"])
        return other["process"],title+" - "+other["tool"],"현재 다른 분야의 작업 목적과 무관합니다"
    if relevance == 2:
        title = rng.choice([recipe["neighbor"],recipe["neighbor"]+" 배경 읽기",recipe["neighbor_item"]+" - "+action])
        return process,title+" - "+tool,"분야는 같지만 현재 다른 작업에 주는 도움은 거의 없습니다"
    if relevance == 3:
        title = rng.choice([f"{goal} 용어 공부",f"{goal} 배경 읽기",f"{goal} 관련 동아리 대화",f"{goal} 개념 자료 읽기"])
        if recipe["event"][1].startswith("joke") or recipe["event"][1]=="community_news":
            title = rng.choice([goal,f"{item} - 항목 설명 검토",f"{goal} 진행 상황 검토"])
        return process,title+" - "+tool,"주제는 같지만 현재 학습·대화 목적에 주는 도움은 불분명합니다"
    if relevance == 4:
        if recipe["event"][1] in ("course_ad","coupon_ad"):
            return process,f"{goal} 참고 자료 구매 예산 비교 - {tool}","현재 참고 자료 구매 비용 비교에 실제 가격·할인 조건이 직접 도움이 됩니다"
        title = rng.choice([goal,goal,f"{goal} 진행 상황 검토",f"{goal} 준비 사항 점검"])
        if recipe["event"][1] in ("theme","color_option"):
            title = f"{goal} 자료 표시 환경 검토"
        return process,title+" - "+tool,"알림의 작업 정보가 현재 진행 계획·준비 확인에 직접 도움이 됩니다"
    if recipe["event"][1] in ("course_ad","coupon_ad"):
        return process,f"{goal} 참고 {'강좌' if recipe['event'][1]=='course_ad' else '자료'} - {action} - {tool}","현재 구매 조건을 비교하는 정확한 참고 자료의 가격·할인 정보입니다"
    if recipe["event"][1] == "bookmark":
        return process,f"{goal} 공개 참고 페이지 위치 정리 - {tool}","현재 정리하는 정확한 주제의 공개 참고 페이지 위치입니다"
    title = rng.choice([f"{item} - {action}",f"{action}: {item}",f"{item} {action} 작업",f"{item} / {action} 확인"])
    return process,title+" - "+tool,"현재 작업의 정확한 대상과 확인·처리 행동에 연결됩니다"


def generate_bulk(source,output,approved,count=5000):
    if output.exists():
        raise ValueError("preserve earlier generation versions")
    approval = json.loads((approved/"approval.json").read_text(encoding="utf-8"))
    approved_manifest = json.loads((approved/"manifest.json").read_text(encoding="utf-8"))
    if digest(approved/"dataset.jsonl") != approval["dataset_sha256"] or digest(approved/"approval.json") != approved_manifest["approval_sha256"]:
        raise ValueError("approved preview changed")
    original_manifest = json.loads((source/"prepared/manifest.json").read_text(encoding="utf-8"))
    if digest(source/"dataset.jsonl") != original_manifest["source_sha256"]:
        raise ValueError("source dataset changed")
    recipes = bulk_recipes()
    if count != 5000 or len(recipes) != 5000:
        raise ValueError("this recipe catalog supports exactly 5000 originals; expand semantic catalog before requesting more")
    rng = random.Random(42)
    rng.shuffle(recipes)
    source_samples = load_samples(source/"dataset.jsonl")+load_samples(approved/"dataset.jsonl")
    earlier = source.parent/"v3_score_review_02_approved_01/dataset.jsonl"
    if earlier.exists(): source_samples += load_samples(earlier)
    normalize = lambda value: re.sub(r"\s+","",value).casefold()
    known_bodies = {normalize(s.notification.body) for s in source_samples}
    known_windows = {(s.context.active_process,s.context.window_title) for s in source_samples if s.context.active_process}
    rows,lineage,seen,windows = [],[],set(),set()
    relevance_counts = Counter()
    primary_urgency = Counter()
    variants_per_score = Counter()
    for recipe in recipes:
        urgency,event_key,category,action,text,urgency_reason = recipe["event"]
        body = text.format(**recipe)
        normalized = normalize(body)
        if normalized in seen or normalized in known_bodies:
            raise ValueError("exact notification duplicate in generated pool or original sources")
        seen.add(normalized)
        # A joke is never used to propose relevance 4/5 under the approved general rule.
        allowed = (1,2,3) if event_key.startswith("joke") or event_key=="community_news" else (1,2,3,4,5)
        allowed = [score for score in allowed if relevance_counts[score]<1000]
        if not allowed: raise ValueError("relevance quota exhausted")
        relevance = min(allowed,key=lambda score:(relevance_counts[score],score))
        relevance_counts[relevance] += 1
        primary_urgency[urgency] += 1
        category = category or ("개인 일반" if recipe["personal"] else "일반 업무")
        if urgency>=4 and category not in ("일정/회의","시스템/보안"):
            category = "개인 중요" if recipe["personal"] else "긴급 업무"
        key = hashlib.sha256((recipe["domain"]+"|"+recipe["goal"]+"|"+event_key).encode("utf-8")).hexdigest()[:16]
        notification = {"id":f"bulk01_{key}_primary","app_name":rng.choice(["KakaoTalk.exe","이메일","Slack","Microsoft Teams","Notion"]) if category!="시스템/보안" else rng.choice(["Windows 보안","파일 관리 알림","공유 관리 알림"]),
            "sender":rng.choice(["담당자","동료","자료 관리","작업 안내","진행 담당"]),"title":recipe["item"]+" · "+action,
            "body":body,"timestamp":"2026-10-04T01:00:00Z"}
        other = rng.choice(recipes)
        while other["domain"]==recipe["domain"]:
            other = rng.choice(recipes)
        variants = [("primary",relevance)]
        if variants_per_score[urgency]<100:
            variants.append(("context_check",1))
            variants_per_score[urgency] += 1
        for role,score in variants:
            process,window,reason = bulk_context(recipe,score,rng,other)
            if role=="context_check" and (process,window)==(rows[-1].context.active_process,rows[-1].context.window_title):
                if process:
                    process,window,reason = "","","현재 작업 정보가 없어 관련성을 확인할 수 없습니다"
                else:
                    process,window,reason = other["process"],other["goal"]+" - "+other["tool"],"현재 다른 분야의 작업 목적과 무관합니다"
            if (process,window) in known_windows and process:
                raise ValueError("populated window overlaps frozen source")
            if process: windows.add((process,window))
            sample = FilteringSample.model_validate({"notification":{**notification,"id":f"bulk01_{key}_{role}"},
                "context":{"active_process":process,"window_title":window,"last_updated":"2026-10-04T01:00:00Z",
                    "duration_seconds":rng.choice([12,47,103,216,391]) if process else 0,"recent_processes":[]},
                "label":{"urgency_score":urgency,"relevance_score":score,"category":category,"ai_summary_reason":urgency_reason+"; "+reason+"."}})
            parse_model_output(json.dumps(sample.label.model_dump(),ensure_ascii=False))
            rows.append(sample)
            lineage.append({"notification_id":sample.notification.id,"original_id":f"bulk01_{key}","domain":recipe["domain"],"task":recipe["goal"],
                "event_family":event_key,"semantic_group":event_key,"task_group":recipe["domain"],"context_role":role,
                "origin":"authored_scenario_composition","review_status":"synthetic_unreviewed","label_source":"rule_proposal",
                "intended_split":"train_only","recipe_key":key})
    unique_urgency_samples(rows)
    if len(seen)!=5000 or relevance_counts!=Counter({i:1000 for i in range(1,6)}):
        raise ValueError("wrong original count or primary relevance quotas")
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl",[s.model_dump(mode="json") for s in rows])
    write_lines(output/"lineage.jsonl",lineage)
    manifest = {"purpose":"5000 composed synthetic notification candidates; not individually human-reviewed or final evaluation", "created_date":"2026-10-04",
        "status":"synthetic_candidates_pending_quality_review","originals":5000,"primary_rows":5000,"context_check_rows":500,"rows":len(rows),
        "seed":42,"policy_version":POLICY_VERSION,"human_reviewed_originals_in_bulk":0,
        "approved_preview_originals":approval["reviewed_originals"],"approved_preview_rows":approval["reviewed_context_rows"],
        "approved_manifest_sha256":digest(approved/"manifest.json"),"approved_dataset_sha256":digest(approved/"dataset.jsonl"),
        "source_sha256":digest(source/"dataset.jsonl"),"source_manifest_sha256":digest(source/"prepared/manifest.json"),
        "code_sha256":digest(Path(__file__)),"candidate_sha256":digest(output/"candidates.jsonl"),"lineage_sha256":digest(output/"lineage.jsonl"),
        "domains":len(BULK_DOMAINS),"tasks":100,"event_families":len(BULK_EVENTS),"populated_windows":len(windows),
        "primary_urgency_counts":dict(primary_urgency),"primary_relevance_counts":dict(relevance_counts),
        "all_urgency_counts":dict(Counter(s.label.urgency_score for s in rows)),"all_relevance_counts":dict(Counter(s.label.relevance_score for s in rows)),
        "category_counts":dict(Counter(s.label.category for s in rows)),"exact_body_overlap":0,"source_populated_window_overlap":0,
        "urgency_context_invariance":"checked for all 500 paired originals", "training_used":False,"test_inference_used":False,
        "composition":"20 domains x 5 tasks x 50 event recipes; not 5000 independently authored scenarios",
        "limitations":["repeated event phrasing and rule-derived labels can create shortcuts","semantic overlap with heldout families is not exhaustively ruled out",
            "urgent 5 recipes overrepresent digital data and privacy incidents","uniform scores are a training coverage choice, not deployment prevalence",
            "review scope covers only the separate approved preview; bulk labels require quality review"],
        "scale_policy":"extend domains, distinct tasks and event semantics for future versions; preserve stable recipe hashes and lineage, never fill larger counts by renaming people/products",
        "files":{"candidates.jsonl":"5000 original alerts / 5500 context rows with provisional labels","lineage.jsonl":"per-row recipe, origin, review scope and train-only provenance"}}
    write_json(output/"manifest.json",manifest)
    return rows


@lru_cache(maxsize=1)
def _bulk_particle_nouns():
    return tuple(sorted({r[key] for r in bulk_recipes() for key in ("item", "field")}, key=len, reverse=True))


def correct_bulk_particles(text):
    """Repair only catalog slots followed by a generated Korean particle."""
    for noun in _bulk_particle_nouns():
        final = ord(noun[-1]) - 0xAC00
        if not 0 <= final < 11172:
            continue
        consonant = final % 28 != 0
        for wrong, right in (("를", "을"), ("가", "이")) if consonant else (("을", "를"), ("이", "가")):
            text = re.sub(re.escape(noun + wrong) + r"(?=\s|[.,!?]|$)", noun + right, text)
    return text


def revise_bulk(source, output, feedback_path):
    """Apply reviewed fields and declared family-level proposals to a fresh version."""
    if output.exists():
        raise ValueError("preserve earlier generation versions")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    source_hash = digest(source / "candidates.jsonl")
    if source_hash != manifest["candidate_sha256"] or source_hash != feedback["source_sha256"]:
        raise ValueError("candidate or review source changed")
    if digest(source / "lineage.jsonl") != manifest["lineage_sha256"]:
        raise ValueError("lineage changed")
    samples = load_samples(source / "candidates.jsonl")
    rows = [s.model_dump(mode="json") for s in samples]
    lineage = [json.loads(v) for v in (source / "lineage.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {row["notification"]["id"]: row for row in rows}
    by_lineage = {info["notification_id"]: info for info in lineage}
    if len(by_id) != len(rows) or len(by_lineage) != len(lineage) or set(by_id) != set(by_lineage):
        raise ValueError("row/lineage identity mismatch")
    rules = {
        "coupon_ad": (2, "선택적 구매이지만 5분 뒤 할인 종료 — 검수 사례 2 기반 합성 규칙"),
        "near_close": (5, "제출 마감 10분 전 미제출 — 검수 사례 8 기반 합성 규칙"),
        "live_access": (5, "이미 시작한 약속에 즉시 재접속 요청 — 검수 사례 9 기반 합성 규칙"),
    }
    expected_rules = {"coupon_ad": 2, "near_close": 5, "live_access": 5}
    confirmed_rules = {}
    for case in feedback["approved_cases"]:
        if case["notification_id"] not in by_id:
            raise ValueError("reviewed id absent")
        family = by_lineage[case["notification_id"]]["event_family"]
        if case["field"] == "urgency_score":
            confirmed_rules[family] = case["approved_value"]
    if confirmed_rules != expected_rules:
        raise ValueError("family revisions require the expected reviewed urgency examples")
    # Verify the explicit overrides against the untouched candidate labels first.
    overrides = {}
    for item in feedback["row_overrides"]:
        id_, field = item["notification_id"], item["field"]
        if field not in ("urgency_score", "relevance_score") or not 1 <= item["approved_value"] <= 5:
            raise ValueError("invalid reviewed score")
        if id_ not in by_id or by_id[id_]["label"][field] != item["candidate_value"]:
            raise ValueError("review override does not match source")
        if (id_, field) in overrides:
            raise ValueError("duplicate review override")
        overrides[id_, field] = item
    for case in feedback["approved_cases"]:
        item = overrides.get((case["notification_id"], case["field"]))
        if not item or item["approved_value"] != case["approved_value"]:
            raise ValueError("reviewed case missing matching override")
    grammar_count = rule_count = 0
    for row in rows:
        id_ = row["notification"]["id"]
        info = by_lineage[id_]
        body = row["notification"]["body"]
        row["notification"]["body"] = correct_bulk_particles(body)
        grammar_count += body != row["notification"]["body"]
        family = info["event_family"]
        if family in rules:
            urgency, reason = rules[family]
            row["label"]["urgency_score"] = urgency
            prior_reason = row["label"]["ai_summary_reason"].partition("; ")[2]
            row["label"]["ai_summary_reason"] = reason + "; " + prior_reason
            info["urgency_revision_source"] = "family_rule_inferred_from_reviewed_example"
            rule_count += 1
        applied = []
        for field in ("urgency_score", "relevance_score"):
            item = overrides.get((id_, field))
            if item:
                row["label"][field] = item["approved_value"]
                applied.append({"field": field, "value": item["approved_value"], "scope": item["approval_scope"]})
        if applied:
            info["reviewed_field_overrides"] = applied
        # Field review does not turn the whole sample into a human-approved record.
        info["review_status"] = "synthetic_unreviewed"
    revised = [FilteringSample.model_validate(row) for row in rows]
    unique_urgency_samples(revised)
    primary = [s for s in revised if by_lineage[s.notification.id]["context_role"] == "primary"]
    if len({s.notification.body for s in primary}) != len(primary):
        raise ValueError("grammar correction merged originals")
    for s in revised:
        parse_model_output(json.dumps(s.label.model_dump(), ensure_ascii=False))
    current = deepcopy(manifest)
    for stale in ("quality_gate", "audit_sha256"):
        current.pop(stale, None)
    current.update(purpose="Reviewed-field and grammar revision of existing synthetic candidates; not an additional 5000 originals",
                   created_date="2026-10-04", status="synthetic_candidates_pending_quality_review",
                   parent_candidate_sha256=source_hash, parent_manifest_sha256=digest(source / "manifest.json"),
                   feedback_sha256=digest(feedback_path), code_sha256=digest(Path(__file__)),
                   grammar_changed_rows=grammar_count, inferred_urgency_rule_rows=rule_count,
                   explicit_field_review_cases=len(feedback["approved_cases"]), reviewed_override_rows=len(overrides),
                   human_reviewed_originals_in_bulk=0, training_used=False, test_inference_used=False)
    current["family_urgency_revisions"] = {key: value[0] for key, value in rules.items()}
    current["limitations"] = [item for item in current["limitations"] if not item.startswith("review scope covers")]
    current["limitations"].append("four explicit field reviews; inferred family changes are not individual human review")
    current["primary_urgency_counts"] = dict(Counter(s.label.urgency_score for s in primary))
    current["all_urgency_counts"] = dict(Counter(s.label.urgency_score for s in revised))
    current["all_relevance_counts"] = dict(Counter(s.label.relevance_score for s in revised))
    current["quality_gate"] = {"ready_for_training": False, "review_scope": "four explicit fields; family propagation remains synthetic",
                               "remaining": ["urgent score 5 still dominated by digital incidents", "window wording shortcuts", "missing category 기타", "semantic heldout overlap not exhaustively audited"]}
    current["files"] = {"candidates.jsonl": "revised 5500 synthetic candidate rows; use instead of parent version",
                        "lineage.jsonl": "stable original ids and explicit-field versus inferred-rule provenance"}
    output.mkdir(parents=True)
    write_lines(output / "candidates.jsonl", rows)
    write_lines(output / "lineage.jsonl", lineage)
    current["candidate_sha256"] = digest(output / "candidates.jsonl")
    current["lineage_sha256"] = digest(output / "lineage.jsonl")
    write_json(output / "manifest.json", current)
    return revised


# Domain, target, four genuinely different states (urgency 5/4/3/2), train links.
# State changes describe different events, not name substitutions or paraphrases.
URGENT_STATE_CASES = [
    ("건강","가족 의식 소실 연락",
     "가족이 갑자기 의식을 잃어 구급대가 현장에 도착했습니다. 보호자 확인을 위해 지금 전화를 받아주세요.",
     "가족은 의식을 회복해 병원에서 관찰 중입니다. 담당자가 지금 보호자 연락처를 확인하며 답변을 기다리고 있습니다.",
     "가족이 병원 관찰을 마쳤고 오늘 오후 귀가 준비를 합니다. 점심 이후 보호자 연락 가능 시간을 알려주세요.",
     "가족의 지난 진료 때 등록한 보호자 연락처 목록을 정리했습니다. 여유 있을 때 변경할 내용이 있는지 봐주세요.",["v3_09"]),
    ("건강","응급 이송 보호자 확인",
     "가족이 응급 이송 중입니다. 동행한 구급대가 보호자 정보를 확인해야 하니 지금 연락을 받아주세요.",
     "가족은 병원에 도착해 상태가 안정됐습니다. 접수 담당자가 지금 보호자 인적사항 확인 답변을 기다리고 있습니다.",
     "가족의 검사 결과 설명이 오늘 오후 예정돼 있습니다. 그 전에 보호자 연락 가능한 시간을 회신해주세요.",
     "지난 이송 때 작성한 보호자 정보 사본을 보냈습니다. 급한 수정은 없으며 다음 정리 때 보관 위치를 확인해주세요.",["v3_09"]),
    ("건강","낙상 구조 보호자 연락",
     "가족이 계단에서 넘어져 일어나지 못하고 구조대가 출동했습니다. 현장 담당자가 보호자와 지금 통화해야 합니다.",
     "가족은 진료를 마쳐 안전한 장소에서 기다리고 있습니다. 지금 이동 차량을 배정하려니 보호자 동행 여부를 답해주세요.",
     "가족의 다음 진료에 동행할 사람을 오늘 저녁까지 정해주세요. 현재 추가 응급 상황은 없습니다.",
     "가족 진료 때 이용한 이동 지원 연락처를 공유했습니다. 필요할 때 참고하시고 지금 답할 필요는 없습니다.",["v3_09"]),
    ("건강","수술 중 보호자 연락",
     "가족의 응급 수술이 진행 중이며 의료진이 보호자와 즉시 연락해야 한다고 요청했습니다. 지금 전화를 받아주세요.",
     "가족의 수술이 끝나 회복실에서 안정적으로 관찰 중입니다. 담당자가 지금 보호자 방문 위치를 안내하려고 기다립니다.",
     "가족의 내일 회복실 방문을 준비합니다. 오늘 저녁까지 방문할 보호자 이름을 보내주세요.",
     "가족의 회복실 면회 안내 자료를 보냈습니다. 다음 방문 준비 때 읽어보시면 됩니다.",["v3_09"]),
    ("건강","호흡 곤란 보호자 연락",
     "가족이 숨쉬기 힘들어 구급대가 도착했습니다. 현장 확인을 위해 보호자와 즉시 통화가 필요합니다.",
     "가족은 진료 후 호흡이 안정됐습니다. 담당자가 지금 보호자에게 귀가 안내를 전달하려고 기다립니다.",
     "가족이 진료 후 집에서 쉬고 있습니다. 오늘 저녁까지 다음 방문에 동행할 수 있는지 알려주세요.",
     "가족의 지난 병원 방문 연락 기록을 보냈습니다. 필요할 때 참고하면 되고 회신 기한은 없습니다.",["v3_09"]),
    ("안전","건물 화재 대피",
     "현재 머무는 건물에서 실제 화재가 확인됐습니다. 훈련이 아니며 안전 담당자의 즉시 대피 안내가 전달됐습니다.",
     "화재 위험은 해소됐고 모두 안전한 장소에 있습니다. 안전 담당자가 지금 인원 확인을 하니 대피 여부를 답해주세요.",
     "내일 건물 대피 동선 점검이 예정돼 있습니다. 오늘 저녁까지 참석 가능한 시간을 회신해주세요.",
     "완료된 건물 대피 훈련 결과를 공유합니다. 현재 화재나 점검 요청은 없고 다음 안전 교육 때 참고하면 됩니다.",[]),
    ("안전","사업장 가스 누출 통제",
     "사업장 가스 누출이 확인돼 해당 구역 출입이 즉시 통제됐습니다. 현재 구역에 있는 사람은 현장 대피 안내를 확인해주세요.",
     "가스 누출 구역은 차단됐고 작업자는 모두 밖에 있습니다. 통제 담당자가 지금 인원 확인 답변을 기다립니다.",
     "다음 주 사업장 가스 설비 점검을 준비합니다. 오늘 저녁까지 출입 담당자 이름을 보내주세요.",
     "사업장 가스 설비의 지난 점검 기록을 보냈습니다. 현재 이상은 없으며 다음 기록 정리 때 확인하면 됩니다.",["v3_10"]),
    ("안전","하천 범람 대피",
     "현재 있는 야영 구역에 하천 범람으로 대피 명령이 내려졌습니다. 현장 통제 담당자의 즉시 이동 안내를 확인해주세요.",
     "야영 구역 인원은 모두 안전한 집결지에 있습니다. 담당자가 지금 집결 여부를 확인하니 바로 답해주세요.",
     "주말 야영지 이용 전 안전 안내를 확인해야 합니다. 오늘 저녁까지 참가 인원 명단을 보내주세요.",
     "지난 야영지 안전 점검 결과를 공유했습니다. 현재 대피나 회신 요청은 없고 다음 계획 때 참고하면 됩니다.",[]),
    ("안전","작업장 유해물질 통제",
     "작업장에 유해물질 유출이 확인돼 현장 즉시 대피 안내가 나왔습니다. 해당 구역 작업자는 지금 통제 안내를 확인해주세요.",
     "유출 구역은 차단됐고 작업자는 안전한 곳으로 이동했습니다. 담당자가 지금 작업자 위치 확인을 기다립니다.",
     "내일 작업장 안전 설비 점검을 준비합니다. 오늘 오후까지 구역별 출입 담당자를 회신해주세요.",
     "작업장 안전 교육 자료가 보관함에 올라왔습니다. 현재 사고는 없으며 다음 교육 때 참고하면 됩니다.",[]),
    ("안전","어린이 이동 안전 확인",
     "아이의 통학 차량에서 하차가 확인되지 않아 안전 담당자가 현장 확인 중입니다. 보호자가 지금 연락해 아이의 위치를 확인해야 합니다.",
     "아이는 안전한 장소에 있고 보호자 인계를 기다립니다. 담당자가 지금 인수할 사람을 확인하니 바로 답해주세요.",
     "내일 아이를 데려갈 보호자 명단을 오늘 저녁까지 보내주세요. 현재 아이의 위치나 안전에 이상은 없습니다.",
     "지난달 통학 차량 이용 기록을 공유했습니다. 현재 확인 요청은 없으며 필요할 때 참고하면 됩니다.",[]),
    ("금전","미등록 수취인 이체",
     "미등록 수취인에게 본인이 요청하지 않은 이체가 반복 실행되고 있습니다. 지금 거래 중지 접수를 해야 합니다.",
     "이상 이체는 차단됐고 추가 출금은 없습니다. 상담원이 지금 피해 접수를 진행하니 확인 질문에 답해주세요.",
     "이상 거래 조사에 필요한 증빙 목록을 보냈습니다. 오늘 저녁까지 제출 가능한 자료를 알려주세요.",
     "이상 거래 상담 기록을 보냈습니다. 접수는 완료됐고 현재 추가 대응 요청이나 기한은 없습니다.",["v3_42"]),
    ("금전","중복 자동 지급 중지",
     "같은 인건비가 자동 작업으로 반복 지급되고 있습니다. 추가 지급이 계속 발생하니 지금 지급 작업을 중지해야 합니다.",
     "중복 지급 작업은 중지됐습니다. 회계 담당자가 지금 수령인 확인을 진행하니 확인 가능한 명단을 답해주세요.",
     "중복 지급 내역은 정리됐습니다. 오늘 오후까지 환수 담당자 연락 가능 시간을 회신해주세요.",
     "완료된 지급 작업의 확인서를 보냈습니다. 추가 지급은 없고 다음 회계 정리 때 참고하면 됩니다.",[]),
    ("금전","착오 이체 취소 마감",
     "다른 계좌로 보낸 이체가 아직 실행 대기 중입니다. 5분 뒤 취소가 불가능해지니 지금 취소 여부를 확인해주세요.",
     "착오 이체는 취소돼 출금이 없습니다. 담당자가 지금 올바른 수취인 확인을 기다리니 답해주세요.",
     "재송금할 수취인 정보를 오늘 저녁까지 확인해주세요. 현재 이체는 취소된 상태이고 추가 출금은 없습니다.",
     "취소가 완료된 이체 영수증을 보냈습니다. 다시 확인할 기한은 없으며 필요할 때 참고하면 됩니다.",[]),
    ("금전","결제 도용 추가 승인",
     "본인이 요청하지 않은 카드 결제가 연속 승인되고 있습니다. 추가 결제를 막으려면 지금 이용 중지 접수를 해야 합니다.",
     "카드 이용은 중지돼 추가 승인이 없습니다. 담당자가 지금 도용 거래 목록을 확인하니 바로 답해주세요.",
     "도용 거래 조사에 필요한 구매 내역을 오늘 저녁까지 보내주세요. 현재 추가 결제는 차단돼 있습니다.",
     "지난 결제 도용 상담의 처리 결과를 보냈습니다. 추가 대응 요청은 없으며 필요할 때 읽어보시면 됩니다.",["v3_42"]),
    ("금전","청약 잔금 처리 마감",
     "신청한 청약의 필수 잔금 처리 마감이 7분 남았는데 아직 미처리 상태입니다. 마감 후 신청이 취소되니 지금 확인해주세요.",
     "청약 잔금은 접수됐습니다. 담당자가 지금 입금자 명의를 확인하니 신청자 이름을 바로 답해주세요.",
     "청약 잔금 처리에 필요한 안내서를 보냈습니다. 내일 오후 처리 전까지 입금자 정보를 확인해주세요.",
     "완료된 청약 잔금 처리 내역을 보냈습니다. 추가 확인 기한은 없으며 다음 서류 정리 때 보관하면 됩니다.",[]),
    ("필수 일정","장학 서류 접수",
     "신청한 장학금 필수 서류가 누락됐고 보완 마감이 6분 남았습니다. 마감 뒤 접수가 거절되니 지금 보완해주세요.",
     "장학 서류는 접수됐습니다. 담당자가 지금 신청자 정보를 대조하며 답변을 기다리니 확인해주세요.",
     "장학 신청의 첨부 목록을 보냈습니다. 내일 오후 제출 전에 서류가 준비됐는지 회신해주세요.",
     "지난 장학 신청의 접수 내역을 보냈습니다. 지금 진행할 신청이나 확인 기한은 없습니다.",["v3_43"]),
    ("필수 일정","실기시험 입실",
     "예약한 실기시험 본인 확인이 시작됐고 2분 뒤 입실을 마감합니다. 지금 확인 링크에 접속해주세요.",
     "실기시험 입실은 완료됐습니다. 감독관이 지금 준비물 확인을 진행하니 확인 요청에 답해주세요.",
     "내일 실기시험에 가져갈 준비물을 보냈습니다. 오늘 저녁까지 누락된 항목이 있는지 확인해주세요.",
     "완료된 실기시험 참석 기록을 보냈습니다. 지금 응시할 시험이나 회신 기한은 없습니다.",["v3_29","v3_31"]),
    ("필수 일정","항공편 탑승 마감",
     "예약한 항공편이 최종 탑승 안내 중이며 4분 뒤 탑승구가 닫힙니다. 지금 탑승 안내를 확인해주세요.",
     "항공편 탑승 확인은 끝났습니다. 담당자가 지금 좌석 정보를 확인하고 있으니 안내 질문에 답해주세요.",
     "내일 예약한 항공편의 탑승 안내를 보냈습니다. 오늘 저녁까지 준비할 서류를 확인해주세요.",
     "완료된 항공편 이용 내역을 보냈습니다. 현재 탑승이나 응답 요청은 없으며 필요할 때 참고하면 됩니다.",[]),
    ("필수 일정","자격 등록 보완 마감",
     "신청한 자격 등록의 필수 증빙이 누락됐습니다. 보완 마감이 9분 남았고 이후 등록이 취소되니 지금 제출해주세요.",
     "자격 등록 증빙은 접수됐습니다. 담당자가 지금 신원 정보 대조를 진행하니 확인 질문에 답해주세요.",
     "자격 등록 증빙 목록을 보냈습니다. 내일 제출 전에 오늘 오후 누락 여부를 확인해주세요.",
     "완료된 자격 등록 확인서를 보냈습니다. 지금 추가 제출할 자료나 기한은 없습니다.",["v3_43"]),
    ("필수 일정","교육 필수 출석 확인",
     "신청한 필수 교육이 시작됐고 5분 뒤 출석 확인을 마감합니다. 미확인 시 이수가 인정되지 않으니 지금 접속해주세요.",
     "필수 교육 출석은 확인됐습니다. 진행자가 지금 자료 수신 여부를 확인하니 바로 답해주세요.",
     "내일 필수 교육 접속 준비를 합니다. 오늘 저녁까지 사용할 연락처를 회신해주세요.",
     "완료된 필수 교육의 참석 내역을 보냈습니다. 현재 교육이나 추가 확인 기한은 없습니다.",["v3_29","v3_31"]),
]


def expand_urgent_states(source, approved, output, original_source=None):
    if output.exists():
        raise ValueError("preserve earlier expansion versions")
    base_manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    if digest(source/"candidates.jsonl") != base_manifest["candidate_sha256"] or digest(source/"lineage.jsonl") != base_manifest["lineage_sha256"]:
        raise ValueError("base candidate pool changed")
    approval = json.loads((approved/"approval.json").read_text(encoding="utf-8"))
    approved_manifest = json.loads((approved/"manifest.json").read_text(encoding="utf-8"))
    if digest(approved/"dataset.jsonl") != approval["dataset_sha256"] or digest(approved/"approval.json") != approved_manifest["approval_sha256"]:
        raise ValueError("approved urgent review changed")
    if approval["reviewed_originals"] != 10 or approval["reviewed_context_rows"] != 13:
        raise ValueError("expected approved urgent boundary review")
    original = original_source or source.parent/"v3_reviewed_01"
    old_manifest = json.loads((original/"prepared/manifest.json").read_text(encoding="utf-8"))
    if digest(original/"dataset.jsonl") != old_manifest["source_sha256"]:
        raise ValueError("original data changed")
    if approved_manifest["source_v3_sha256"] != old_manifest["source_sha256"]:
        raise ValueError("urgent approval came from a different source")
    train_ids = set(old_manifest["splits"]["train"]["message_ids"])
    if any(not set(case[-1]).issubset(train_ids) for case in URGENT_STATE_CASES):
        raise ValueError("expanded semantic links must be train-only")
    rows = load_samples(source/"candidates.jsonl")
    lineage = [json.loads(x) for x in (source/"lineage.jsonl").read_text(encoding="utf-8").splitlines()]
    reviewed = load_samples(approved/"dataset.jsonl")
    known = rows + reviewed + load_samples(original/"dataset.jsonl")
    normalize = lambda body: re.sub(r"\s+","",body).casefold()
    bodies = {normalize(s.notification.body) for s in known}
    known_windows = {(s.context.active_process,s.context.window_title) for s in known if s.context.active_process}
    unrelated_window = ("Code.exe","이미지 업로드 확장자 검사 구현 - Visual Studio Code")
    if unrelated_window in known_windows:
        raise ValueError("new unrelated task window overlaps a source")
    windows = set()
    additions = []
    for domain,target,high,current,soon,reference,links in URGENT_STATE_CASES:
        for urgency,body,stage in ((5,high,"즉시 대응"),(4,current,"현재 진행 확인"),(3,soon,"가까운 일정 준비"),(2,reference,"완료 기록 참고")):
            if normalize(body) in bodies:
                raise ValueError("new state duplicates existing original")
            bodies.add(normalize(body))
            key = "state01_"+hashlib.sha256((domain+"|"+target+"|"+stage).encode("utf-8")).hexdigest()[:16]
            title = target
            # Completed reference records are useful to an explicit record-checking task.
            window = target+" 안내 내용 확인 - Chrome"
            if ("chrome.exe",window) in known_windows:
                raise ValueError("new task window overlaps a frozen source")
            for role,process,context,relevance in (("exact","chrome.exe",window,5),
                                                   ("unrelated","Code.exe","이미지 업로드 확장자 검사 구현 - Visual Studio Code",1),
                                                   ("empty","","",1)):
                s=FilteringSample.model_validate({"notification":{"id":key+"_"+role,"app_name":"안내 메시지","sender":"담당자",
                    "title":title,"body":body,"timestamp":"2026-10-04T01:00:00Z"},
                    "context":{"active_process":process,"window_title":context,"last_updated":"2026-10-04T01:00:00Z",
                               "duration_seconds":60 if process else 0,"recent_processes":[]},
                    "label":{"urgency_score":urgency,"relevance_score":relevance,
                             "category":"개인 중요" if domain in ("건강","안전","금전") else "일정/회의",
                             "ai_summary_reason":stage+" 상태를 알림 자체로 판단; "+("정확한 대상과 안내 확인 작업에 연결됩니다." if role=="exact" else "현재 작업 정보가 없거나 다른 작업과 무관합니다.")}})
                parse_model_output(json.dumps(s.label.model_dump(),ensure_ascii=False))
                additions.append(s)
                lineage.append({"notification_id":s.notification.id,"original_id":key,"domain":domain,"task":target,
                                "event_family":domain+"|"+target,"semantic_group":domain+"|"+target,"state":stage,"train_links":links,
                                "context_role":role,"origin":"authored_event_state_contrast","review_status":"synthetic_unreviewed",
                                "label_source":"approved_boundary_rules_applied_to_new_scenarios","intended_split":"train_only"})
                if process: windows.add((process,context))
    for s in reviewed:
        original_id = s.notification.id.rsplit("_",1)[0]
        family_info = approved_manifest["families"][original_id]
        lineage.append({"notification_id":s.notification.id,"original_id":s.notification.id.rsplit("_",1)[0],
                        "event_family":family_info["family"],"semantic_group":family_info["family"],"train_links":family_info["train_links"],
                        "origin":"approved_urgent_review","review_status":"human_approved","context_role":s.notification.id.rsplit("_",1)[1],
                        "intended_split":"train_only","approval_sha256":digest(approved/"approval.json")})
    all_rows = rows + additions + reviewed
    if len({s.notification.id for s in all_rows}) != len(all_rows):
        raise ValueError("duplicate stable id")
    originals = unique_urgency_samples(all_rows)
    from filtering_training.quality.audit_dataset import audit_samples
    audit = audit_samples(all_rows)
    if audit["contradictory_notification_groups"] or audit["conflicting_identical_inputs"]:
        raise ValueError("contradictory candidate labels")
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl",[s.model_dump(mode="json") for s in all_rows])
    write_lines(output/"lineage.jsonl",lineage)
    write_json(output/"manifest.json",{"purpose":"one combined candidate pool with non-digital event-state contrasts; use instead of parent pool",
        "created_date":"2026-10-04","policy_version":POLICY_VERSION,"status":"synthetic_pool_pending_training_preparation","originals":len(originals),"rows":len(all_rows),
        "parent_candidate_sha256":digest(source/"candidates.jsonl"),"parent_lineage_sha256":digest(source/"lineage.jsonl"),
        "approved_dataset_sha256":digest(approved/"dataset.jsonl"),"approval_sha256":digest(approved/"approval.json"),
        "source_dataset_sha256":digest(original/"dataset.jsonl"),"source_manifest_sha256":digest(original/"prepared/manifest.json"),
        "candidate_sha256":digest(output/"candidates.jsonl"),"lineage_sha256":digest(output/"lineage.jsonl"),"code_sha256":digest(Path(__file__)),
        "added_synthetic_originals":len(unique_urgency_samples(additions)),"added_synthetic_rows":len(additions),
        "approved_urgent_originals":10,"approved_urgent_rows":13,"training_used":False,"test_inference_used":False,
        "all_urgency_counts":dict(Counter(s.label.urgency_score for s in all_rows)),"all_relevance_counts":dict(Counter(s.label.relevance_score for s in all_rows)),
        "new_scenario_domains":dict(Counter(s[0] for s in URGENT_STATE_CASES)),"audit":audit,
        "sampling_plan":{"status":"must be implemented and checked in training preparation",
                         "urgency5_digital_max_fraction":0.25,"reason":"raw pool still dominated by repeated digital incident recipes; cap their sampled contribution without deleting source data"},
        "quality_gate":{"ready_for_training":False,"remaining":["prepare heldout-safe splits and prompts","implement and inspect sampler contribution", "window wording shortcuts remain"]},
        "limitations":["20 scenario targets x 4 event states; not 80 unrelated independently reviewed events",
                        "new synthetic labels are not individual human approvals", "exact overlap checks and declared train links do not prove semantic isolation"],
        "files":{"candidates.jsonl":"combined candidates, approved cases and 80 new event states","lineage.jsonl":"stable original ids, semantic families and review scope"}})
    return all_rows


def generate_validation_review(source, output):
    """Reserve new authored validation candidates before training expansion."""
    if output.exists():
        raise ValueError("do not overwrite an earlier validation review")
    # Paired cases share windows and declared families: keep each entire family
    # outside training, including any later paraphrases or context variants.
    cases = [
        ("printer_queue", "Slack", "인쇄 담당", "인쇄 순서 확인",
         "내일 배포할 교육 안내문 인쇄 순서를 오늘 오후까지 알려주세요. 지금 인쇄 중인 작업은 없습니다.",
         "일반 업무", 3, 1, "EXCEL.EXE", "가계부 식비 항목 정리 - Excel"),
        ("printer_queue", "Slack", "인쇄 담당", "진행 중인 인쇄 중단",
         "교육 안내문 인쇄가 용지 선택 답변을 기다리며 멈춰 있습니다. 담당자가 프린터 앞에 있으니 지금 사용할 용지를 알려주세요. 장비 고장이나 마감 손실은 없습니다.",
         "긴급 업무", 4, 1, "EXCEL.EXE", "가계부 식비 항목 정리 - Excel"),
        ("visitor_interpretation", "Microsoft Teams", "통역 담당", "방문 안내 준비",
         "내일 방문객에게 안내할 출입 절차 번역을 오늘 퇴근 전까지 확인해 주세요. 방문객은 아직 도착하지 않았습니다.",
         "일반 업무", 3, 5, "WINWORD.EXE", "방문객 출입 절차 영문 번역 확인 - Word"),
        ("visitor_interpretation", "Microsoft Teams", "통역 담당", "현장 출입 안내 대기",
         "방문객이 지금 접수대에서 출입 절차 설명을 기다리고 있습니다. 안내 직원이 사용할 영문 문장을 바로 확인해 주세요. 예약 취소나 안전 사고는 없습니다.",
         "긴급 업무", 4, 5, "WINWORD.EXE", "방문객 출입 절차 영문 번역 확인 - Word"),
        ("caption_readability", "Slack", "영상 동료", "자막 가독성 자료",
         "작은 화면에서 자막을 읽기 쉽게 만드는 줄 길이와 대비 조절 방법을 정리했어요. 급한 확인은 아니니 필요할 때 참고하세요.",
         "일반 업무", 2, 4, "DaVinciResolve.exe", "교육 영상 모바일 자막 가독성 조정 - Resolve"),
        ("caption_readability", "쇼핑 앱", "영상 굿즈몰", "자막 디자인 머그컵",
         "자막 편집자용 말풍선 머그컵 출시! 영상 작업 책상에 분위기를 더해 보세요. 구매는 선택입니다.",
         "광고/홍보", 1, 3, "DaVinciResolve.exe", "교육 영상 모바일 자막 가독성 조정 - Resolve"),
        ("boardgame_rules", "KakaoTalk.exe", "게임 모임", "규칙 공부 잡담",
         "별항로 보드게임 규칙 외우다 내 머리도 우주로 떠난 듯ㅋㅋ",
         "개인 일반", 1, 3, "chrome.exe", "별항로 보드게임 자원 교환 규칙 확인 - Chrome"),
        ("boardgame_rules", "KakaoTalk.exe", "게임 모임", "별항로 공식 규칙 답변",
         "별항로에서 자원을 교환한 턴에 이동도 가능한지 설명한 공식 FAQ를 보냈어. 규칙 확인할 때 참고해. 지금 답할 필요는 없어.",
         "개인 일반", 2, 5, "chrome.exe", "별항로 보드게임 자원 교환 규칙 확인 - Chrome"),
        ("gas_isolation", "시설 알림", "관리실", "현재 가스 누출 확인",
         "현재 사용 중인 작업실에서 가스 누출이 확인됐습니다. 작업을 중단하고 즉시 밖으로 나와 관리실 안내를 따라주세요. 점검 훈련이 아닙니다.",
         "개인 중요", 5, 1, "", ""),
        ("gas_isolation", "시설 알림", "관리실", "가스 차단 후 점검 일정",
         "작업실 가스 공급은 이미 차단됐고 누출 조치와 안전 확인이 끝났습니다. 내일 재점검 방문 시간을 오늘 저녁까지 알려주세요. 지금 대피할 상황은 아닙니다.",
         "개인 중요", 3, 1, "", ""),
    ]
    protected = [source/"dataset.jsonl",
                 source.parent/"v3_score_review_02_approved_01/dataset.jsonl",
                 source.parent/"v3_expansion_review_01_approved/dataset.jsonl",
                 source.parent/"v4_urgent_review_01_approved/dataset.jsonl",
                 source.parent/"v4_training_pool_01/candidates.jsonl"]
    if not all(path.exists() for path in protected):
        raise ValueError("all current source datasets must be available for overlap checks")
    known = [sample for path in protected for sample in load_samples(path)]
    normalize = lambda text: re.sub(r"\s+", "", text).casefold()
    bodies = {normalize(s.notification.body) for s in known}
    windows = {(s.context.active_process,s.context.window_title) for s in known if s.context.active_process}
    timestamp = "2026-10-04T12:00:00Z"
    samples = []
    for index, case in enumerate(cases, 1):
        family, app, sender, title, body, category, urgency, relevance, process, window = case
        for role, proc, win, rel in (("primary",process,window,relevance),
                                      ("unrelated","Notion.exe","가정 채소 파종 일정 정리 - Notion",1),
                                      ("empty","","",1)):
            samples.append(FilteringSample.model_validate({
                "notification":{"id":f"val05_{index:02d}_{role}","app_name":app,"sender":sender,
                                "title":title,"body":body,"timestamp":timestamp},
                "context":{"active_process":proc,"window_title":win,"last_updated":timestamp,
                           "duration_seconds":60 if proc else 0,"recent_processes":[]},
                "label":{"urgency_score":urgency,"relevance_score":rel,"category":category,
                         "ai_summary_reason":"사용자 검수 전 제안 점수; 긴급도는 알림만, 관련도는 현재 목적 기준."}}))
    if any(normalize(s.notification.body) in bodies for s in samples):
        raise ValueError("validation candidate overlaps an existing source body")
    if any((s.context.active_process,s.context.window_title) in windows for s in samples if s.context.active_process):
        raise ValueError("validation candidate overlaps an existing populated window")
    unique_urgency_samples(samples)
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl",[s.model_dump(mode="json") for s in samples])
    lines = ["<!-- 용도: 학습에서 제외할 새 검증 원문의 첫 검수 묶음.\n생성일: 2026-10-04 (Asia/Seoul)\n상태: 사용자 미승인, 학습·모델 평가 미사용. -->\n",
             "# 새 검증 자료 1차 검수\n",
             "원문 10개·문맥 30행. 집중 모드 ON이며 빈 창의 관련도는 1입니다. 긴급도는 세 문맥에서 유지합니다. 모든 점수는 검수 전 제안입니다.\n",
             "| 번호 | 알림 원문 | 주 문맥 | 긴급도 | 관련도 | 판단 |",
             "| --- | --- | --- | ---: | ---: | --- |"]
    for i,case in enumerate(cases,1):
        family, app, sender, title, body, category, urgency, relevance, process, window = case
        lines.append(f"| {i} | {body} | {window or '(빈 창)'} | {urgency} | {relevance} | {decision_label(urgency,relevance)} |")
    lines += ["", "무관 문맥은 `가정 채소 파종 일정 정리 - Notion`, 빈 창은 프로세스·창 제목 모두 빈 값입니다. 두 변형의 관련도는 모두 1입니다.\n",
              "## 확장 구성\n",
              "- 새 검증 목표: 원문 100개. 이번 10개는 첫 묶음이며 전체 100개가 준비된 상태는 아닙니다. 동일 사건 계열·창·문맥 변형은 학습에서 제외합니다.\n",
              "- 학습 후보 목표: 기존 통합 원문 5,090개를 보존하고 새 원문 4,910개를 추가해 10,000개. 문맥 변형은 원문 수에 포함하지 않습니다.\n",
              "- 추가 원문 구성: 긴급도 3↔4 대비 1,600개, 비디지털 긴급 사건 1,200개, 직접 도움과 주제 일치 대비 1,600개, 일반 비긴급 510개. 이 수는 생성 계획이며 생성 완료 건수가 아닙니다.\n",
              "- 다양성 목표: 25개 이상 세부 영역, 단일 영역 10% 이하, 동일 사건·문장 틀 최대 20개. 숫자·이름만 바꾼 사례는 별도 다양성 확보로 간주하지 않습니다.\n",
              "- 학습 비교: 같은 확장 자료로 기존 어댑터 누적 학습과 기반 모델에서 새 어댑터 학습을 비교합니다. 기존·신규 자료를 함께 사용합니다. 라벨 확정 전 평가나 학습은 하지 않습니다.\n",
              "- 최근 창 비교: 사용자 확인으로 앱 이름과 각 창 제목을 수집할 수 있습니다. 현재 스키마는 최근 앱 이름 최대 3개만 지원하며 기존 통합 후보 5,753행은 모두 빈 목록입니다. 확장 입력 구현 후 같은 알림·현재 창을 유지한 채 기록 없음/앱 이름만/앱과 창 제목을 비교합니다. 작업 연속, 다른 작업으로 전환, 같은 앱의 다른 창을 함께 포함합니다. 최근 기록만으로 현재 목적을 단정하지 않습니다. 긴급도에는 추가하지 않습니다. 이번 30행에는 최근 창 정보가 아직 없습니다.\n",
              "## 격리 범위\n",
              "기존 자료와 정규화 본문·정보 창의 완전 일치가 없음을 확인했습니다. 의미상 모든 유사성이 배제됐다는 뜻은 아닙니다. 지정한 5개 사건 계열은 이후 생성 자료와 함께 다시 감사해야 합니다. 기존 최종 테스트는 유지합니다.\n"]
    (output/"review.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"manifest.json",{
        "purpose":"reserved validation candidate review, never a training supplement","created_date":"2026-10-04",
        "status":"unapproved_validation_candidates","intended_split":"validation","policy_version":POLICY_VERSION,
        "originals":10,"rows":len(samples),"human_reviewed_originals":0,"training_used":False,"model_evaluation_used":False,
        "candidate_sha256":digest(output/"candidates.jsonl"),"review_sha256":digest(output/"review.md"),
        "protected_sources":[{"path":str(p),"sha256":digest(p)} for p in protected],
        "exact_body_overlap":0,"populated_window_overlap":0,
        "reserved_families":{family:[f"val05_{i:02d}" for i,c in enumerate(cases,1) if c[0]==family] for family in sorted({c[0] for c in cases})},
        "expansion_plan":{"validation_originals_target":100,"existing_pool_originals":5090,
                          "new_training_originals_target":4910,"combined_pool_originals_target":10000,
                          "new_training_originals_generated":0,"training_blocked_until_review":True},
        "recent_context_plan":{"schema_fields":["recent_processes"],"max_apps":3,
                               "comparison":"matched notification/current window: no history versus app names versus names and titles",
                               "available_fields":["app_name","window_title"],"implementation_status":"planned_not_in_current_candidates",
                               "urgency_context_excluded":True,"history_alone_does_not_establish_relevance4":True},
        "files":{"candidates.jsonl":"provisional validation rows, not training data","review.md":"review table and expansion plan"}})
    return samples


def generate_history_review(source, output):
    """Author paired history cases without altering existing approved validation."""
    if output.exists():
        raise ValueError("do not overwrite an earlier history review")
    source_manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    approval = json.loads((source/"approval.json").read_text(encoding="utf-8"))
    if approval.get("intended_split") != "validation" or digest(source/"dataset.jsonl") != approval["dataset_sha256"]:
        raise ValueError("history preview requires the approved validation source")
    cases = [
        ("visitor_interpretation","Teams","통역 담당","방문 안내 참고 문장",
         "견학 참가자에게 안전모 착용 위치를 설명하는 영문 예문을 보냈습니다. 안내 문구 작성에 참고하세요. 지금 답할 필요는 없습니다.",
         "일반 업무",2,"WINWORD.EXE","견학 안전 안내 영문 문구 작성 - Word",4,
         [("chrome.exe","견학 안전모 착용 지점 안내 - Chrome"),("AcroRd32.exe","견학 이동 동선.pdf"),("ms-teams.exe","견학 안내 문구 검토 - Teams")]),
        ("caption_readability","Slack","영상 동료","자막 대비 참고",
         "밝은 배경에서 자막 테두리와 음영을 조절하는 방법을 정리해 뒀어요. 영상 자막 수정에 참고하세요. 급한 요청은 아닙니다.",
         "일반 업무",2,"DaVinciResolve.exe","야외 촬영 교육 영상 자막 대비 조정 - Resolve",4,
         [("chrome.exe","밝은 배경 자막 대비 예시 - Chrome"),("explorer.exe","야외 교육 영상 소스 - 파일 탐색기"),("WINWORD.EXE","야외 교육 영상 대본 - Word")]),
        ("caption_readability","쇼핑 앱","영상 굿즈몰","영상 편집 스티커",
         "자막 편집 단축키 모양 스티커 출시! 노트북을 꾸며 보세요. 영상 편집 기능이나 강의는 제공하지 않습니다.",
         "광고/홍보",1,"DaVinciResolve.exe","야외 촬영 교육 영상 자막 대비 조정 - Resolve",3,
         [("chrome.exe","밝은 배경 자막 대비 예시 - Chrome"),("explorer.exe","야외 교육 영상 소스 - 파일 탐색기"),("WINWORD.EXE","야외 교육 영상 대본 - Word")]),
        ("boardgame_rules","KakaoTalk.exe","게임 모임","별항로 탐사 카드 FAQ",
         "별항로에서 탐사 카드를 쓴 뒤 이동할 수 있는지 설명한 공식 FAQ 링크야. 그 규칙 확인할 때 참고해. 급하게 답하지 않아도 돼.",
         "개인 일반",2,"chrome.exe","별항로 탐사 카드 사용 후 이동 규칙 확인 - Chrome",5,
         [("AcroRd32.exe","별항로 탐사 카드 규칙.pdf"),("Notion.exe","별항로 카드와 이동 질문 정리 - Notion"),("KakaoTalk.exe","별항로 규칙 질문 모임")]),
        ("gas_isolation","시설 알림","관리실","점검 완료 회신",
         "가스 설비 점검이 끝났고 정상 상태를 확인했습니다. 다음 주 점검 가능 시간을 오늘 저녁까지 알려주세요. 현재 누출이나 대피 요청은 없습니다.",
         "개인 중요",3,"","",1,
         [("chrome.exe","작업실 가스 설비 점검 신청 - Chrome"),("EXCEL.EXE","작업실 점검 가능 시간 - Excel"),("ms-teams.exe","시설 점검 일정 협의 - Teams")]),
    ]
    irrelevant = [("chrome.exe","주말 빵 반죽 발효 시간 - Chrome"),
                  ("EXCEL.EXE","주간 식재료 비용 - Excel"),("Notion.exe","빵 레시피 정리 - Notion")]
    timestamp = "2026-10-04T12:00:00Z"
    rows = []
    reviewed_relevance = {
        (1,"distracting"):1, (2,"distracting"):1,
        (3,"supporting"):2, (3,"distracting"):1,
        (4,"distracting"):1, (5,"supporting"):5,
    }
    for i,case in enumerate(cases,1):
        family,app,sender,title,body,category,urgency,process,window,relevance,history = case
        for role,entries in (("supporting",history),("distracting",irrelevant)):
            sample = HistorySample.model_validate({
                "notification":{"id":f"hist05_{i:02d}_{role}","app_name":app,"sender":sender,
                                "title":title,"body":body,"timestamp":timestamp},
                "context":{"active_process":process,"window_title":window,"last_updated":timestamp,
                           "duration_seconds":60 if process else 0,"recent_processes":[app for app,_ in entries],
                           "recent_windows":[{"app_name":app,"window_title":title} for app,title in entries]},
                "label":{"urgency_score":urgency,"relevance_score":reviewed_relevance.get((i,role),relevance),"category":category,
                         "ai_summary_reason":"제안: 현재 창과 최근 창의 작업 대상·행동에 알림이 주는 도움으로 판단. 명시적으로 수정된 사례의 관련도는 사용자 검수 기준을 반영."}})
            rows.append(sample)
    known_paths = [Path(item["path"]) for item in source_manifest["protected_sources"]] + [source/"dataset.jsonl"]
    normalize = lambda text: re.sub(r"\s+","",text).casefold()
    known_bodies = {normalize(s.notification.body) for p in known_paths for s in load_samples(p)}
    if any(normalize(s.notification.body) in known_bodies for s in rows):
        raise ValueError("history preview body overlaps an existing original")
    unique_urgency_samples(rows)
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl",[s.model_dump(mode="json") for s in rows])
    lines = ["<!-- 용도: 최근 앱·창 제목 입력과 무관한 기록·빈 창 처리 기준 검수.\n생성일: 2026-10-04 (Asia/Seoul)\n상태: 사용자 미승인, 학습·모델 평가 미사용. -->\n",
             "# 최근 창 입력 검수 10건\n",
             "원문 5개 × 최근 기록 2종 = 검수 10건입니다. 번호별 점수는 제안입니다. 기록 순서는 수집 순서를 유지하되 시각이나 체류 시간을 만들어 넣지 않습니다.\n",
             "현재 창과 최근 창의 작업 흐름을 함께 봅니다. 최근 기록이 다른 목적에 연결되면 현재 창의 단어 일치만으로 관련도를 고정하지 않습니다. 빈 창이어도 최근 기록이 정확한 대상·행동에 연결되면 높은 관련도가 가능합니다. 앱 개수의 다수결이나 입력에 없는 시각으로 목적을 추측하지 않습니다.\n"]
    for index,s in enumerate(rows,1):
        lines += [f"## {index}. {s.notification.title}\n",f"알림: {s.notification.body}\n",
                  f"현재 창: {s.context.window_title or '(빈 창)'}\n",
                  "| 최근 앱 | 창 제목 |","| --- | --- |"]
        lines += [f"| {w.app_name} | {w.window_title} |" for w in s.context.recent_windows]
        lines += ["",f"제안: 긴급도 **{s.label.urgency_score}**, 관련도 **{s.label.relevance_score}**, **{decision_label(s.label.urgency_score,s.label.relevance_score)}**.\n"]
    lines += ["## 입력 비교와 보존\n",
              "동일 원문·현재 창·정답을 유지하고 입력만 기록 없음/앱 이름만/앱과 창 제목으로 바꿔 비교합니다. 이번 묶음은 최근 기록에 끌려 잘못 판단하는지 확인하는 사례이며, 제목 추가의 정확도 개선을 입증한 결과가 아닙니다.\n",
              "5개 원문은 기존 검증 사건 계열에만 연결하며 학습에 넣지 않습니다. 기존 승인 10개 원문과 그 점수는 수정하지 않습니다. 오프라인 학습용 확장 스키마에서 최근 창을 보존하며 앱 실행 경로는 아직 변경하지 않습니다.\n"]
    (output/"review.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"manifest.json",{
        "purpose":"offline recent-window validation preview; not training data","created_date":"2026-10-04",
        "intended_split":"validation","status":"unapproved_history_candidates","originals":5,"rows":10,
        "schema":"HistorySample","history_fields":["app_name","window_title"],"max_recent_windows":3,
        "label_basis":"author proposals using six explicitly user-corrected history cases; new generation is not individually approved",
        "candidate_sha256":digest(output/"candidates.jsonl"),"review_sha256":digest(output/"review.md"),
        "approved_source_sha256":digest(source/"dataset.jsonl"),"approved_source_approval_sha256":digest(source/"approval.json"),
        "protected_sources":[{"path":str(p),"sha256":digest(p)} for p in known_paths],
        "reserved_families":{family:[f"hist05_{i:02d}" for i,c in enumerate(cases,1) if c[0]==family] for family in sorted({c[0] for c in cases})},
        "policy_version":POLICY_VERSION,"training_used":False,"model_evaluation_used":False,
        "files":{"candidates.jsonl":"history rows; read using HistorySample, not FilteringSample",
                 "review.md":"ten displayed history cases"}})
    return rows


def revise_history_review(source, feedback_path):
    """Keep original candidates; save user corrections in the same batch folder."""
    from filtering_training.common.score_tasks import load_score_samples
    if feedback_path.resolve().parent != source.resolve():
        raise ValueError("history feedback must stay in its batch folder")
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    if (source/"approval.json").exists():
        raise ValueError("do not overwrite approved history")
    if manifest.get("schema") != "HistorySample" or digest(source/"candidates.jsonl") != manifest["candidate_sha256"]:
        raise ValueError("history source changed")
    previous_revision = manifest.get("revision")
    expected_review = previous_revision["review_sha256"] if previous_revision else manifest["review_sha256"]
    if digest(source/"review.md") != expected_review:
        raise ValueError("original history review changed")
    if previous_revision:
        if digest(source/"revised.jsonl") != previous_revision["dataset_sha256"]:
            raise ValueError("previous revised history changed")
        feedback.setdefault("revision_history",[]).append({
            "revision_metadata":previous_revision,
            "revised_rows":[json.loads(line) for line in (source/"revised.jsonl").read_text(encoding="utf-8").splitlines()],
            "review":(source/"review.md").read_text(encoding="utf-8")})
    samples = load_score_samples(source/"candidates.jsonl")
    corrections = feedback["corrections"]
    numbers = [item["review_number"] for item in corrections]
    if len(numbers) != len(set(numbers)):
        raise ValueError("duplicate correction number")
    for item in corrections:
        number = item["review_number"]
        if type(number) is not int or not 1 <= number <= len(samples):
            raise ValueError("invalid review number")
        sample = samples[number-1]
        if sample.notification.id != item["notification_id"] or sample.label.relevance_score != item["before"]:
            raise ValueError("correction does not refer to the displayed original")
        if type(item["after"]) is not int or not 1 <= item["after"] <= 5:
            raise ValueError("invalid corrected relevance")
        sample.label.relevance_score = item["after"]
        sample.label.ai_summary_reason = item["reason"]
    unique_urgency_samples(samples)
    original_review = feedback.get("original_review",(source/"review.md").read_text(encoding="utf-8"))
    revised_review = original_review
    for item in corrections:
        number = item["review_number"]
        start = revised_review.index(f"## {number}. ")
        end = revised_review.find("\n## ", start+1)
        end = len(revised_review) if end < 0 else end
        section = revised_review[start:end]
        sample = samples[number-1]
        section = re.sub(r"제안: 긴급도 \*\*\d\*\*, 관련도 \*\*\d\*\*, \*\*(?:PASS|BLOCK)\*\*\.",
                         f"사용자 수정: 긴급도 **{sample.label.urgency_score}**, 관련도 **{sample.label.relevance_score}**, **{decision_label(sample.label.urgency_score,sample.label.relevance_score)}**.\n\n수정 근거: {item['reason']}",section)
        revised_review = revised_review[:start]+section+revised_review[end:]
    revised_review = revised_review.replace(
        "관련도는 현재 목적 기준입니다. 최근 창은 보조 근거이며 무관한 과거 기록 때문에 명확한 현재 작업의 관련도를 낮추지 않습니다. 빈 창에서는 과거 목적을 현재 목적이라고 추정하지 않습니다.",
        "관련도는 현재 창과 최근 앱·창 제목을 함께 보고 판단합니다. 사용자 검수에서는 2·4·6·8번의 최근 작업 흐름을 무관하게 판단했고, 9번은 빈 창이어도 최근 기록이 정확한 점검 일정 작업에 연결돼 관련도 5로 지정했습니다. 2번의 ‘0’은 완전 무관이라는 뜻으로 확인되어 기존 최저점 1을 사용합니다. 이 수정은 해당 사례에 적용하며 모든 빈 창을 관련도 5로 취급하거나 최근 기록이 항상 현재 창보다 우선한다는 규칙으로 일반화하지 않습니다.")
    revised_review = revised_review.replace(
        "동일 원문·현재 창·정답을 유지하고 입력만 기록 없음/앱 이름만/앱과 창 제목으로 바꿔 비교합니다.",
        "사용자 수정으로 같은 원문·현재 창이어도 최근 창 내용에 따라 정답이 달라집니다. 수정된 전체 문맥 정답을 고정하고 입력 정보만 기록 없음/앱 이름만/앱과 창 제목으로 바꿔 비교합니다. 기록을 뺀 입력에서는 정답을 판단할 근거가 부족할 수 있음을 별도 표시합니다.")
    # Store exact original review and manifest inside the feedback artifact;
    # candidates.jsonl remains untouched, with revised labels in a separate file.
    feedback.update(purpose="explicit history-label corrections with original review preserved",created_date="2026-10-04",
                    original_review=original_review,original_manifest=feedback.get("original_manifest",manifest),training_used=False,model_evaluation_used=False)
    write_json(feedback_path,feedback)
    write_lines(source/"revised.jsonl",[s.model_dump(mode="json") for s in samples])
    (source/"review.md").write_text(revised_review,encoding="utf-8")
    manifest["revision"] = {"dataset_file":"revised.jsonl","dataset_sha256":digest(source/"revised.jsonl"),
                            "review_sha256":digest(source/"review.md"),"feedback_file":feedback_path.name,
                            "feedback_sha256":digest(feedback_path),"explicitly_corrected_rows":len(corrections),
                            "status":"user_corrections_applied_remaining_labels_provisional"}
    write_json(source/"manifest.json",manifest)
    return samples


def generate_validation_pool(source, output):
    """Combine approved seeds with 85 authored originals reserved for validation."""
    from filtering_training.common.score_tasks import load_score_samples
    if output.exists():
        raise ValueError("do not overwrite a validation pool")
    seed_folders = [source.parent/"v5_validation_review_01",source]
    protected = []
    seed_rows = []
    families = {}
    for folder in seed_folders:
        approval = json.loads((folder/"approval.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        if approval.get("intended_split") != "validation" or digest(folder/"dataset.jsonl") != approval["dataset_sha256"]:
            raise ValueError("validation seed approval changed")
        for item in manifest["protected_sources"]:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError("protected source changed after validation review")
        seed_rows.extend(load_score_samples(folder/"dataset.jsonl"))
        protected += [Path(item["path"]) for item in manifest["protected_sources"]]
        protected += [folder/"dataset.jsonl",folder/"approval.json"]
        for family,ids in manifest["reserved_families"].items():
            families.setdefault(family,[]).extend(ids)
    if len(unique_urgency_samples(seed_rows)) != 15:
        raise ValueError("expected 15 approved seed originals")
    # Every notification body is authored below. Context variants are not counted
    # as originals. Families must remain outside the later training expansion.
    recipes = [
        ("coffee_sample_roast","로스팅","Artisan.exe","R21 원두 샘플 배전 곡선 조정 - Artisan",
         "R21 원두의 1차 크랙 구간이 표시된 샘플 기록을 보냈어요. 지금 비교하는 배전 곡선에 참고하세요. 급한 답은 필요 없습니다.",
         "내일 시음에 쓸 R21 원두의 샘플 수량을 오늘 저녁까지 알려주세요. 아직 로스팅은 시작하지 않았습니다.",
         "R21 샘플 포장이 담당자 수량 답변을 기다리며 멈췄습니다. 작업자가 대기 중이니 지금 수량을 알려주세요. 사고나 마감 손실은 없습니다."),
        ("pottery_glaze_tiles","도예","chrome.exe","청색 유약 시험편 발색 비교 - Chrome",
         "유약 시험편을 같은 조명에서 비교하는 촬영 방법을 정리했어요. 발색을 비교할 때 참고하세요. 나중에 확인해도 됩니다.",
         "청색 유약 시험편에 붙일 번호 목록을 오늘 퇴근 전까지 확인해 주세요. 다음 소성은 내일입니다.",
         "청색 유약 시험편 분류가 번호 확인을 기다리고 있습니다. 담당자가 앞에서 기다리니 사용할 번호를 바로 알려주세요. 설비 이상은 없습니다."),
        ("apiary_hive_records","양봉","EXCEL.EXE","B6 벌통 점검 기록 정리 - Excel",
         "벌통 기록 정리하다가 나도 일벌처럼 야근하는 기분ㅋㅋ 답 안 해도 돼요.",
         "B6 벌통 점검표의 빈 날짜 칸을 오늘 저녁까지 채워주세요. 다음 현장 확인은 내일입니다.",
         "현장 담당자가 B6 점검 기록을 넘기려고 기다리고 있습니다. 인계받을 사람을 지금 알려주세요. 벌통 사고는 없습니다."),
        ("sailboat_rope_inventory","요트","EXCEL.EXE","S4 요트 계류 로프 길이 점검 - Excel",
         "요트 로프 모양 열쇠고리 출시! 항해 분위기를 내 보세요. 계류 장비로 사용할 수 없는 장식품입니다.",
         "S4 요트용 계류 로프의 재고 길이를 오늘 오후까지 확인해 주세요. 정박 작업은 내일 예정입니다.",
         "S4 로프 인계 담당자가 창고 앞에서 기다립니다. 가져갈 로프 번호를 지금 알려주세요. 선박 위험이나 출항 마감은 없습니다."),
        ("braille_leaflet_layout","점자","WINWORD.EXE","전시 안내 점자 리플릿 줄 배치 - Word",
         "이번 주말에 동네 축구 보러 갈 사람? 참가 여부는 아무 때나 알려줘.",
         "점자 리플릿의 페이지 수를 오늘 저녁까지 확인해 주세요. 인쇄는 내일 시작합니다.",
         "점자 리플릿 시안 전달이 페이지 수 확인을 기다리고 있습니다. 담당자가 연결돼 있으니 지금 수를 알려주세요. 인쇄 마감은 아닙니다."),
        ("herbarium_label_matching","표본","EXCEL.EXE","H8 식물 표본 채집지 라벨 대조 - Excel",
         "H8 표본의 채집지 원본 기록을 공유했어요. 현재 대조하는 라벨 번호도 적어뒀습니다. 급한 요청은 아닙니다.",
         "H8 표본 라벨의 채집 날짜를 오늘 퇴근 전까지 확인해 주세요. 보관함 정리는 내일입니다.",
         "H8 표본함을 인계할 직원이 도착했습니다. 인수할 사람을 지금 알려주세요. 표본 손상이나 반출 마감은 없습니다."),
        ("mosaic_cutting_plan","모자이크","Inkscape.exe","벽면 모자이크 타일 절단 배치 - Inkscape",
         "모자이크 타일 간격을 일정하게 배치하는 방법을 정리했어요. 도안 검토에 참고하세요. 나중에 보셔도 됩니다.",
         "벽면 모자이크에 쓸 타일 색상 수량을 오늘 오후까지 알려주세요. 절단은 내일부터 합니다.",
         "모자이크 재료 배분이 색상 수량 답변을 기다리고 있습니다. 담당자에게 지금 수량을 알려주세요. 장비 이상은 없습니다."),
        ("aquarium_feeding_chart","수조","EXCEL.EXE","A3 수조 사료 급여량 기록 비교 - Excel",
         "물고기 밥 기록하다 보니 내 점심 메뉴도 표로 만들고 싶네ㅋㅋ",
         "A3 수조 기록에서 지난 주 급여량을 오늘 저녁까지 확인해 주세요. 급여 일정 변경은 내일 논의합니다.",
         "A3 급여 기록을 인계할 관리자가 대기 중입니다. 기록을 받을 사람을 지금 알려주세요. 생물 이상이나 급여 중단은 없습니다."),
        ("clarinet_reed_trials","악기","EXCEL.EXE","클라리넷 리드 강도별 연주 기록 비교 - Excel",
         "리드 모양 책갈피 출시! 악보집을 꾸미는 장식품이며 악기 부품은 아닙니다.",
         "내일 리드 비교 연습에 사용할 악보 목록을 오늘 저녁까지 알려주세요. 지금 연습 중인 세션은 없습니다.",
         "리드 비교 연습 자료 전달이 악보 목록 답변을 기다립니다. 담당자가 통화 중이니 바로 목록을 확인해 주세요. 공연 마감은 없습니다."),
        ("meteor_camera_alignment","천체촬영","chrome.exe","유성 관측 카메라 촬영 구도 조정 - Chrome",
         "동네 빵집 새 쿠키 맛있더라. 언제든 시간 나면 먹어봐.",
         "내일 유성 관측 카메라 배치도를 오늘 저녁까지 확인해 주세요. 아직 현장 설치는 시작하지 않았습니다.",
         "관측 카메라 설치 담당자가 장비를 전달하려고 기다립니다. 인계 장소를 지금 알려주세요. 기상 위험이나 촬영 마감은 없습니다."),
        ("museum_mount_dimensions","전시","FreeCAD.exe","M12 전시 받침대 치수 검토 - FreeCAD",
         "M12 받침대의 바닥 체결 위치가 표시된 도면을 올렸습니다. 지금 검토하는 치수와 함께 보세요. 급하지 않습니다.",
         "M12 받침대 치수표의 단위를 오늘 오후까지 확인해 주세요. 제작은 다음 주입니다.",
         "M12 받침대 시안 인계가 단위 확인 답변을 기다립니다. 검토자가 접속 중이니 지금 단위를 알려주세요. 제작 오류는 아직 발생하지 않았습니다."),
        ("archive_scan_exposure","기록물","Photoshop.exe","옛 엽서 스캔 노출 보정 비교 - Photoshop",
         "낡은 종이 스캔에서 밝기를 비교하는 방법을 정리했어요. 노출 보정에 참고하세요. 시간 날 때 확인하시면 됩니다.",
         "옛 엽서 스캔 파일의 순서를 오늘 퇴근 전까지 알려주세요. 공개 작업은 내일입니다.",
         "엽서 스캔 파일 전달이 순서 답변을 기다리고 있습니다. 담당자가 연결돼 있으니 지금 순서를 알려주세요. 원본 손상은 없습니다."),
        ("kennel_shift_roster","보호소","EXCEL.EXE","K2 보호소 산책 담당표 조정 - Excel",
         "강아지 산책 담당표 짜다 내가 산책 가고 싶어지는 중ㅋㅋ",
         "K2 보호소 내일 산책 담당 시간을 오늘 저녁까지 알려주세요. 오늘 담당은 이미 배정됐습니다.",
         "K2 다음 근무자 인계가 담당자 확인을 기다립니다. 교대 직원이 현장에 있으니 지금 인수 담당을 알려주세요. 동물 응급 상황은 없습니다."),
        ("charity_meal_portions","급식","EXCEL.EXE","나눔 급식 식재료 분량 산출 - Excel",
         "국그릇 모양 마그넷 신상품! 냉장고 장식용이며 조리 도구는 아닙니다.",
         "내일 나눔 급식에 쓸 채소 분량을 오늘 오후까지 알려주세요. 조리는 내일 시작합니다.",
         "급식 재료 인계 담당자가 도착해 기다립니다. 받아갈 장소를 지금 알려주세요. 식재료 이상이나 배식 마감은 없습니다."),
        ("terrain_sensor_positions","측량","QGIS.exe","T7 경사면 센서 설치 위치 검토 - QGIS",
         "이번 주말 동네 영화 모임 모집해요. 신청은 선택이고 언제든 연락 주세요.",
         "T7 센서 설치 위치표를 오늘 퇴근 전까지 확인해 주세요. 현장 설치는 다음 주입니다.",
         "T7 센서 자료 전달이 담당자 답변을 기다립니다. 설치 담당자가 접속 중이니 받을 사람을 지금 알려주세요. 지반 위험은 없습니다."),
        ("bookbinding_fold_sequence","제본","InDesign.exe","Z3 소책자 접지 순서 확인 - InDesign",
         "Z3 소책자 접지 순서가 표시된 펼침 도면을 보냈어요. 현재 확인하는 페이지 배열과 대조해 보세요. 급하지 않습니다.",
         "Z3 소책자 종이 종류를 오늘 저녁까지 알려주세요. 인쇄는 내일 예정입니다.",
         "Z3 접지 샘플 인계가 종이 종류 확인을 기다리고 있습니다. 담당자에게 지금 답해주세요. 제본 사고나 마감은 없습니다."),
        ("dance_rehearsal_spacing","무용","chrome.exe","군무 리허설 무대 간격 확인 - Chrome",
         "여러 사람이 무대에서 간격을 맞추는 연습 방법을 정리했어요. 리허설 준비에 참고하세요. 나중에 보셔도 됩니다.",
         "내일 군무 리허설 동선 순서를 오늘 저녁까지 확인해 주세요. 공연은 다음 달입니다.",
         "군무 리허설 진행자가 연결돼 대기 중입니다. 시작할 대형 번호를 지금 알려주세요. 부상이나 공연 마감은 없습니다."),
        ("bicycle_wheel_log","정비","EXCEL.EXE","자전거 W5 휠 점검 기록 대조 - Excel",
         "휠 점검 기록하다 내 집중력도 빙글빙글 도네ㅋㅋ",
         "W5 휠 점검표의 측정 날짜를 오늘 오후까지 확인해 주세요. 실제 정비는 내일입니다.",
         "W5 점검 자료 인계가 받을 사람 확인을 기다립니다. 작업자가 통화 중이니 지금 담당을 알려주세요. 주행 사고는 없습니다."),
        ("radio_show_cue_sheet","방송","EXCEL.EXE","라디오 R9 녹음 큐시트 순서 정리 - Excel",
         "마이크 모양 쿠션 출시! 방송 책상을 꾸미는 장식용이며 녹음 장비가 아닙니다.",
         "R9 녹음 큐시트의 코너 순서를 오늘 저녁까지 알려주세요. 녹음은 내일입니다.",
         "R9 자료 전달 담당자가 대기 중입니다. 보내줄 파일 순서를 지금 알려주세요. 방송 송출이나 녹음 마감은 없습니다."),
        ("shoe_pattern_seams","재봉","Inkscape.exe","실내화 P4 재봉선 도안 확인 - Inkscape",
         "동네 산책 모임에서 주말에 사진 찍으러 가요. 참가 여부는 아무 때나 알려줘요.",
         "P4 실내화 도안의 사이즈 표기를 오늘 오후까지 확인해 주세요. 재단은 다음 주입니다.",
         "P4 도안 전달이 사이즈 확인 답변을 기다립니다. 담당자가 연결돼 있으니 지금 표기를 알려주세요. 생산 마감은 없습니다."),
        ("parcel_fragile_packing","포장","EXCEL.EXE","F2 유리 소품 완충 포장 수량 검토 - Excel",
         "F2 유리 소품의 완충재 배치 사진을 올렸어요. 지금 검토하는 포장 수량과 함께 참고하세요. 급하지 않습니다.",
         "F2 소품 포장 상자 수량을 오늘 저녁까지 확인해 주세요. 출하는 내일입니다.",
         "F2 포장 자료 인계 직원이 대기 중입니다. 상자 번호를 지금 알려주세요. 파손이나 출하 마감은 없습니다."),
        ("orchard_pruning_diagram","과수","chrome.exe","O8 과수 가지치기 위치 도면 확인 - Chrome",
         "가지치기 도면에 작업 순서를 표시하는 방법을 정리했어요. 위치 검토에 참고하세요. 나중에 확인하셔도 됩니다.",
         "O8 과수 도면의 나무 번호를 오늘 오후까지 확인해 주세요. 현장 작업은 다음 주입니다.",
         "O8 도면 인계가 나무 번호 확인 답변을 기다립니다. 담당자가 통화 중이니 지금 번호를 알려주세요. 작업 사고는 없습니다."),
        ("canoe_route_briefing","카누","chrome.exe","C7 카누 체험 이동 경로 안내 작성 - Chrome",
         "카누 안내문 쓰다 내 글도 물길처럼 흘러가면 좋겠다ㅋㅋ",
         "C7 카누 체험 안내문의 집합 장소를 오늘 저녁까지 확인해 주세요. 체험은 다음 주입니다.",
         "C7 안내문 전달 담당자가 대기 중입니다. 사용할 집합 장소 문구를 지금 알려주세요. 참가자는 아직 출발하지 않았습니다."),
        ("theater_prop_labels","연극","EXCEL.EXE","L6 연극 소품 보관 라벨 대조 - Excel",
         "연극 무대 모양 스티커 출시! 소품 기능이 없는 장식용 상품입니다.",
         "L6 연극 소품 목록의 보관 번호를 오늘 오후까지 확인해 주세요. 공연은 다음 달입니다.",
         "L6 소품 자료를 전달할 담당자가 기다립니다. 받을 사람을 지금 알려주세요. 공연 진행이나 안전 사고는 없습니다."),
        ("ceramic_delivery_schedule","배송","EXCEL.EXE","D8 도자기 배송 방문 시간 조율 - Excel",
         "새로 나온 빵집 메뉴 사진 보냈어. 아무 때나 구경해 봐.",
         "D8 도자기 배송에 가능한 방문 시간을 오늘 저녁까지 알려주세요. 배송은 다음 주입니다.",
         "D8 배송 안내 자료 인계가 담당자 확인을 기다립니다. 직원이 통화 중이니 지금 담당을 알려주세요. 배송 중 파손은 없습니다."),
    ]
    critical = [
        "로스팅 작업실에서 실제 화재가 확인됐습니다. 작업실에 있는 사람은 즉시 밖으로 대피하세요. 훈련이 아닙니다.",
        "도예 작업실 천장 구조물이 지금 무너져 내리고 있습니다. 작업을 멈추고 즉시 안전한 곳으로 이동하세요.",
        "벌통 점검 중 직원이 의식을 잃어 구급대가 도착했습니다. 현장 보호자 확인이 필요하니 지금 연락 받아주세요.",
        "정박 중인 S4 요트 안으로 물이 빠르게 들어오고 있습니다. 탑승자는 즉시 현장 구조 안내를 따라주세요.",
        "이미 신청한 점자 안내물 제작 지원의 필수 동의서가 누락됐습니다. 3분 뒤 보완이 마감되고 이번 신청이 취소되니 지금 제출해 주세요.",
        "식물 표본 보관실에서 연기가 나고 실제 화재가 확인됐습니다. 내부에 있는 사람은 즉시 대피하세요.",
        "모자이크 작업 구역에서 고정된 벽면이 지금 떨어져 내려오고 있습니다. 주변 사람은 즉시 구역 밖으로 이동하세요.",
        "수조 설비실에서 전기 설비 화재가 확인됐습니다. 사람이 있는 구역이니 즉시 대피 안내를 따라주세요.",
        "리드 비교 연습실의 참가자가 갑자기 의식을 잃어 구급대가 현장에 있습니다. 보호자 확인을 위해 지금 연락 받아주세요.",
        "이미 예약한 관측 시설의 필수 출입 확인이 시작됐습니다. 3분 뒤 입장이 마감되며 미확인 예약은 취소되니 지금 확인해 주세요.",
    ]
    known_paths = sorted(set(protected))
    original_paths = [p for p in known_paths if p.name in ("dataset.jsonl","candidates.jsonl")]
    known = [s for p in original_paths for s in load_score_samples(p)]
    normalize = lambda text:re.sub(r"\s+","",text).casefold()
    known_bodies = {normalize(s.notification.body) for s in known}
    known_windows = {(s.context.active_process,s.context.window_title) for s in known if s.context.active_process}
    new_rows = []
    new_originals = []
    timestamp = "2026-10-04T12:00:00Z"
    for index,recipe in enumerate(recipes):
        family,domain,process,window,reference,planned,waiting = recipe
        reference_rel = [5,4,3,2,1][index%5]
        category = "광고/홍보" if reference_rel==2 else "개인 일반" if reference_rel<=3 else "일반 업무"
        events = [("reference",reference,1 if reference_rel<=3 else 2,reference_rel,category),
                  ("planned",planned,3,5,"일반 업무"),("waiting",waiting,4,5,"긴급 업무")]
        if index < len(critical):
            events.append(("critical",critical[index],5,5,"개인 중요"))
        history = [(process,window),("Notion.exe",f"{domain} 작업 대상과 진행 내용 - Notion"),
                   ("explorer.exe",f"{domain} 작업 자료 - 파일 탐색기")]
        other = recipes[(index+11)%len(recipes)]
        switched = [(other[2],other[3]),("Notion.exe",f"{other[1]} 작업 대상과 진행 내용 - Notion"),
                    ("explorer.exe",f"{other[1]} 작업 자료 - 파일 탐색기")]
        for event,body,urgency,relevance,category in events:
            original_id = f"valpool05_{index+1:02d}_{event}"
            if normalize(body) in known_bodies or normalize(body) in {normalize(s.notification.body) for s in new_rows}:
                raise ValueError("new validation body duplicates an existing original")
            families.setdefault(family,[]).append(original_id)
            new_originals.append(original_id)
            for role,proc,win,entries,rel in (
                ("matching",process,window,history,relevance),
                ("switched",process,window,switched,1),
                ("empty_recent","","",history,relevance),
                ("empty_unknown","","",[],1)):
                if proc and (proc,win) in known_windows:
                    raise ValueError("new validation window duplicates an existing populated window")
                new_rows.append(HistorySample.model_validate({
                    "notification":{"id":original_id+"_"+role,"app_name":["Slack","Microsoft Teams","KakaoTalk.exe","메일","업무 포털"][index%5],"sender":domain+" 담당",
                                    "title":domain+" 알림","body":body,"timestamp":timestamp},
                    "context":{"active_process":proc,"window_title":win,"last_updated":timestamp,
                               "duration_seconds":60 if proc else 0,"recent_processes":[a for a,_ in entries],
                               "recent_windows":[{"app_name":a,"window_title":w} for a,w in entries]},
                    "label":{"urgency_score":urgency,"relevance_score":rel,"category":category,
                             "ai_summary_reason":"검수 전 합성 제안: 긴급도는 사건 상태·시점, 관련도는 현재 창과 최근 작업 대상·행동 기준."}}))
    all_rows = seed_rows+new_rows
    if len(unique_urgency_samples(all_rows)) != 100 or len(new_originals) != 85:
        raise ValueError("expected 100 originals, including 85 new ones")
    ids = [s.notification.id for s in all_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate validation ids")
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl",[s.model_dump(mode="json") for s in all_rows])
    preview_ids = ["valpool05_01_reference_matching","valpool05_01_reference_switched",
                   "valpool05_02_reference_matching","valpool05_02_reference_empty_recent",
                   "valpool05_03_reference_matching","valpool05_04_reference_matching",
                   "valpool05_05_reference_matching","valpool05_06_planned_matching",
                   "valpool05_06_waiting_switched","valpool05_01_critical_empty_unknown"]
    by_id = {s.notification.id:s for s in all_rows}
    lines = ["<!-- 용도: 검증 전용 100원문 후보 풀의 첫 10건 점수 검수.\n생성일: 2026-10-04 (Asia/Seoul)\n상태: 기존 15원문 승인, 신규 85원문 미승인·학습 및 모델 평가 미사용. -->\n",
             "# 검증 후보 100원문: 신규 경계 사례 10건\n",
             "기존 승인 15원문·40행을 보존하고 25개 작업 영역에서 새 원문 85개·340행을 추가했습니다. 전체는 100원문·380행입니다. 신규 자료는 합성 후보이며 이번 10건 검수로 나머지까지 인간 검수했다고 기록하지 않습니다.\n",
             "최근 기록이 다른 작업으로 이어지는 경우 현재 창 제목만으로 높은 관련도를 유지하지 않습니다. 빈 창이어도 최근 제목이 정확한 작업에 연결되면 높은 관련도가 가능합니다. 최근 기록이 없는 빈 창은 목적을 추측하지 않습니다.\n"]
    for i,key in enumerate(preview_ids,1):
        s=by_id[key]
        lines += [f"## {i}. {key}\n",f"알림: {s.notification.body}\n",f"현재 창: {s.context.window_title or '(빈 창)'}\n",
                  "| 최근 앱 | 창 제목 |","| --- | --- |"]
        lines += [f"| {w.app_name} | {w.window_title} |" for w in s.context.recent_windows] or ["| (없음) | (없음) |"]
        lines += ["",f"제안: 긴급도 **{s.label.urgency_score}**, 관련도 **{s.label.relevance_score}**, **{decision_label(s.label.urgency_score,s.label.relevance_score)}**.\n"]
    lines += ["## 분할과 검수 범위\n",
              "원문·모든 문맥·지정 사건 계열은 검증에만 예약합니다. 기존 자료와 정규화 본문·정보 창 완전 일치를 검사했으며 의미상 모든 유사성이 제거됐다는 주장은 하지 않습니다. 향후 학습 확장에서도 이 원문·창·사건 계열을 다시 제외해야 합니다. 최종 테스트는 그대로 보존합니다.\n",
              "이번 생성은 비교군 모델의 추론 결과를 보고 점수를 고르지 않았습니다. 아직 모델 평가용 확정 분할이 아니며 신규 점수는 검수 전 제안입니다. 현재 앱 알림 명칭은 학습 보완 시 실제 앱 다양화가 필요합니다.\n"]
    (output/"review.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"manifest.json",{
        "purpose":"100-original validation-only candidate pool; not a training expansion","created_date":"2026-10-04",
        "intended_split":"validation","status":"candidate_pool_pending_review","policy_version":POLICY_VERSION,
        "originals":100,"rows":len(all_rows),"approved_seed_originals":15,"approved_seed_rows":len(seed_rows),
        "new_synthetic_originals":85,"new_synthetic_rows":len(new_rows),"new_domains":len(recipes),
        "displayed_review_ids":preview_ids,"new_human_reviewed_originals":0,
        "candidate_sha256":digest(output/"candidates.jsonl"),"review_sha256":digest(output/"review.md"),
        "protected_sources":[{"path":str(p),"sha256":digest(p)} for p in known_paths],
        "reserved_families":families,"training_used":False,"model_evaluation_used":False,"test_inference_used":False,
        "reserved_windows":sorted({(s.context.active_process,s.context.window_title) for s in all_rows if s.context.active_process}
                                  | {(w.app_name,w.window_title) for s in all_rows if isinstance(s.context,HistoryContext) for w in s.context.recent_windows}),
        "source_id_groups":{family:sorted(set(v)) for family,v in families.items()},
        "new_urgency_original_counts":dict(Counter(s.label.urgency_score for s in unique_urgency_samples(new_rows))),
        "new_relevance_row_counts":dict(Counter(s.label.relevance_score for s in new_rows)),
        "context_roles":["matching","switched","empty_recent","empty_unknown"],
        "exact_body_overlap":0,"populated_window_overlap":0,
        "files":{"candidates.jsonl":"15 approved originals plus 85 unapproved originals; context histories preserved",
                 "review.md":"ten selected cases, not approval of the full pool"}})
    return all_rows


TRAINING_EXPANSION_DOMAINS = [
    ("웹 품질 검사","Code.exe","VS Code","검색 결과 정렬 검사|검색 검사표|동점 처리;회원 탈퇴 화면 검사|탈퇴 검사표|재확인 단계;첨부 파일 미리보기 검사|첨부 검사표|지원 형식;주소 입력 오류 검사|주소 검사표|오류 안내;장바구니 수량 검사|수량 검사표|상한 처리"),
    ("서비스 관측","EXCEL.EXE","Excel","응답 시간 구간 분류|응답 시간 기록표|느린 구간;오류 로그 유형 정리|오류 유형표|재발 구간;예약 작업 실행 기록 검토|작업 실행표|누락 시각;업로드 실패 원인 분류|실패 원인표|파일 종류;알림 발송 지연 점검|발송 기록표|대기 구간"),
    ("데이터 정리","EXCEL.EXE","Excel","고객 중복 행 정리|중복 고객표|중복 기준;주소 표기 통일|주소 대조표|띄어쓰기;상품 코드 매핑|코드 매핑표|누락 코드;설문 응답 빈칸 분류|응답 빈칸표|미응답 구분;재고 단위 통일|재고 단위표|단위 변환"),
    ("사용성 조사","WINWORD.EXE","Word","가입 과정 인터뷰 정리|가입 인터뷰표|막힌 단계;상품 찾기 행동 분류|검색 행동표|이동 경로;설정 메뉴 이해도 정리|설정 조사표|혼동 항목;문의 작성 과정 분석|문의 과정표|입력 부담;결제 안내 이해도 조사|결제 조사표|안내 표현"),
    ("화면 디자인","Figma.exe","Figma","예약 달력 표시 정리|달력 시안|휴무 표시;문의 내역 카드 구성|문의 카드 시안|상태 표시;배송 단계 화면 구성|배송 시안|진행 단계;사진 업로드 버튼 배치|업로드 시안|버튼 간격;검색 필터 화면 검토|필터 시안|선택 해제"),
    ("고객 상담","EXCEL.EXE","Excel","수령 주소 변경 문의 분류|주소 문의표|변경 가능 상태;환불 진행 문의 정리|환불 문의표|처리 단계;멤버십 해지 문의 분석|해지 문의표|남은 기간;사용 설명 문의 분류|설명 문의표|제품 종류;교환 신청 사유 정리|교환 문의표|상품 상태"),
    ("매장 운영","EXCEL.EXE","Excel","생활용품 진열 위치 점검|진열 위치표|판매 구역;재입고 안내 문구 정리|재입고 안내표|안내 대상;매장 휴무 공지 작성|휴무 공지|운영 날짜;반품 보관 구역 배치|반품 구역표|확인 상태;매장 인수인계 목록 정리|매장 인계표|미처리 항목"),
    ("숙박 운영","EXCEL.EXE","Excel","객실 청소 배정 정리|청소 배정표|담당 객실;분실물 접수 분류|분실물 접수표|발견 장소;체크인 문의 정리|체크인 문의표|도착 시간;객실 비품 수량 확인|비품 수량표|보충 항목;장기 투숙 요청 정리|투숙 요청표|연장 날짜"),
    ("행정 회계","EXCEL.EXE","Excel","출장 증빙 누락 점검|출장 증빙표|첨부 구분;회의비 지출 내역 대조|회의비 지출표|참석 인원;교육비 정산 자료 정리|교육비 정산표|납부 내역;우편 발송 비용 분류|우편 비용표|발송 종류;공용 물품 영수증 정리|물품 영수증표|구매 구분"),
    ("인사 운영","EXCEL.EXE","Excel","신규 직원 장비 배정|장비 배정표|인수 날짜;직무 교육 신청 분류|교육 신청표|희망 과정;휴가 인계 사항 정리|휴가 인계표|대체 담당;사무실 좌석 이동 정리|좌석 이동표|이동 순서;출입증 재발급 문의 정리|출입증 문의표|신청 사유"),
    ("교육 운영","WINWORD.EXE","Word","온라인 수업 질문 정리|수업 질문표|질문 단원;과제 제출 안내 작성|과제 안내문|첨부 형식;실습 장비 사용 순서 정리|실습 순서표|사용 시간;학습 자료 배포 목록 정리|배포 목록표|받을 그룹;보충 수업 희망 시간 정리|보충 신청표|희망 시간"),
    ("도서관 업무","EXCEL.EXE","Excel","도서 반납함 수거 일정|반납함 수거표|수거 위치;책 검색 키워드 정리|도서 검색표|유사 제목;좌석 예약 문의 분류|좌석 문의표|예약 상태;분실 도서 접수 정리|분실 접수표|대출 번호;독서실 이용 안내 교정|이용 안내문|운영 시간"),
    ("연구 지원","WINWORD.EXE","Word","참여자 안내문 표현 검토|참여자 안내문|연락 방법;연구 자료 반출 목록 정리|반출 목록표|승인 항목;실험실 방문 신청 정리|방문 신청표|담당 연락처;관측 기록 양식 교정|관측 기록지|기록 단위;설문 배포 경로 정리|설문 배포표|응답 경로"),
    ("행사 접수","EXCEL.EXE","Excel","참가자 식단 요청 분류|식단 요청표|요청 구분;현장 안내 데스크 배정|데스크 배정표|담당 시간;참가 명찰 출력 목록|명찰 출력표|이름 표기;대기자 연락 순서 정리|대기자 연락표|신청 순서;행사 설문 회수 일정|설문 회수표|회수 위치"),
    ("물품 대여","EXCEL.EXE","Excel","대여 의자 회수 목록|의자 회수표|대여 장소;휴대용 스캐너 대여 정리|스캐너 대여표|반납 상태;행사용 조명 인수 기록|조명 인수표|인수 수량;회의용 태블릿 반납 점검|태블릿 반납표|부속품;보조 배터리 대여 수량|배터리 대여표|남은 수량"),
    ("대중교통 안내","chrome.exe","Chrome","정류장 이전 공지 정리|정류장 공지|대체 위치;분실 카드 문의 분류|카드 문의표|접수 경로;노선 환승 안내 교정|환승 안내문|환승 구간;운행 변경 공지 검토|운행 공지|적용 날짜;이동 지원 예약 문의 정리|지원 문의표|탑승 장소"),
    ("주차 운영","EXCEL.EXE","Excel","방문 차량 등록 안내|차량 등록표|방문 구분;정기 주차 갱신 문의|갱신 문의표|갱신 기간;주차 구역 표지 점검|구역 표지표|안내 방향;임시 주차 신청 정리|임시 신청표|사용 날짜;차량 출입 기록 대조|출입 기록표|확인 시각"),
    ("주거 관리","Notion.exe","Notion","이사 차량 출입 일정|이사 일정표|사용 구역;택배 보관 위치 안내|보관 위치표|수령 절차;공용 창고 사용 목록|창고 사용표|보관 기간;방문 수리 접수 정리|수리 접수표|방문 시간;공동 현관 안내문 교정|현관 안내문|출입 방법"),
    ("돌봄 일정","EXCEL.EXE","Excel","방문 돌봄 담당 배정|돌봄 배정표|방문 구역;보호자 연락 시간 정리|연락 시간표|통화 가능 시간;복지 차량 동행 일정|동행 일정표|집합 장소;생활 지원 신청 분류|지원 신청표|필요 항목;방문 준비물 목록 점검|방문 준비표|누락 물품"),
    ("반려동물 용품","chrome.exe","Chrome","급수기 세척 주기 비교|급수기 비교표|세척 방법;운반 가방 크기 비교|운반 가방표|내부 크기;털 관리 도구 비교|관리 도구표|사용 부위;방석 소재 조건 비교|방석 비교표|세탁 조건;장난감 보관 목록 정리|장난감 보관표|파손 상태"),
    ("스포츠 운영","EXCEL.EXE","Excel","배드민턴 코트 배정|코트 배정표|경기 순서;농구 연습 참가 정리|연습 참가표|인원 구분;탁구 대회 진행 목록|탁구 진행표|호출 순서;풋살 팀 교대 시간|교대 시간표|참가 팀;수영장 단체 이용 문의|이용 문의표|입장 시간"),
    ("정원 관리","Notion.exe","Notion","화단 관수 담당 정리|관수 담당표|담당 구역;퇴비 보관 목록 점검|퇴비 보관표|보관 상태;화분 이동 위치 검토|화분 위치표|받침 위치;정원 도구 반납 기록|도구 반납표|미반납 도구;낙엽 수거 일정 정리|낙엽 수거표|수거 구역"),
    ("목공 제작","FreeCAD.exe","FreeCAD","서랍 손잡이 배치|손잡이 배치도|부착 간격;수납함 덮개 치수 검토|덮개 치수표|여닫는 방향;책상 다리 연결 검토|다리 연결도|고정 위치;목재 자투리 분류|자투리 목록표|보관 구역;벽걸이 선반 지지 검토|선반 지지도|지지 간격"),
    ("금속 가공","FreeCAD.exe","FreeCAD","연결판 구멍 위치 검토|연결판 위치도|구멍 간격;벤딩 순서 표기 정리|벤딩 순서표|접는 방향;금속 표면 처리 기록|표면 처리표|처리 구분;절단 잔재 보관 분류|잔재 보관표|보관 길이;체결 와셔 두께 비교|와셔 비교표|부품 두께"),
    ("제품 검사","EXCEL.EXE","Excel","완제품 외관 흠집 분류|외관 검사표|흠집 위치;포장 봉인 상태 점검|봉인 검사표|봉인 구분;부속품 누락 기록 대조|누락 기록표|부품 종류;제품 표시 문구 교정|표시 검토표|주의 문구;검사 사진 이름 정리|검사 사진표|촬영 대상"),
    ("물류 상담","EXCEL.EXE","Excel","수령인 연락처 문의|연락처 문의표|확인 경로;배송 보류 사유 분류|배송 보류표|보류 사유;묶음 배송 요청 정리|묶음 요청표|합칠 주문;반송 주소 확인 기록|반송 주소표|주소 표기;배송 사진 문의 정리|사진 문의표|촬영 위치"),
    ("사진 정리","Lightroom.exe","Lightroom","가족 사진 중복 분류|사진 중복표|촬영 날짜;상품 사진 배경 비교|배경 비교표|배경 밝기;프로필 사진 후보 정리|프로필 후보표|사용 용도;여행 사진 위치 표기|사진 위치표|촬영 구역;앨범 표지 사진 선택|표지 후보표|사진 비율"),
    ("음성 자료","audacity.exe","Audacity","안내 멘트 발음 대조|발음 대조표|발음 구간;인터뷰 음성 이름 정리|음성 이름표|참여자 구분;학습 녹음 반복 구간|반복 구간표|반복 순서;효과음 사용 목록 정리|효과음 목록표|사용 위치;회의 녹음 발언 구분|발언 구분표|발언 순서"),
    ("개인 일정","Notion.exe","Notion","주민센터 방문 준비|민원 방문 준비표|접수 항목;치과 예약 시간 정리|예약 시간표|방문 날짜;자동차 검사 준비 목록|검사 준비표|준비 서류;가족 기념일 장소 비교|장소 비교표|이동 거리;우편물 수령 일정 정리|수령 일정표|받을 장소"),
    ("개인 취미","Notion.exe","Notion","퍼즐 조각 분류 기록|퍼즐 분류표|무늬 구분;수채화 색상 연습 정리|색상 연습표|색상 조합;종이접기 단계 기록|접기 단계표|접는 순서;만년필 잉크 기록 비교|잉크 기록표|번짐 구간;미니어처 가구 배치|가구 배치도|연결 위치"),
]

TRAINING_URGENT_SCENARIOS = [
    ("elevator_entrapment","승강기 안에 사람이 갇혔고 비상 통화가 연결됐습니다. 현장 담당자는 지금 구조 담당 연락을 받아주세요."),
    ("electrical_arc","공용 분전함에서 실제 불꽃과 감전 위험이 확인됐습니다. 주변 사람은 즉시 위험 구역에서 벗어나 안전 안내를 따라주세요."),
    ("chemical_spill","작업 구역에서 유해 물질 누출이 확인됐습니다. 현장에 있는 사람은 즉시 접근을 멈추고 안전 담당자의 이동 안내를 확인하세요."),
    ("flooded_exit","지하 통로로 물이 빠르게 들어오며 출구 접근이 어려워지고 있습니다. 내부 사람은 지금 현장 대피 안내를 확인하세요."),
    ("vehicle_rollaway","주차 구역의 차량이 사람 쪽으로 움직이고 있습니다. 주변 사람은 즉시 안전한 곳으로 이동하고 현장 안내를 확인하세요."),
    ("construction_debris","통행 중인 보행로 위로 공사 자재가 떨어지고 있습니다. 현장 접근을 즉시 멈추고 통제 안내를 확인하세요."),
    ("suspended_load","작업장에 매달린 자재가 이탈해 사람이 있는 구역으로 흔들리고 있습니다. 즉시 위험 구역 밖으로 이동하세요."),
    ("exposed_wire","현재 이용 중인 통로에서 전선 피복 손상과 감전 위험이 확인됐습니다. 즉시 접근을 멈추고 우회 안내를 확인하세요."),
    ("escaped_animal","공원 출입로에서 사람을 공격할 위험이 있는 동물이 접근하고 있습니다. 현장 방문자는 지금 안전 통제 안내를 확인하세요."),
    ("blocked_shelter","폭우 대피 장소의 출입 통제가 바뀌었습니다. 지금 이동 중인 주민은 새 대피 위치 안내를 즉시 확인하세요."),
    ("dam_release","현재 머무는 하천 구역에 방류수가 도달하고 있습니다. 현장 이용자는 즉시 구역 밖으로 이동하라는 안내를 확인하세요."),
    ("broken_bridge","현재 사람이 건너는 임시 통로의 연결부 파손이 확인됐습니다. 즉시 이동을 멈추고 현장 구조 안내를 확인하세요."),
    ("unstable_slope","현재 작업 구역으로 토사가 흘러내리고 있습니다. 작업자는 즉시 위험 구역에서 벗어나 통제 안내를 확인하세요."),
    ("hearing_protection_failure","설비 파손으로 현재 사람 주변에 고온 증기가 분출되고 있습니다. 작업을 멈추고 즉시 안전 담당 안내를 확인하세요."),
    ("missing_child","보호 중인 어린이가 혼자 차도로 향하는 것을 현장 직원이 확인했습니다. 보호자는 지금 현장 연락을 받아주세요."),
    ("dangerous_door","자동 출입문에 사람의 옷이 끼어 문이 계속 움직이고 있습니다. 현장 담당자는 즉시 안전 조치 연락을 받아주세요."),
    ("welfare_contact","지원 대상자가 문을 열지 못한 채 구조 요청을 보내고 있습니다. 방문 담당자는 지금 현장 연락을 확인하세요."),
    ("rail_intrusion","현재 이용 중인 승강장 가까운 선로에 사람이 들어간 것을 확인했습니다. 이용자는 즉시 현장 안전 통제 안내를 확인하세요."),
    ("explosive_battery","휴게 공간의 배터리에서 파열음과 분출이 발생하고 있습니다. 주변 사람은 즉시 접근을 멈추고 안전 안내를 확인하세요."),
    ("emergency_transfer_contact","응급 이송 차량이 현장에 도착했고 환자 확인을 위해 보호자 통화가 필요합니다. 지금 연락을 받아주세요."),
    ("unauthorized_card_spend","본인이 사용하지 않은 카드 결제가 현재 연속 승인되고 있습니다. 추가 피해를 막기 위한 카드 정지 안내를 지금 확인하세요."),
    ("account_change_spend","본인 요청이 아닌 결제 수단 변경 후 구매가 계속 승인되고 있습니다. 지금 계정 보호 안내를 확인하세요."),
    ("incorrect_payout","지급 담당자가 다른 수취인으로 송금을 실행하려고 합니다. 실행 직전 최종 확인 중이니 지금 중지 연락을 받아주세요."),
    ("duplicate_payment","동일 청구의 중복 결제가 계속 처리되고 있습니다. 추가 결제를 막기 위해 지금 결제 담당 연락을 확인하세요."),
    ("fraud_withdrawal","본인이 요청하지 않은 출금이 연속 처리되고 있습니다. 추가 출금을 막는 보호 절차 안내를 지금 확인하세요."),
    ("deposit_deadline","본인이 신청한 임대 계약의 필수 입금 확인이 4분 뒤 마감됩니다. 미확인 시 계약이 취소되므로 지금 확인해주세요."),
    ("benefit_document_deadline","이미 신청한 생활 지원의 필수 증빙이 빠졌습니다. 보완 마감이 3분 남았고 이후 이번 신청이 거절되니 지금 제출해주세요."),
    ("medical_visit_admission","예약한 필수 치료 방문의 접수 확인이 시작됐고 3분 뒤 마감됩니다. 미확인 시 이번 예약이 취소되니 지금 접수 안내를 확인하세요."),
    ("exam_identity_deadline","이미 신청한 자격 심사의 본인 확인이 2분 뒤 마감됩니다. 미확인 시 이번 응시가 취소되니 지금 확인해주세요."),
    ("housing_application_deadline","신청한 주거 지원의 필수 동의가 누락됐습니다. 5분 뒤 접수가 마감돼 이번 신청이 무효가 되니 지금 보완해주세요."),
    ("permit_supplement_deadline","이미 접수한 영업 허가의 필수 보완이 4분 뒤 마감됩니다. 마감 뒤 이번 신청이 취소되니 지금 보완 서류를 제출해주세요."),
    ("court_document_deadline","본인이 진행 중인 접수 건의 필수 서류 제출이 3분 뒤 마감됩니다. 미제출 시 이번 접수가 거절되니 지금 제출 안내를 확인하세요."),
    ("mandatory_course_identity","예약한 필수 안전 과정의 본인 확인이 2분 뒤 마감됩니다. 미확인 시 수료가 인정되지 않으니 지금 접속해주세요."),
    ("medical_access_contact","현재 진행 중인 응급 처치를 위해 보호자의 필수 확인이 필요합니다. 의료 담당자가 연락 중이니 지금 전화를 받아주세요."),
    ("bus_departure_reserved","이미 예약한 단체 이동 차량의 탑승 확인이 3분 뒤 마감됩니다. 미확인 시 예약 좌석이 취소되니 지금 확인해주세요."),
    ("lost_person_traffic","돌봄 대상자가 혼자 차량 통행 구역으로 나간 것을 현장 직원이 확인했습니다. 담당 보호자는 즉시 현장 연락을 받아주세요."),
    ("delivery_person_trapped","배송 담당자가 잠긴 시설 안에서 나가지 못하고 구조 연락을 보냈습니다. 시설 담당자는 지금 연락을 확인해주세요."),
    ("failing_handhold","현재 이용 중인 계단 난간이 빠져 통행자가 추락할 위험이 확인됐습니다. 즉시 통행을 멈추고 안전 안내를 확인하세요."),
    ("power_failure_medical_device","돌봄 현장에서 사용 중인 필수 의료 장비가 전원 장애로 작동을 멈췄습니다. 현장 대응 담당자는 지금 연락을 확인하세요."),
    ("transport_accessibility_emergency","이동 지원 차량 안에서 탑승자의 안전 장치가 풀렸고 차량이 정차했습니다. 담당자는 지금 현장 안전 확인 연락을 받아주세요."),
    ("rescue_location_confirmation","구조대가 요청 장소에 도착했지만 출입 위치 확인이 필요합니다. 현장 신고자는 지금 구조대 연락을 받아주세요."),
]

TRAINING_URGENT_GOALS = [
    "승강기 고립 구조 연락 확인","분전함 감전 위험 통제 확인","유해 물질 누출 접근 통제 확인",
    "지하 통로 침수 대피 경로 확인","주차장 차량 이동 안전 안내 확인","보행로 낙하 자재 통제 확인",
    "매달린 작업 자재 이탈 통제 확인","노출 전선 통로 우회 안내 확인","위험 동물 접근 통제 확인",
    "폭우 대피 장소 변경 확인","하천 방류 구역 대피 확인","임시 통로 파손 구조 안내 확인",
    "토사 유입 작업 구역 통제 확인","고온 증기 분출 안전 안내 확인","어린이 차도 접근 보호 연락 확인",
    "자동문 끼임 안전 조치 연락 확인","돌봄 대상자 구조 연락 확인","선로 접근 안전 통제 확인",
    "배터리 파열 위험 접근 통제 확인","응급 이송 보호자 연락 확인","부정 카드 결제 정지 안내 확인",
    "무단 결제 수단 변경 보호 안내 확인","잘못된 수취인 송금 중지 확인","중복 결제 중단 연락 확인",
    "무단 출금 차단 안내 확인","임대 계약 입금 확인","생활 지원 필수 증빙 보완",
    "필수 치료 방문 접수 확인","자격 심사 본인 확인","주거 지원 필수 동의 제출",
    "영업 허가 보완 서류 제출","진행 중인 접수 필수 서류 제출","필수 안전 과정 본인 확인",
    "응급 처치 보호자 확인 연락","예약 차량 탑승 확인","돌봄 대상자 차도 접근 연락 확인",
    "시설 고립 배송 담당 구조 확인","계단 난간 파손 통행 통제 확인","의료 장비 전원 장애 대응 연락",
    "이동 지원 차량 안전 장치 연락 확인","구조대 출입 위치 연락 확인",
]


# Eight complete phrasings per event preserve its deadline and action semantics.
# These diversify wording, not the number of independent event families.
TRAINING_EVENT_PHRASINGS = {
    "joke_focus": [
        "{goal} 하다가 머릿속도 같이 정렬되는 줄ㅋㅋ 그냥 혼잣말임",
        "{item} 보고 있으니 눈이 먼저 퇴근하겠네ㅎㅎ 답장 안 해도 돼",
        "오늘 {goal} 하면 집중력 만렙 찍는 거 아님?ㅋㅋ",
        "{field}만 보다가 꿈에서도 이 단어 나오겠어ㅋㅋ",
        "{item}보다 내 집중력부터 정리해야 할 듯ㅎㅎ",
        "{goal} 얘기만 들으면 커피 생각부터 난다ㅋㅋ",
        "{field} 외우다 내가 설명서 될 기세ㅋㅋ 그냥 농담이야",
        "{item} 보고 있으면 시간이 순간이동함ㅋㅋ 할 일 요청은 아님",
    ],
    "joke_break": [
        "{goal} 얘기하다가 저녁 메뉴로 샜네ㅎㅎ 일 얘기는 아니야",
        "{item}보다 오늘 간식 투표가 더 치열함ㅋㅋ",
        "{goal} 끝나면 뭐 먹을까? 그냥 수다야ㅎㅎ",
        "{field} 말고 점심 메뉴도 누가 정리해 줬으면ㅋㅋ",
        "{goal} 주제로 모였는데 고양이 사진만 보고 있음ㅎㅎ",
        "{item} 얘기는 잠깐, 주말에 본 영화가 더 기억남ㅋㅋ",
        "{goal} 한다고 모여서 날씨 얘기만 한 듯ㅎㅎ",
        "{field} 이름으로 밴드 만들면 웃기겠다ㅋㅋ 아무 요청 없음",
    ],
    "archive": [
        "완료된 {item} 사본은 보관 폴더에 있습니다. 이번 작업 자료는 아니며 확인 요청도 없습니다.",
        "지난 회차 {item} 보관을 마쳤습니다. 현재 사용본은 그대로이고 추가 처리는 없습니다.",
        "{item}의 이전 결과를 기록용으로 옮겨 뒀어요. 지금 열어 볼 필요는 없어요.",
        "옛 {item} 자료를 보관함에서 찾을 수 있습니다. 새 수정 사항은 없습니다.",
        "보관 완료 안내: 지난번 {item}입니다. 회신이나 확인은 필요하지 않습니다.",
        "지난 작업 때 쓴 {item}는 기록 폴더에 저장됐습니다. 현행 문서에는 영향이 없습니다.",
        "이전 {item} 정리가 끝나 보관 위치만 알려드립니다. 처리할 사항은 없습니다.",
        "{item} 과거판을 기록실에 넣었습니다. 필요해질 때 찾아보시면 됩니다.",
    ],
    "color_option": [
        "{item}의 표시 색을 선택할 수 있게 됐어요. 내용은 같고 기존 색을 계속 써도 됩니다.",
        "선택 설정 안내: {item}에 색상 팔레트가 추가됐습니다. 바꿀 필요는 없습니다.",
        "{item} 화면에 다른 색을 입힐 수 있어요. 항목이나 기능이 바뀌는 것은 아닙니다.",
        "{item}에 표시할 색 선택지가 늘었습니다. 적용 여부는 자유입니다.",
        "새 표시 색상이 준비됐습니다. {item}의 기존 설정에 문제는 없으니 원하는 경우만 바꾸세요.",
        "{item} 색상 변경은 이제 선택할 수 있습니다. 작업 내용에는 영향이 없어요.",
        "취향에 따라 {item}의 표시 색을 바꿔 보세요. 기본 색도 계속 지원합니다.",
        "{item} 꾸미기 메뉴에 색상 옵션을 더했습니다. 설정 변경 요청은 아닙니다.",
    ],
    "community_news": [
        "{goal} 모임 지난 활동 사진입니다. 업무 자료 없이 근황만 올려요. 보고 싶으면 봐요.",
        "{goal} 커뮤니티의 지난 모임 후기를 올렸어요. 그냥 소식 공유이고 할 일은 없습니다.",
        "지난달 {goal} 모임 단체 사진을 공개했습니다. 작업에 반영할 내용은 없습니다.",
        "{goal} 모임 회원들 근황을 게시판에 적었습니다. 확인이나 답장 요청은 아니에요.",
        "{goal} 커뮤니티 소식지에 지난 행사 사진이 실렸습니다. 여유 있을 때 구경하세요.",
        "지난번 {goal} 모임 분위기가 궁금하면 후기 사진을 보세요. 참고 절차 자료는 아닙니다.",
        "{goal} 모임에서 찍은 추억 사진만 공유해요. 관련 작업 지시는 없습니다.",
        "{goal} 커뮤니티 게시판에 근황 글이 올라왔어요. 읽을지는 자유입니다.",
    ],
    "reference": [
        "{field} 정리에 쓸 절차를 참고 문서에 적었습니다. {goal} 진행할 때 편한 시간에 보세요.",
        "{goal}에 도움이 될 {field} 처리 순서를 공유합니다. 바로 읽을 필요는 없습니다.",
        "참고용으로 {field} 정리 과정을 정리했어요. {item} 작업 중 필요할 때 확인하세요.",
        "{item} 작성 시 {field}를 다루는 절차 안내입니다. 검토 기한은 없습니다.",
        "{goal}용 참고 절차를 올렸습니다. {field} 정리할 때 활용하고 확인은 나중에 하셔도 됩니다.",
        "{field} 처리 절차가 참고 폴더에 있어요. {goal} 이어서 할 때 읽어 주세요.",
        "{item} 작업에 사용할 {field} 정리 가이드입니다. 지금 회신할 내용은 없습니다.",
        "{goal} 참고 자료에 {field} 정리 순서를 추가했습니다. 여유 있을 때 검토해 주세요.",
    ],
    "bookmark": [
        "{goal}에 관한 공개 자료 주소를 저장해 뒀어요. 필요한 경우 찾아보세요. 기한은 없습니다.",
        "{goal} 참고 페이지를 북마크에 넣었습니다. 확인 요청이나 마감은 없습니다.",
        "자료 목록에 {goal} 공개 참고 링크를 추가했습니다. 나중에 이용하시면 됩니다.",
        "{goal} 관련 공개 페이지 주소가 참고함에 있습니다. 지금 열어 볼 필요는 없어요.",
        "북마크 저장 알림: {goal} 참고 페이지입니다. 보실 시점은 자유입니다.",
        "{goal} 참고 링크를 목록에서 찾을 수 있게 했습니다. 회신은 필요하지 않습니다.",
        "필요할 때 이용하도록 {goal} 자료 주소만 남겨 둡니다. 확인 기한은 따로 없습니다.",
        "{goal}의 공개 참고 페이지를 링크함에 보관했습니다. 급하게 볼 내용은 아닙니다.",
    ],
    "theme": [
        "{item} 화면에 밝은 테마를 선택할 수 있습니다. 기존 테마로 계속 작업해도 됩니다.",
        "보기 설정에 새 테마가 있습니다. {item} 내용과 기능은 같으며 사용 여부는 자유입니다.",
        "{item}의 배경 테마 선택지가 늘었어요. 기본 표시 방식도 유지됩니다.",
        "원하시면 {item}를 새 테마로 볼 수 있어요. 바꿔야 하는 설정은 아닙니다.",
        "{item} 표시 테마를 추가했습니다. 현재 화면에도 문제가 없으니 원하는 경우만 적용하세요.",
        "선택형 테마 안내입니다. {item}의 표시만 바뀌며 기존 방식도 계속 쓸 수 있습니다.",
        "{item} 화면 꾸미기 메뉴에 테마가 추가됐습니다. 내용 수정이나 확인 요청은 없습니다.",
        "새 화면 테마가 제공됩니다. {item}를 이전 테마로 보는 데에는 영향이 없습니다.",
    ],
    "annotation": [
        "{item}의 {field} 주석이 짧아 이해하기 어렵습니다. 다음 정리 때 설명을 덧붙여 주세요.",
        "다음에 {item}를 손볼 때 {field} 주석을 쉽게 풀어 적어 보면 좋겠어요. 급하지 않습니다.",
        "{field} 옆 주석에 설명을 더 넣자는 제안입니다. {item}의 다음 정리 때 검토해 주세요.",
        "{item} 주석 중 {field} 설명을 보완하면 좋겠습니다. 바로 수정할 필요는 없습니다.",
        "{field} 주석을 읽기 쉽게 바꾸자는 의견을 남겼어요. {item} 정리할 때 반영 여부를 보세요.",
        "{item}에서 {field} 주석을 더 구체적으로 적어 주세요. 다음 작업 때 해도 됩니다.",
        "다음 편집 의견: {item}의 {field} 주석에 용어 설명을 보태면 좋겠습니다. 기한은 없습니다.",
        "{field} 주석 보완을 제안합니다. {item}를 다음에 정리할 때 검토하고 지금 회신은 없어도 됩니다.",
    ],
    "formats": [
        "{item}를 공유할 두 파일 형식의 차이를 비교했습니다. 공유 준비에 참고하세요. 기한은 없습니다.",
        "나중에 {item}를 보낼 때 쓸 형식 비교표입니다. 바로 결정하지 않아도 됩니다.",
        "{item} 공유용 파일 형식 두 가지의 장단점을 적었습니다. 필요한 때 확인하세요.",
        "공유 준비 자료에 {item} 파일 형식별 차이를 추가했습니다. 확인 시점은 자유입니다.",
        "{item}를 어떤 형식으로 전달할지 참고할 설명입니다. 당장 답할 필요는 없습니다.",
        "두 형식으로 {item}를 공유하는 방법을 비교해 뒀어요. 다음 공유 준비 때 읽어 주세요.",
        "{item} 전달용 파일 형식 안내를 올렸습니다. 나중에 선택할 때 활용하세요.",
        "{item} 공유에 쓸 형식 비교 자료가 준비됐습니다. 현재 회신이나 마감은 없습니다.",
    ],
    "faq": [
        "{goal}에서 반복되는 질문과 답변을 모았습니다. 필요한 때 답변 작성에 활용하세요.",
        "{goal} 관련 자주 묻는 질문 자료입니다. 확인 기한 없이 참고용으로 공유합니다.",
        "질문 자료함에 {goal} 답변 예시를 올렸어요. 바로 검토할 필요는 없습니다.",
        "{goal} 중 받을 수 있는 질문을 정리했습니다. 나중에 안내문 작성할 때 참고하세요.",
        "{goal} 자주 묻는 질문 목록을 공유합니다. 지금 회신하지 않아도 됩니다.",
        "필요할 때 쓰도록 {goal} 질문과 답변을 정리해 뒀어요. 확인은 편한 시간에 하세요.",
        "{goal} 참고 자료에 질문별 답변 안내를 추가했습니다. 급한 요청은 아닙니다.",
        "{goal} 질문 대응에 쓸 목록이 준비됐습니다. 다음에 답변할 때 활용하시면 됩니다.",
    ],
    "draft": [
        "{item}의 {field} 초안입니다. 다음 정리 전 여유 있을 때 의견을 부탁드립니다.",
        "{field} 초안을 {item}에 넣어 공유했어요. 급하지 않으니 편한 때 검토해 주세요.",
        "검토할 초안이 있습니다. {item}의 {field}를 다음 정리 전에 보고 의견 남겨 주세요.",
        "{item} 초안 중 {field} 검토를 부탁드려요. 당장 회신하지 않아도 됩니다.",
        "{field} 작성 초안을 올렸습니다. {item}를 다시 정리하기 전 천천히 의견 주세요.",
        "다음 정리용 {item} 초안을 공유합니다. {field} 의견은 여유 있을 때 받겠습니다.",
        "{item}에 들어갈 {field} 초안을 검토해 주세요. 지금 확인할 필요는 없습니다.",
        "{field} 초안 의견 요청입니다. {item} 다음 정리 전까지 편한 시간에 보세요.",
    ],
    "afternoon": [
        "오늘 오후 {item} 정리 예정입니다. 점심 이후 {field} 확인 결과를 보내 주세요.",
        "{field} 검토는 오늘 점심 뒤에 해 주세요. 오후에 {item}를 정리할 때 회신을 사용합니다.",
        "{item}를 오후에 정리하니 점심 이후 {field} 의견을 부탁드립니다. 지금 답할 필요는 없습니다.",
        "오후 정리 전 {item}의 {field}를 확인해 주세요. 답변은 점심 이후 보내시면 됩니다.",
        "오늘 {item} 정리를 오후에 진행합니다. {field} 회신은 점심 뒤에 부탁드려요.",
        "점심 이후 {field} 확인을 부탁드립니다. 오늘 오후 {item} 정리에 필요한 답변입니다.",
        "{item} 오후 정리를 준비 중입니다. 지금 말고 점심 이후 {field} 내용을 회신해 주세요.",
        "오늘 오후에 쓸 {item}의 {field}를 확인해 주세요. 점심 후에 답을 주시면 됩니다.",
    ],
    "tomorrow": [
        "{item}에 {field} 표기가 없습니다. 내일 오전 제출 전에 확인해 알려주세요.",
        "내일 오전 제출할 {item}에서 {field} 표시가 빠졌습니다. 그 전까지 확인해 주세요.",
        "{field} 표기를 못 찾았습니다. {item} 제출은 내일 오전이니 그때까지 답변 부탁드립니다.",
        "{item} 제출 준비 중 {field} 누락을 발견했습니다. 내일 오전 전에 확인 결과를 주세요.",
        "확인이 필요합니다. {item}의 {field} 표시가 없어 내일 오전 제출 전 보완해야 합니다.",
        "{item}에서 빠진 {field} 표기를 확인해 주세요. 제출 시각인 내일 오전까지 알려주시면 됩니다.",
        "내일 오전까지 {field} 표기를 확인해 회신해 주세요. 제출할 {item}에 해당 표시가 없습니다.",
        "{field} 표기 누락 안내입니다. {item}를 내일 오전에 내기 전 확인이 필요합니다.",
    ],
    "queue": [
        "{item} 확인 요청이 여러 건 쌓였습니다. 오늘 중 {field} 처리 순서를 정해 주세요.",
        "오늘 안에 {field} 처리 순서를 알려주세요. {item} 관련 확인 요청이 대기 중입니다.",
        "{item} 요청 대기열이 늘고 있어요. {field}부터 어떤 순서로 처리할지 오늘 정해주세요.",
        "누적된 {item} 확인 요청을 정리해야 합니다. 오늘 중 {field} 처리 순서 회신을 부탁드립니다.",
        "{field} 처리 순서가 정해지지 않아 {item} 확인 요청이 쌓였습니다. 오늘 안에 정리해 주세요.",
        "{item} 관련 요청들이 대기하고 있습니다. {field} 처리 우선순위는 오늘 중 알려주세요.",
        "오늘 처리 순서 확인 건입니다. {item} 대기 요청의 {field} 순서를 정해 주세요.",
        "{item} 요청이 누적됐습니다. 지금 즉답은 아니지만 오늘 중 {field} 처리 순서를 부탁드립니다.",
    ],
    "sync_delay": [
        "{item} 변경 내용의 동기화가 지연됐습니다. 원본은 안전하며 오늘 중 상태 점검을 부탁드립니다.",
        "오늘 안에 {item} 동기화 상태를 확인해 주세요. 반영이 늦지만 원본 손실은 없습니다.",
        "{item}가 다른 기기에 늦게 반영됩니다. 원본은 남아 있으니 오늘 상태를 살펴 주세요.",
        "동기화 지연 안내입니다. {item} 원본은 보존됐으며 오늘 중 점검하면 됩니다.",
        "{item} 수정분 반영에 시간이 걸리고 있어요. 자료 손실은 없고 오늘 중 확인이 필요합니다.",
        "원본은 정상 보관 중이나 {item} 동기화가 늦습니다. 오늘 안에 원인을 확인해 주세요.",
        "{item} 동기화 상태 점검 요청입니다. 변경분만 늦게 반영되며 원본은 안전합니다. 오늘 중 봐주세요.",
        "오늘 중 {item} 변경 내용이 동기화됐는지 확인해 주세요. 원본은 남아 있어 손실 위험은 없습니다.",
    ],
    "capacity": [
        "{item} 폴더 공간이 줄었습니다. 아직 저장은 가능하니 다음 작업 전에 불필요한 파일을 정리해 주세요.",
        "다음 작업 전 {item} 폴더를 정리해 주세요. 공간은 부족해지고 있지만 지금 저장에는 문제가 없습니다.",
        "{item} 작업 공간 점검 안내입니다. 현재 저장 가능하며 다음 작업 시작 전에 불필요한 파일 정리가 필요합니다.",
        "저장은 계속 가능하지만 {item} 폴더 여유가 줄고 있어요. 다음 작업 전 공간을 확보해 주세요.",
        "{item} 폴더의 남은 공간을 확인했습니다. 저장이 막힌 상태는 아니며 다음 작업 전에 정리 부탁드립니다.",
        "다음 작업을 위해 {item} 불필요 파일을 정리해 주세요. 현재 저장 공간은 아직 남아 있습니다.",
        "{item} 공간 확보 요청입니다. 당장 저장 실패는 없지만 다음 작업 전 정리를 해야 합니다.",
        "{item} 작업 폴더가 차고 있습니다. 지금 저장은 가능하니 다음 작업 전 불필요 자료를 정리하세요.",
    ],
    "review_slot": [
        "오늘 오후 {item} 검토가 예정돼 있습니다. 시작 전 {field} 의견을 보내 주세요.",
        "{field} 의견을 오늘 오후 검토 전에 정리해 주세요. 검토 대상은 {item}입니다.",
        "오후에 {item}를 함께 검토합니다. 그 전까지 {field}에 대한 의견 회신을 부탁드립니다.",
        "{item} 검토 시간은 오늘 오후입니다. 시작하기 전에 {field} 의견을 받아야 합니다.",
        "오늘 오후 검토 자료로 {item}를 사용합니다. {field} 의견은 그 전에 보내 주세요.",
        "{field} 검토 의견을 준비해 주세요. 오늘 오후 {item} 검토 전까지 보내시면 됩니다.",
        "{item}에 대한 오후 검토를 준비 중입니다. 시작 전 {field} 의견을 정리해 회신해 주세요.",
        "검토 준비 요청입니다. 오늘 오후 {item}를 다루기 전 {field} 의견을 보내 주시면 됩니다.",
    ],
    "retry": [
        "{item} 공유가 실패했습니다. 원본은 안전하며 오늘 저녁까지 다시 보내면 됩니다.",
        "오늘 저녁까지 {item}를 다시 공유해 주세요. 한 번 전송에 실패했지만 원본에는 문제가 없습니다.",
        "{item} 전송 실패 안내입니다. 원본은 남아 있고 재공유 기한은 오늘 저녁입니다.",
        "공유가 되지 않아 {item}를 다시 보내야 합니다. 원본은 안전하니 오늘 저녁까지 처리해 주세요.",
        "{item} 첫 공유가 실패했어요. 자료 손실은 없고 오늘 저녁까지 재시도하면 됩니다.",
        "오늘 저녁까지 다시 공유 부탁드립니다. {item} 원본은 정상이고 전송만 한 번 실패했습니다.",
        "{item} 공유를 재시도해 주세요. 원본은 안전하게 보관됐고 기한은 오늘 저녁입니다.",
        "원본은 남아 있으나 {item} 공유가 완료되지 않았습니다. 오늘 저녁 전 다시 보내 주세요.",
    ],
    "missing_attachment": [
        "내일 {goal} 점검에 사용할 {item} 참고 첨부가 없습니다. 오늘 안에 보완해 주세요.",
        "{item} 참고 첨부를 오늘 추가해 주세요. 내일 {goal} 점검 자료에 필요합니다.",
        "내일 점검 준비 중 {item}의 참고 첨부 누락을 발견했습니다. {goal} 준비를 위해 오늘 보완하세요.",
        "{goal} 점검은 내일입니다. {item} 참고 첨부가 빠졌으니 오늘 중 채워 주세요.",
        "오늘 보완할 자료가 있습니다. 내일 {goal} 점검에 쓸 {item} 참고 첨부입니다.",
        "{item}에 참고 파일이 없어 내일 {goal} 점검 전 준비가 필요합니다. 첨부는 오늘 중 부탁드립니다.",
        "내일 사용할 {item}의 참고 첨부를 확인해 주세요. 빠져 있어 오늘 중 추가해야 합니다.",
        "{goal} 내일 점검 자료 보완 요청입니다. {item} 참고 첨부 누락을 오늘 해결해 주세요.",
    ],
    "callback": [
        "{item}의 {field} 문의가 들어왔습니다. 상대방이 오늘 오후 답변을 기다리니 확인 부탁드립니다.",
        "오늘 오후까지 {field} 문의 답변을 주세요. 상대방이 {item}에 관한 확인을 기다립니다.",
        "{field} 내용 확인을 부탁드립니다. {item} 문의 상대에게 오늘 오후에 답해야 합니다.",
        "{item} 문의 회신이 필요합니다. 오늘 오후까지 {field} 내용을 확인해 알려주세요.",
        "상대방이 {item}의 {field} 답변을 기다리고 있습니다. 오늘 오후 회신할 수 있게 확인해 주세요.",
        "오늘 오후 답변할 문의입니다. {item}의 {field} 확인 결과를 보내 주세요.",
        "{item} 관련 문의가 있어 전달합니다. {field} 답변을 오늘 오후까지 부탁드립니다.",
        "{field}를 물어본 분에게 오늘 오후에 답변해야 합니다. {item} 내용을 확인해 주세요.",
    ],
    "disagreement": [
        "{item}의 {field} 의견이 갈립니다. 내일 작업 시작 전 기준을 정리해 주세요.",
        "내일 시작 전에 {field} 기준을 합의해 주세요. {item}에 서로 다른 의견이 남았습니다.",
        "{field} 판단 기준을 정리해야 합니다. {item} 의견이 달라 내일 작업 시작 전 확인 부탁드립니다.",
        "{item}에서 {field}에 대한 의견 차이가 발견됐습니다. 내일 작업하기 전에 기준을 맞춰 주세요.",
        "내일 작업용 {item}에 의견이 나뉘었습니다. 시작 전 {field} 기준을 정리하면 됩니다.",
        "{field} 의견 차이를 내일 시작 전까지 정리해 주세요. {item}에 두 기준이 함께 적혀 있습니다.",
        "{item} 의견 조율 요청입니다. {field}에 서로 다른 의견이 있어 내일 작업 전 기준이 필요합니다.",
        "내일 작업을 시작할 때 혼동 없도록 {item}의 {field} 의견 차이를 그 전에 정리해 주세요.",
    ],
    "live_access": [
        "{goal} 점검이 진행 중입니다. 연결이 끊겨 새 입장 링크를 보내니 지금 다시 들어와 주세요.",
        "지금 {goal} 점검에 재입장 부탁드립니다. 이미 시작했고 끊긴 연결 대신 쓸 새 링크입니다.",
        "새 연결 주소를 보냅니다. {goal} 점검이 이미 시작됐으니 지금 참가해 주세요.",
        "{goal} 점검 참가 연결이 끊겼습니다. 점검은 진행 중이므로 새 링크로 바로 들어오세요.",
        "진행 중인 {goal} 점검에 참가해 주세요. 기존 연결이 끊겨 입장 링크를 다시 보냅니다.",
        "{goal} 점검은 이미 진행하고 있어요. 끊긴 연결을 이 링크로 바꿔 지금 입장해 주세요.",
        "지금 재접속이 필요합니다. {goal} 점검이 시작됐고 새 입장 링크를 전달했습니다.",
        "{goal} 점검 중 연결이 끊겼네요. 새 링크로 지금 돌아와 주세요. 점검은 계속 진행 중입니다.",
    ],
    "waiting": [
        "{goal} 검토가 지금 {field} 답변을 기다리며 멈춰 있습니다. 진행할 수 있게 바로 알려주세요.",
        "지금 {field} 확인이 필요합니다. {goal} 검토 참여자들이 답변을 기다리고 있습니다.",
        "{goal} 검토 중 {field}를 확인하지 못해 진행이 멈췄습니다. 지금 회신해 주세요.",
        "{field} 답변이 있어야 {goal} 검토를 계속할 수 있습니다. 모두 기다리니 바로 확인해 주세요.",
        "실시간 검토가 막혔습니다. {goal}의 {field} 확인 답변을 지금 보내 주세요.",
        "{goal} 검토를 중단한 채 기다리고 있어요. {field} 확인 내용을 바로 부탁드립니다.",
        "지금 진행 중인 {goal} 검토에서 {field} 답변이 필요합니다. 답이 없어 멈춰 있으니 바로 회신해 주세요.",
        "{field} 확인 때문에 {goal} 검토를 이어가지 못합니다. 기다리는 중이니 지금 알려주세요.",
    ],
    "presentation_turn": [
        "{item} 설명 차례가 앞당겨졌습니다. 곧 호출하니 지금 자료를 준비해 주세요.",
        "곧 {item} 설명을 부탁드립니다. 차례가 예상보다 빨라졌으니 지금 자료를 열어 주세요.",
        "지금 {item}를 준비해 주세요. 설명 순서가 당겨져 잠시 뒤 호출합니다.",
        "{item} 발표 순서가 바뀌어 곧 차례입니다. 지금 열고 준비 부탁드립니다.",
        "설명 차례 변경 안내입니다. {item}를 곧 설명해야 하니 지금 자료를 열어 두세요.",
        "{item} 설명 순서를 앞당겼습니다. 잠시 뒤 바로 부르니 지금 준비해 주세요.",
        "곧 호출할 예정입니다. {item} 설명 차례가 빨라져 지금 준비가 필요합니다.",
        "{item}를 설명할 차례가 예상보다 빨리 옵니다. 잠시 후 호출하니 바로 자료를 준비하세요.",
    ],
    "handoff": [
        "{item} 인계 담당자가 현장에서 기다립니다. {field}를 확인하고 지금 답해주세요.",
        "지금 {field} 확인 회신을 부탁드립니다. {item}를 받을 담당자가 인계 장소에 와 있습니다.",
        "{item} 인계를 위해 담당자가 기다리고 있습니다. {field} 확인 후 바로 알려주세요.",
        "인계 현장에서 대기 중입니다. {item}의 {field} 확인 내용을 지금 보내 주세요.",
        "{field} 확인이 필요해 연락드립니다. {item} 담당자가 인계 장소에 있으니 바로 답해 주세요.",
        "{item}를 넘길 담당자가 이미 도착했습니다. 기다리고 있으니 {field}를 지금 확인해 주세요.",
        "지금 인계할 {item}의 {field} 답변을 주세요. 담당자가 현장에서 대기 중입니다.",
        "{item} 인계가 {field} 확인을 기다립니다. 담당자가 현장에 있으니 지금 회신해 주세요.",
    ],
    "wrong_version": [
        "{goal} 설명 중 이전 {item}를 쓰고 있습니다. 지금 최신 자료로 교체해 주세요.",
        "지금 열려 있는 {item}는 이전 버전입니다. {goal} 설명이 진행 중이니 최신판으로 바꿔 주세요.",
        "{goal} 설명에 구판 {item}가 표시됩니다. 진행 중이므로 바로 최신 버전을 열어 주세요.",
        "최신 {item}로 지금 교체 부탁드립니다. {goal} 설명에서 옛 자료를 사용하고 있습니다.",
        "현재 설명 자료가 잘못됐습니다. {goal} 진행에 쓰는 {item}를 바로 최신판으로 바꿔 주세요.",
        "{item} 이전판이 열려 있어요. {goal} 설명을 진행하고 있으니 지금 교체해야 합니다.",
        "진행 중인 {goal} 설명 자료를 확인했습니다. 구판 {item}를 쓰고 있어 바로 최신판 교체가 필요합니다.",
        "{goal} 설명 중입니다. 현재 사용한 {item}는 이전 버전이니 지금 최신 버전으로 전환해 주세요.",
    ],
    "today_slot": [
        "{item}를 함께 볼 수 있는 시간이 지금뿐입니다. 다음 일정으로 가기 전 {field} 질문을 바로 확인해 주세요.",
        "지금 {field} 질문을 확인해 주세요. {item} 공동 확인은 다음 일정 전인 지금만 가능합니다.",
        "곧 다른 일정으로 이동합니다. 지금만 {item}를 함께 볼 수 있으니 {field} 질문에 바로 답해 주세요.",
        "{item} 확인 시간이 지금만 열려 있습니다. 이동 전에 {field} 질문을 바로 봐주세요.",
        "다음 일정 때문에 함께 확인할 시간은 지금뿐입니다. {item}의 {field} 질문을 즉시 확인해 주세요.",
        "{field} 질문을 지금 확인 부탁드립니다. 곧 이동해서 {item}를 같이 볼 수 있는 시간이 끝납니다.",
        "지금 {item}를 같이 볼 수 있습니다. 이후 다른 일정으로 이동하니 {field} 질문을 바로 확인하세요.",
        "{item} 확인을 위해 지금 참여해 주세요. 다음 일정 전 유일한 시간이라 {field} 질문을 바로 봐야 합니다.",
    ],
    "live_field": [
        "{item} 설명 중 {field} 내용이 없어 멈췄습니다. 확인한 내용을 지금 보내 주세요.",
        "진행 중인 {item} 설명에 {field}가 빠져 있습니다. 설명을 계속할 수 있게 바로 알려주세요.",
        "{field} 설명이 필요합니다. 지금 {item} 설명을 중단한 채 기다리니 확인 가능한 내용을 보내 주세요.",
        "{item} 설명을 이어가지 못하고 있어요. {field} 내용이 비어 있으니 지금 확인해 주세요.",
        "지금 {field} 내용을 보내 주세요. {item} 설명이 항목 누락 때문에 멈춰 있습니다.",
        "설명 도중 {field}가 비어 있는 것을 확인했습니다. {item} 진행을 위해 내용을 바로 부탁드립니다.",
        "{item} 설명 중단 안내입니다. {field} 설명을 못 찾아 진행이 멈췄으니 지금 보내 주세요.",
        "{field} 누락으로 {item} 설명을 계속하지 못합니다. 확인 가능한 내용을 바로 전달하세요.",
    ],
    "live_selection": [
        "담당자와 지금 통화 중입니다. {item} 전달에 필요한 {field} 확인이 막혀 있으니 바로 알려주세요.",
        "{item} 전달 담당자가 통화에서 기다립니다. {field} 확인 결과를 지금 보내 주세요.",
        "지금 {field}를 확인해 주세요. {item} 전달이 답변을 기다리고 있으며 담당자는 통화 중입니다.",
        "통화를 이어가려면 {field} 답이 필요합니다. {item} 전달 담당자가 지금 기다리고 있습니다.",
        "{item} 전달을 진행 중이나 {field}를 확인하지 못했습니다. 담당자가 통화 중이니 즉시 알려주세요.",
        "지금 통화 중인 담당자에게 {field} 확인을 전달해야 합니다. {item} 인계가 기다리고 있으니 바로 답해주세요.",
        "{field} 확인 결과를 지금 부탁드립니다. {item} 전달 통화가 답변을 기다리고 있습니다.",
        "{item} 전달을 위한 통화 중입니다. {field} 확인이 필요해 멈췄으니 지금 회신하세요.",
    ],
}


def diversify_training_pool(pool):
    """Revise unreviewed bodies in place; retain candidate and approval snapshots."""
    from filtering_training.common.score_tasks import load_score_samples
    manifest = json.loads((pool/"manifest.json").read_text(encoding="utf-8"))
    revision = manifest["revision"]
    approval = json.loads((pool/"approval.json").read_text(encoding="utf-8"))
    if manifest.get("intended_split") != "train" or (pool/"diversified.jsonl").exists():
        raise ValueError("expected an unmodified training expansion")
    for name,expected in (("candidates.jsonl",manifest["candidate_sha256"]),
                          ("revised.jsonl",revision["dataset_sha256"]),
                          ("approval.json",revision["approval_sha256"]),
                          ("lineage.jsonl",manifest["lineage_sha256"])):
        if digest(pool/name) != expected:
            raise ValueError("reviewed source changed")
    samples = load_score_samples(pool/"revised.jsonl")
    original_rows = {s.notification.id:s.model_dump(mode="json") for s in samples}
    lineage = [json.loads(line) for line in (pool/"lineage.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {s.notification.id:s for s in samples}
    if set(by_id) != {item["notification_id"] for item in lineage}:
        raise ValueError("lineage ids differ")
    protected_originals = {item["original_id"] for item in lineage if item["notification_id"] in approval["approved_ids"]}
    recipes = {goal:{"goal":goal,"item":item,"field":field}
               for _,_,_,text in TRAINING_EXPANSION_DOMAINS
               for goal,item,field in (part.split("|") for part in text.split(";"))}
    tasks = {goal:index for index,goal in enumerate(recipes)}
    if len(TRAINING_EVENT_PHRASINGS) != 30 or any(len(v)!=8 for v in TRAINING_EVENT_PHRASINGS.values()):
        raise ValueError("expected 30 event families with eight phrasings each")
    def render(template,recipe):
        def particle(match):
            word=recipe[match[1]]
            has_final=0xAC00<=ord(word[-1])<=0xD7A3 and (ord(word[-1])-0xAC00)%28!=0
            pairs={"을":("을","를"),"를":("을","를"),"은":("은","는"),"는":("은","는"),"이":("이","가"),"가":("이","가")}
            return word+pairs[match[2]][0 if has_final else 1]
        return re.sub(r"\{(goal|item|field)\}([을를은는이가])",particle,template).format(**recipe)
    changed_originals = set()
    uses = {}
    for item in lineage:
        original = item.get("original_id", "")
        if not original.startswith("train05_") or original.startswith("train05_urgent_") or original in protected_originals:
            continue
        key = item["event_key"]
        index = tasks[item["goal"]] % 8
        by_id[item["notification_id"]].notification.body = render(TRAINING_EVENT_PHRASINGS[key][index],recipes[item["goal"]])
        changed_originals.add(original)
        uses[original]=(key,index)
    repeats=Counter(uses.values())
    if not repeats or max(repeats.values())>20 or len(unique_urgency_samples(samples))!=manifest["originals"]:
        raise ValueError("wording count or original count failed")
    for approved in approval["approved_cases"]:
        if by_id[approved["notification"]["id"]].model_dump(mode="json")!=approved:
            raise ValueError("approved context row changed")
    for sample in samples:
        current=sample.model_dump(mode="json")
        current["notification"]["body"]=original_rows[sample.notification.id]["notification"]["body"]
        if current!=original_rows[sample.notification.id]:
            raise ValueError("wording refinement changed labels or context")
    output=pool/"diversified.jsonl"
    write_lines(output,[s.model_dump(mode="json") for s in samples])
    report={"purpose":"wording refinement of unreviewed general candidates; unchanged event diversity",
            "created_date":"2026-10-04","dataset_file":output.name,"dataset_sha256":digest(output),
            "source_dataset_file":"revised.jsonl","source_dataset_sha256":revision["dataset_sha256"],
            "changed_originals":len(changed_originals),"complete_phrasings":len(repeats),
            "max_phrasing_repeats":max(repeats.values()),"protected_originals":sorted(protected_originals),
            "labels_contexts_unchanged":True,"reviewed_new_bodies":False,
            "limitation":"eight phrasings per event diversify wording; same 150 tasks and 30 event families; urgent wrappers unchanged"}
    manifest["wording_refinement"]=report
    manifest["files"][output.name]="unreviewed wording refinement; labels and all approved originals preserved"
    manifest["quality_gate"]["pending"]=["refined wording preview","training input preparation"]
    manifest["quality_gate"]["phrasing_max_repeats"]=max(repeats.values())
    manifest["quality_gate"]["phrasing_repeat_target_met"]=True
    # The earlier structural-template audit remains true: wording is not event diversity.
    write_json(pool/"manifest.json",manifest)
    return report


def generate_training_expansion(source, output, validation):
    """Append synthetic training candidates; keep original rows and holdout intact."""
    from filtering_training.common.score_tasks import load_score_samples
    from filtering_training.quality.audit_digit_training import audit_pool_separation
    if output.exists():
        raise ValueError("do not overwrite a training expansion")
    base_manifest=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    val_manifest=json.loads((validation/"manifest.json").read_text(encoding="utf-8"))
    preview=json.loads((validation/"approval.json").read_text(encoding="utf-8"))
    if digest(source/"candidates.jsonl")!=base_manifest["candidate_sha256"]:
        raise ValueError("base training pool changed")
    if digest(validation/"candidates.jsonl")!=val_manifest["candidate_sha256"] or digest(validation/"candidates.jsonl")!=preview["candidate_sha256"]:
        raise ValueError("reviewed validation pool changed")
    if digest(validation/"review.md")!=preview["review_sha256"]:
        raise ValueError("reviewed generation criteria changed")
    if digest(source/"lineage.jsonl")!=base_manifest["lineage_sha256"]:
        raise ValueError("base lineage changed")
    base=load_score_samples(source/"candidates.jsonl")
    heldout=load_score_samples(validation/"candidates.jsonl")
    if len(unique_urgency_samples(base))!=5090:
        raise ValueError("expected 5090 base originals")
    normalize=lambda text:re.sub(r"\s+","",text).casefold()
    seen={normalize(s.notification.body) for s in base+heldout}
    recipes=[]
    for domain,process,tool,text in TRAINING_EXPANSION_DOMAINS:
        for goal,item,field in (part.split("|") for part in text.split(";")):
            recipes.append({"domain":domain,"process":process,"tool":tool,"goal":goal,"item":item,"field":field})
    if len(recipes)!=150 or len(TRAINING_URGENT_SCENARIOS)!=41:
        raise ValueError("expected 150 tasks and 41 urgent scenarios")
    keys=["joke_focus","joke_break","archive","color_option","community_news",
          "reference","bookmark","theme","annotation","formats","faq","draft",
          "afternoon","tomorrow","queue","sync_delay","capacity","review_slot","retry","missing_attachment","callback","disagreement",
          "live_access","waiting","presentation_turn","handoff","wrong_version","today_slot","live_field"]
    event_map={event[1]:event for event in BULK_EVENTS}
    events=[event_map[key] for key in keys]
    events.append((4,"live_selection",None,"실시간 항목 확인","지금 {item} 전달이 {field} 확인을 기다리고 있습니다. 담당자가 통화 중이니 바로 알려주세요. 안전 사고나 임박한 마감은 없습니다.","현장 진행이 현재 답변을 기다림"))
    if len(events)!=30:
        raise ValueError("expected 30 general templates")
    def render(template,recipe):
        def particle(match):
            word=recipe[match[1]]
            has_final=0xAC00<=ord(word[-1])<=0xD7A3 and (ord(word[-1])-0xAC00)%28!=0
            pairs={"을":("을","를"),"를":("을","를"),"은":("은","는"),"는":("은","는"),"이":("이","가"),"가":("이","가"),"과":("과","와"),"와":("과","와")}
            return word+pairs[match[2]][0 if has_final else 1]
        template=re.sub(r"\{(goal|item|field)\}([을를은는이가과와])",particle,template)
        return template.format(**recipe)
    rows=[]
    lineage=[]
    timestamp="2026-10-04T12:00:00Z"
    def add_original(identifier,body,urgency,relevance,category,recipe,family,event,reason):
        body_key=normalize(body)
        if body_key in seen:
            raise ValueError(f"new original duplicates existing body: {identifier}: {body}")
        seen.add(body_key)
        note_app,note_tool=("WINWORD.EXE","Word") if recipe["process"]=="Notion.exe" else ("Notion.exe","Notion")
        matching=[(recipe["process"],recipe["goal"]+" - "+recipe["tool"]),
                  (note_app,recipe["item"]+" 작업 메모 - "+note_tool),("explorer.exe",recipe["item"]+" 참고 자료 - 파일 탐색기")]
        other=recipes[(recipes.index(recipe)+47)%len(recipes)] if recipe in recipes else recipes[len(rows)%len(recipes)]
        other_note_app,other_note_tool=("WINWORD.EXE","Word") if other["process"]=="Notion.exe" else ("Notion.exe","Notion")
        switched=[(other["process"],other["goal"]+" - "+other["tool"]),
                  (other_note_app,other["item"]+" 진행 메모 - "+other_note_tool),("explorer.exe",other["item"]+" 참고 자료 - 파일 탐색기")]
        variants=[("matching",recipe["process"],recipe["goal"]+" - "+recipe["tool"],matching,relevance),
                  ("switched",recipe["process"],recipe["goal"]+" - "+recipe["tool"],switched,1)]
        # Add empty-window coverage to a fixed fifth of originals, without
        # calling context rows additional notification originals.
        if int(hashlib.sha256(identifier.encode()).hexdigest()[:8],16)%5==0:
            variants += [("empty_recent","","",matching,relevance),("empty_unknown","","",[],1)]
        for role,process,window,history,rel in variants:
            sample=HistorySample.model_validate({
                "notification":{"id":identifier+"_"+role,"app_name":"KakaoTalk.exe" if category=="개인 일반" else "Slack",
                                "sender":recipe["domain"]+" 담당","title":recipe["domain"]+" 소식","body":body,"timestamp":timestamp},
                "context":{"active_process":process,"window_title":window,"last_updated":timestamp,
                           "duration_seconds":60 if process else 0,"recent_processes":[a for a,_ in history],
                           "recent_windows":[{"app_name":a,"window_title":w} for a,w in history]},
                "label":{"urgency_score":urgency,"relevance_score":rel,"category":category,
                         "ai_summary_reason":reason+"; 검수 전 합성 제안, 최근 작업 흐름 기준."}})
            rows.append(sample)
            lineage.append({"notification_id":sample.notification.id,"original_id":identifier,"family":family,
                            "domain":recipe["domain"],"goal":recipe["goal"],"event_key":event,
                            "context_role":role,"origin":"synthetic_expansion","human_reviewed":False,"train_links":[]})
    for index,recipe in enumerate(recipes):
        for event in events:
            urgency,event_key,override,action,template,reason=event
            relevance=3 if event_key.startswith("joke") or event_key=="community_news" else 2 if event_key in ("color_option","theme") else 4 if event_key in ("reference","formats","faq","archive") else 5
            category=override or "일반 업무"
            add_original(f"train05_{index+1:03d}_{event_key}",render(template,recipe),urgency,relevance,category,
                         recipe,"training_task_"+recipe["goal"],event_key,reason)
    wrappers=[("", ""),("확인 부탁드립니다. ",""),("현장 전달: ",""),("담당자 연락입니다. ",""),
              ("", " 안내를 놓치지 않도록 확인 부탁드립니다."),("", " 지금 담당 연락을 확인해주세요."),
              ("상황 안내입니다. ",""),("", " 현장 안내가 전달됐습니다."),("담당 확인 요청: ",""),("중요 연락입니다. ","")]
    for index,(family,body) in enumerate(TRAINING_URGENT_SCENARIOS):
        goal=TRAINING_URGENT_GOALS[index]
        recipe={"domain":"현장 대응" if index<20 or index>=35 else "필수 확인",
                "process":"chrome.exe","tool":"Chrome","goal":goal,"item":"현장 대응 안내","field":"확인 요청"}
        for rewrite,(prefix,suffix) in enumerate(wrappers,1):
            add_original(f"train05_urgent_{index+1:02d}_{rewrite:02d}",prefix+body+suffix,5,5,"개인 중요",
                         recipe,family,"urgent_notice_style","알림에 실제 위험 또는 임박한 필수 기회 상실이 명시됨")
    originals=unique_urgency_samples(rows)
    if len(originals)!=4910:
        raise ValueError("expected 4910 new originals")
    base_lineage=[json.loads(line) for line in (source/"lineage.jsonl").read_text(encoding="utf-8").splitlines()]
    all_rows=base+rows
    if len(unique_urgency_samples(all_rows))!=10000:
        raise ValueError("expected 10000 combined originals")
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl",[s.model_dump(mode="json") for s in all_rows])
    write_lines(output/"lineage.jsonl",base_lineage+lineage)
    manifest={"purpose":"10000-original synthetic training candidate pool; original pool preserved","created_date":"2026-10-04",
              "intended_split":"train","status":"synthetic_candidates_pending_audit_and_training_preparation",
              "policy_version":POLICY_VERSION,"originals":10000,"rows":len(all_rows),"new_originals":4910,"new_rows":len(rows),
              "candidate_sha256":digest(output/"candidates.jsonl"),"lineage_sha256":digest(output/"lineage.jsonl"),
              "parent_candidate_sha256":digest(source/"candidates.jsonl"),"parent_manifest_sha256":digest(source/"manifest.json"),
              "validation_candidate_sha256":digest(validation/"candidates.jsonl"),"validation_approval_sha256":digest(validation/"approval.json"),
              "validation_reserved_families":sorted(val_manifest["reserved_families"]),
              "new_domains":len(TRAINING_EXPANSION_DOMAINS),"new_work_tasks":len(recipes),"general_notification_templates":len(events),
              "urgent_scenarios":len(TRAINING_URGENT_SCENARIOS),"urgent_wording_variants_per_scenario":len(wrappers),
              "new_individually_reviewed_originals":0,"training_used":False,"model_evaluation_used":False,
              "new_urgency_original_counts":dict(Counter(s.label.urgency_score for s in originals)),
              "new_relevance_row_counts":dict(Counter(s.label.relevance_score for s in rows)),
              "diversity_limitations":"4500 task/event combinations use 30 shared templates; 410 urgent originals are wording variants of 41 events, not 410 independent events",
              "files":{"candidates.jsonl":"existing 5090 originals plus 4910 unreviewed synthetic originals",
                       "lineage.jsonl":"notification, task, event and context grouping","review.md":"small pre-training check and limitations"}}
    write_json(output/"manifest.json",manifest)
    audit=audit_pool_separation(output,validation)
    manifest["validation_isolation_audit"]=audit
    manifest["status"]="synthetic_candidates_pending_review" if audit["passed"] else "blocked_validation_overlap"
    write_json(output/"manifest.json",manifest)
    if not audit["passed"]:
        raise ValueError("training expansion overlaps reserved validation; do not train")
    preview_ids=["train05_001_reference_matching","train05_001_reference_switched",
                 "train05_051_draft_matching","train05_101_joke_focus_matching","train05_131_color_option_matching",
                 "train05_086_afternoon_switched","train05_091_waiting_switched",
                 "train05_urgent_01_01_matching","train05_urgent_21_01_switched","train05_urgent_27_01_switched"]
    by_id={s.notification.id:s for s in rows}
    selected=[by_id[key] for key in preview_ids]
    lines=["<!-- 용도: 1만 원문 학습 후보의 새 사례 검수와 생성 한계 기록.\n생성일: 2026-10-04 (Asia/Seoul)\n상태: 신규 원문 미검수, 모델 학습·평가 미사용. -->\n",
           "# 1만 원문 학습 후보\n",
           f"기존 5,090원문을 그대로 보존하고 4,910 합성 원문을 추가했습니다. 전체 문맥 행은 {len(all_rows)}개입니다. 새 자료의 개별 인간 검수는 아직 0개입니다.\n",
           "새 일반 원문 4,500개는 30영역·150작업과 공통 문장 틀 30종의 조합입니다. 긴급 원문 410개는 사건 41종의 말투 변형 10종입니다. 독립 사건 4,910개를 수작업 작성한 자료가 아닙니다. 이름·숫자만 바꾼 것이 아니라 작업과 사건 상태를 조합했지만 문장 반복 한계가 있으므로 학습 전 감사가 필요합니다.\n"]
    for number,s in enumerate(selected,1):
        lines += [f"## {number}. {s.notification.id}\n",f"알림: {s.notification.body}\n",f"현재 창: {s.context.window_title or '(빈 창)'}\n",
                  "최근 창: "+" / ".join(w.window_title for w in s.context.recent_windows)+"\n",
                  f"제안: 긴급도 **{s.label.urgency_score}**, 관련도 **{s.label.relevance_score}**, **{decision_label(s.label.urgency_score,s.label.relevance_score)}**.\n"]
    lines += ["## 학습 전 확인\n","검증 후보 100원문 전체의 원문·현재/최근 창·선언 계열과 완전 일치 없음. 모든 의미 유사성이 제거됐다는 뜻은 아닙니다. 기존 승인 예외는 원본 그대로 보존했습니다. 원문·사건 계열 단위로 학습 분할에만 넣으며 최종 테스트는 사용하지 않습니다.\n",
              "아직 학습 입력 준비나 어댑터 누적 학습을 실행하지 않았습니다. 반복 문장 감사·긴급도 분포·관련도 기준 검수 후 다음 단계를 진행합니다.\n"]
    (output/"review.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest["review_sha256"]=digest(output/"review.md")
    manifest["displayed_review_ids"]=[s.notification.id for s in selected]
    write_json(output/"manifest.json",manifest)
    return all_rows


def generate_targeted_review(source, output):
    """Create a small training-only preview; require review before bulk expansion."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from filtering_training.common.score_tasks import load_score_samples

    if output.exists():
        raise ValueError("do not overwrite an earlier targeted review")
    protected = [source/"dataset.jsonl",
                 Path("filtering_training/outputs/v5_validation_pool_01/candidates.jsonl"),
                 Path("filtering_training/outputs/v3_reviewed_01/dataset.jsonl")]
    normalize = lambda value: re.sub(r"\s+", "", value).casefold()
    known = {normalize(s.notification.body) for p in protected for s in load_score_samples(p)}
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    timestamp = datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    unrelated = ("chrome.exe", "주말 등산 버스 노선 확인 - Chrome")
    hikes = [("chrome.exe", "주말 등산 버스 노선 확인 - Chrome"),
             ("EXCEL.EXE", "등산 교통비 정리 - Excel"),
             ("Notion.exe", "등산 출발 장소와 준비물 - Notion")]
    energy = [("chrome.exe", "아파트 전기요금 절약 방법 - Chrome"),
              ("EXCEL.EXE", "월별 전력 사용량 비교 - Excel"),
              ("AcroRd32.exe", "가정용 전기요금 구간표.pdf")]
    photo = [("chrome.exe", "판매 사진 흰색 반사 줄이는 방법 - Chrome"),
             ("Photoshop.exe", "중고 물품 판매 사진 보정 - Photoshop"),
             ("explorer.exe", "중고 물품 촬영 원본 - 파일 탐색기")]
    invoice = "택배비 정산 파일에서 이번 달 비용이 두 번 더해졌어요. 30분 뒤 이 금액으로 자동이체가 실행되니 실행 전에 정정 부탁드립니다."
    guide = "광택 있는 물건을 찍을 때 흰색 반사를 줄이는 조명 배치 방법을 정리했어요. 촬영할 때 참고하세요. 나중에 읽어도 됩니다."
    # Same original is paired only when its body and urgency stay unchanged.
    cases = [
        ("01", "billing_execution", "중복 정산 실행 임박", invoice, 4, 1, unrelated, hikes,
         "자동이체 실행 전 금전 오류를 바로잡아야 함. 등산 목적과는 무관."),
        ("01", "billing_execution", "중복 정산 실행 임박", invoice, 4, 1, ("", ""), [],
         "작업 정보가 없어도 자동이체 실행 전 오류 정정의 긴급도는 동일."),
        ("02", "billing_draft", "다음 달 정산 초안", "다음 달 택배비 정산 초안에 같은 항목이 두 번 들어갔어요. 아직 송금 예약은 없고 다음 주 검토 때 고치면 됩니다.",
         2, 1, unrelated, hikes, "실행 예정이나 즉시 피해가 없고 다음 주 수정 가능."),
        ("03", "ordinary_authentication", "계정 접속 확인", "새 기기 접속 확인 요청입니다. 본인이 시작한 접속이면 확인을 눌러 주세요. 낯선 요청이면 취소할 수 있습니다.",
         3, 1, unrelated, hikes, "접속 확인은 비교적 빠른 확인 대상이나 사고 발생은 명시되지 않음."),
        ("04", "confirmed_account_abuse", "계정 변경 진행 중", "본인이 하지 않은 접속에서 복구 이메일 변경이 진행 중인 것을 확인했습니다. 계정 접근을 잃을 수 있으니 지금 세션을 차단하고 변경을 취소해 주세요.",
         5, 1, unrelated, hikes, "확인된 계정 악용과 진행 중인 접근 상실 위험. 등산과 무관해도 즉시 대응."),
        ("05", "practical_photo_reference", "반사 줄이는 촬영 자료", guide, 2, 4,
         ("Photoshop.exe", "중고 물품 판매 사진 보정 - Photoshop"), photo,
         "정확한 물품 지정은 없지만 판매용 사진 품질을 높이는 촬영 방법이 직접 도움."),
        ("06", "decorative_photo_ad", "사진가용 책상 장식", "카메라 렌즈 모양 연필꽂이가 나왔어요. 촬영하는 분의 책상을 꾸며 보세요. 조명이나 사진 편집 기능은 없는 장식품입니다.",
         1, 2, ("Photoshop.exe", "중고 물품 판매 사진 보정 - Photoshop"), photo,
         "사진 분야 장식 광고이며 실제 촬영·보정 목적에는 거의 도움 없음."),
        ("07", "unrelated_social_chat", "영화 모임 잡담", "지난번 영화 마지막 장면 생각나서 또 웃었네ㅋㅋ 다음에 만나면 얘기하자.",
         1, 1, ("EXCEL.EXE", "월별 전력 사용량 비교 - Excel"), energy,
         "영화 잡담은 확인 가능한 전기요금 비교 흐름과 무관."),
        ("08", "exact_reference_from_history", "판매 물품 촬영 순서", "네가 판매 사진 찍는 A17 조명의 밝기별 촬영 순서표를 보냈어. 제품 사진 만들 때 그 순서대로 찍으면 돼. 급하게 확인할 필요는 없어.",
         2, 5, ("", ""), [("Photoshop.exe", "A17 조명 판매용 사진 편집 - Photoshop"),
                               ("EXCEL.EXE", "A17 조명 밝기별 촬영 목록 - Excel"),
                               ("explorer.exe", "A17 조명 판매 사진 원본 - 파일 탐색기")],
         "빈 현재 창이어도 최근 기록이 정확한 A17 조명의 판매 사진 작업을 보여줌."),
        ("05", "practical_photo_reference", "반사 줄이는 촬영 자료", guide, 2, 4,
         ("chrome.exe", "판매 사진 흰색 반사 줄이는 방법 - Chrome"), energy,
         "현재 창의 촬영 방법 확인에 직접 도움. 최근 창이 다른 작업이어도 현재 창을 우선."),
    ]
    rows, lines = [], [f"<!-- 용도: 남은 긴급·관련 경계의 학습 보강 기준 검수.\n생성일: {date} (Asia/Seoul)\n상태: 미검수 후보, 학습·모델 평가 미사용. -->\n",
                       "# 부족 유형 보강 검수 10건\n",
                       "원문 8개·문맥 10행입니다. 점수는 제안이며 기존 검증 오답 원문을 복사하지 않았습니다. 최근 기록의 순서 외 시각이나 작업 전환은 추정하지 않습니다.\n"]
    for index, (original, family, title, body, urgency, relevance, current, history, reason) in enumerate(cases, 1):
        if normalize(body) in known:
            raise ValueError("targeted preview overlaps an existing notification body")
        sample = HistorySample.model_validate({
            "notification": {"id": f"target06_{original}_{index:02d}", "app_name": "Teams", "sender": "검수 후보", "title": title, "body": body, "timestamp": timestamp},
            "context": {"active_process": current[0], "window_title": current[1], "last_updated": timestamp,
                        "duration_seconds": 0, "recent_processes": [a for a, _ in history],
                        "recent_windows": [{"app_name": a, "window_title": w} for a, w in history]},
            "label": {"urgency_score": urgency, "relevance_score": relevance,
                      "category": "광고/홍보" if original == "06" else "개인 일반" if original == "07" else "시스템/보안" if original in ("03", "04") else "일반 업무",
                      "ai_summary_reason": "제안: " + reason}})
        rows.append(sample)
        lines += [f"## {index}. {title}\n", f"알림: {body}\n", f"현재 창: {current[0]} / {current[1]}\n" if current[0] else "현재 창: 비어 있음\n",
                  "최근 창: " + (" / ".join(f"{a}: {w}" for a, w in history) or "없음") + "\n",
                  f"제안: 긴급도 **{urgency}**, 관련도 **{relevance}**, **{decision_label(urgency, relevance)}**. {reason}\n"]
    unique_urgency_samples(rows)
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl", [s.model_dump(mode="json") for s in rows])
    (output/"review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output/"manifest.json", {
        "purpose": "review proposals for targeted training supplementation", "created_date": date,
        "intended_split": "train", "status": "unapproved_candidates", "originals": 8, "rows": len(rows),
        "training_used": False, "model_evaluation_used": False, "candidate_sha256": digest(output/"candidates.jsonl"),
        "review_sha256": digest(output/"review.md"), "policy_version": POLICY_VERSION,
        "displayed_review_ids": [s.notification.id for s in rows],
        "lineage": [{"id": s.notification.id, "original_id": f"target06_{c[0]}", "family": c[1]} for s, c in zip(rows, cases)],
        "protected_sources": [{"path": str(p), "sha256": digest(p)} for p in protected],
        "audit": {"normalized_body_overlap": 0, "urgency_invariance": "passed", "semantic_independence": "requires review; exact-body check alone does not establish it"},
        "expansion_target": {"new_originals": 2000, "approval_required": True, "replay_share_proposal": [0.2, 0.3]},
        "files": {"candidates.jsonl": "unapproved HistorySample proposals", "review.md": "ten displayed cases"}})
    return rows


def targeted_editor_title(goal, original_index):
    """Represent collected editor titles without inventing a task description."""
    index = (original_index - 1001) % 4
    if index in (1, 2):
        return ("review.md" if index == 1 else "manifest.json"), True
    filenames = {"검색 결과 정렬 검사": "search_sort_test.py", "회원 탈퇴 화면 검사": "account_deletion_test.py",
                 "첨부 파일 미리보기 검사": "attachment_preview_test.py", "주소 입력 오류 검사": "address_validation_test.py",
                 "장바구니 수량 검사": "cart_quantity_test.py"}
    return filenames.get(goal, goal.replace(" ", "_") + ".py"), False


def generate_targeted_pool(source, output):
    """Generate training-only proposals with current-window priority and unscored ambiguity."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from filtering_training.common.score_tasks import load_score_samples

    if output.exists():
        raise ValueError("do not overwrite an existing targeted pool")
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    timestamp = datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    protected = [source/"dataset.jsonl",
                 Path("filtering_training/outputs/v5_validation_pool_01/candidates.jsonl"),
                 Path("filtering_training/outputs/v3_reviewed_01/dataset.jsonl")]
    normalize = lambda value: re.sub(r"\s+", "", value).casefold()
    old_rows = [s for p in protected for s in load_score_samples(p)]
    known_bodies = {normalize(s.notification.body) for s in old_rows}
    val_rows = load_score_samples(protected[1])
    reserved_windows = {(s.context.active_process.casefold(), normalize(s.context.window_title)) for s in val_rows if s.context.window_title}
    reserved_windows |= {(w.app_name.casefold(), normalize(w.window_title)) for s in val_rows for w in getattr(s.context, "recent_windows", []) if w.window_title}
    financial_items = ["냉장 운송료", "포장 상자 구매비", "매장 임차료", "행사 장소 대관료", "자격시험 응시료",
                       "학교 급식 납부금", "현장 방문 교통비", "서버 이용 요금", "실험 소모품 구매비", "공용 수도 요금"]
    errors = [("중복 항목", "동일 비용이 두 번 합산되어 청구 금액이 커졌습니다"),
              ("환산 비율", "환산 비율을 잘못 적용해 청구 금액이 커졌습니다"),
              ("수취 계좌", "다른 수취인의 계좌가 지급 계좌에 들어갔습니다"),
              ("환불 반영", "취소한 비용의 환불이 빠져 다시 청구됩니다"),
              ("청구 수량", "실제보다 많은 수량으로 청구 금액이 계산됐습니다")]
    money_states = [
        ("archive", 1, "{item}의 {error} 사례는 지난달에 정정 완료됐고 오늘은 처리 이력만 보관했습니다. {issue}. 추가 조치는 없습니다."),
        ("draft", 2, "{item} 계산 초안에서 {issue}. 지급 예약은 아직 없으니 다음 주 검토 때 {error} 부분을 고치면 됩니다."),
        ("today_review", 3, "{item} 검토 중 {issue}. 실제 지급은 내일 오후입니다. 오늘 퇴근 전 {error} 부분을 확인해 주세요."),
        ("soon", 4, "{item}에서 {issue}. 20분 뒤 잘못된 값으로 지급 승인이 넘어가니 그 전에 {error} 부분을 수정해 주세요."),
        ("running", 5, "{item}에서 {issue}. 잘못된 지급이 지금 실행되고 있습니다. 추가 지급을 막고 취소 가능한 처리를 즉시 중지해 주세요."),
        ("paused", 3, "{item}에서 {issue}. 지급을 일시 정지했고 내일 재개 전에 {error} 부분을 확인해야 합니다. 오늘 안에 검토해 주세요."),
        ("resolved", 1, "{item}의 {error} 문제를 정정하고 지급 취소까지 확인했습니다. 이전에는 {issue}. 재청구나 남은 대응은 없습니다."),
        ("locked", 4, "{item}에서 {issue}. 10분 후 지급 명세가 잠기면 이번 지급을 수정할 수 없습니다. 지금 {error} 부분을 확인해 주세요."),
        ("next_month", 2, "다음 달 {item} 계획표에 {issue}. 승인 전 초안이고 자동 지급도 없으니 다음 정기 점검에서 {error} 부분을 고쳐 주세요."),
        ("waiting", 4, "{item}의 {error} 확인 때문에 현재 지급 승인 회의가 멈춰 있습니다. {issue}. 담당자들이 답을 기다리니 지금 확인 부탁드립니다."),
    ]
    services = ["학교 포털", "매장 재고 서비스", "출장 예약 서비스", "도서 대여 서비스", "사진 백업 서비스",
                "시설 출입 서비스", "소모품 구매 서비스", "수업 자료 서비스", "대관 신청 서비스", "급여 조회 서비스"]
    actions = ["복구 이메일 변경", "새 기기 등록", "이중 인증 설정 변경", "접근 토큰 발급", "계정 자료 내보내기"]
    auth_states = [
        ("request", 3, "{service}에서 {action} 확인 요청이 왔습니다. 직접 요청했다면 승인하고 낯선 요청이라면 거절하세요."),
        ("planned", 2, "{service}의 {action} 절차를 다음 주에 안내할 예정입니다. 본인이 신청한 예약이고 지금 승인할 요청은 없습니다."),
        ("approved", 2, "본인이 승인한 {service}의 {action} 처리가 완료됐습니다. 신청 내용과 일치하며 추가 확인 요청은 없습니다."),
        ("denied", 2, "{service}의 {action} 요청을 거절해 실행되지 않았습니다. 다른 접근은 차단됐고 남은 처리도 없습니다."),
        ("confirmed_pending", 4, "본인이 요청하지 않은 {service}의 {action} 예약을 확인했습니다. 15분 뒤 적용되면 접근이나 자료를 잃을 수 있으니 지금 예약을 취소해 주세요."),
        ("confirmed_running", 5, "본인이 요청하지 않은 {service}의 {action} 처리가 지금 실행 중인 것을 확인했습니다. 계정이나 비공개 자료를 보호하려면 즉시 세션을 막고 작업을 중단해야 합니다."),
        ("expired", 2, "{service}의 {action} 확인 요청은 승인 없이 만료됐습니다. 아무 변경도 적용되지 않았고 새 요청도 없습니다."),
        ("verification", 3, "본인이 시작한 {service}의 {action} 요청입니다. 오늘 중 본인 확인을 하면 이어 진행할 수 있습니다. 확인 전에는 변경되지 않습니다."),
        ("incident_contained", 3, "본인이 하지 않은 {service}의 {action} 처리를 차단하고 기존 접근을 복원했습니다. 침입 경로는 닫혔으며 오늘 중 처리 내역을 검토해 주세요."),
        ("ongoing_access", 5, "본인이 하지 않은 {service}의 {action} 이후 외부 세션이 계속 활동하는 것을 확인했습니다. 추가 변경과 비공개 자료 접근을 막기 위해 지금 접근을 끊어야 합니다."),
    ]
    originals = []
    for item in financial_items:
        for error, issue in errors:
            for state, urgency, template in money_states:
                originals.append({"group": "urgency", "family": "financial_" + state, "scenario": item + " / " + error,
                                  "body": template.format(item=item, error=error, issue=issue), "urgency": urgency,
                                  "relevance": 5, "goal": f"{item}의 {error} 검토", "process": "EXCEL.EXE", "tool": "Excel", "category": "일반 업무"})
    for service in services:
        for action in actions:
            for state, urgency, template in auth_states:
                originals.append({"group": "urgency", "family": "authentication_" + state, "scenario": service + " / " + action,
                                  "body": template.format(service=service, action=action), "urgency": urgency,
                                  "relevance": 5, "goal": f"{service} {action} 확인", "process": "chrome.exe", "tool": "Chrome", "category": "시스템/보안"})
    # Reuse task vocabulary, but author new notification events; do not count it as new domains.
    reference_events = [
        ("practical", 2, 4, "{goal}에서 {field} 부분을 확인하는 방법을 묶어 정리했습니다. 지금 작업에 참고할 수 있고 급하게 읽을 필요는 없습니다."),
        ("exact_artifact", 2, 5, "지금 다루는 {item}의 {field} 부분만 표시한 검토본을 보냈습니다. 해당 항목을 수정할 때 이 검토본을 참고하세요. 나중에 확인해도 됩니다."),
        ("exact_answer", 2, 5, "{item}의 {field}에 남긴 질문에 답을 달았습니다. 지금 확인하는 그 항목의 판단 근거도 함께 적었고 답장 기한은 없습니다."),
        ("method", 2, 4, "{goal} 작업에서 흔히 놓치는 조건을 점검하는 순서를 정리했어요. {field} 확인에도 쓸 수 있습니다. 시간 될 때 보세요."),
        ("decorative_ad", 1, 2, "{goal} 문구가 적힌 장식용 책상 깃발을 판매합니다. 작업 도구나 참고 자료가 아니라 책상을 꾸미는 소품입니다."),
        ("topic_joke", 1, 3, "{goal}만 계속 하다 보니 꿈에서도 {field} 확인할 것 같네ㅋㅋ 그냥 농담이고 확인 요청은 아니야."),
        ("topic_chat", 1, 3, "오늘도 {goal} 얘기가 많네요. {field}라는 말도 자주 들었어요. 구체적인 정보 없이 잡담만 남깁니다."),
        ("wrong_artifact", 2, 2, "다른 담당자가 보는 {neighbor_item} 자료를 보냈습니다. 지금 다루는 {item}에는 적용할 수 없는 별도 작업 자료이며 급한 요청은 아닙니다."),
        ("old_reference", 2, 3, "예전 {item}의 {field} 예시를 찾았습니다. 지금 작업 조건과 다를 수 있어 그대로 쓸 수 있는지는 아직 확인하지 않았습니다. 나중에 읽어도 됩니다."),
        ("usable_tool_ad", 1, 4, "{goal}에 필요한 {field} 점검 기능을 제공하는 도구의 사용 예시를 공개했습니다. 기능을 확인할 수 있는 선택적 홍보이며 구매나 답변 기한은 없습니다."),
    ]
    tasks = []
    for domain, process, tool, text in TRAINING_EXPANSION_DOMAINS[:20]:
        triples = [part.split("|") for part in text.split(";")]
        for i, (goal, item, field) in enumerate(triples):
            tasks.append({"domain": domain, "process": process, "tool": tool, "goal": goal,
                          "item": item, "field": field, "neighbor_item": triples[(i+1) % len(triples)][1]})
    for task in tasks:
        for family, urgency, relevance, template in reference_events:
            originals.append({**task, "group": "relevance", "family": family, "scenario": task["goal"],
                              "body": template.format(**task), "urgency": urgency, "relevance": relevance,
                              "category": "광고/홍보" if family.endswith("ad") else "일반 업무"})
    assert len(originals) == 2000 and len(tasks) == 100
    bodies = [normalize(o["body"]) for o in originals]
    if len(set(bodies)) != 2000 or set(bodies) & known_bodies:
        raise ValueError("targeted originals duplicate existing or new notification bodies")
    rows, unknown, lineage, by_original = [], [], [], {}
    unrelated = [("chrome.exe", "캠핑 침낭 온도별 무게 비교 - Chrome"),
                 ("EXCEL.EXE", "캠핑 침낭 구매 예산 - Excel"), ("Notion.exe", "캠핑 장비 보관 계획 - Notion")]
    vague = [("chrome.exe", "YouTube"), ("chrome.exe", "Instagram"), ("chrome.exe", "YouTube")]
    for i, original in enumerate(originals, 1):
        base = f"targetpool06_{i:04d}"
        process, goal, tool = original["process"], original["goal"], original["tool"]
        active = (process, f"{goal} 실무 점검 - {tool}")
        recent = [(process, f"{goal} 검토 내용 - {tool}"), ("Notion.exe", f"{goal} 작업 메모 - Notion"),
                  ("explorer.exe", f"{goal} 참고 파일 - 파일 탐색기")]
        generic_editor = False
        if process.casefold() == "code.exe":
            filename, generic_editor = targeted_editor_title(goal, i)
            active = (process, filename)
            recent[0] = (process, filename)
        profiles = [("current_matching", active, recent, original["relevance"]),
                    ("current_priority", active, unrelated, original["relevance"]),
                    ("vague_supported", ("chrome.exe", "Instagram"), recent, original["relevance"]),
                    ("empty_supported", ("", ""), recent, original["relevance"]),
                    ("unrelated", unrelated[0], unrelated, 1)]
        notification = {"app_name": "Slack" if i % 3 == 0 else "Teams" if i % 3 == 1 else "KakaoTalk.exe",
                        "sender": "보강 합성 후보", "title": goal, "body": original["body"], "timestamp": timestamp}
        for role, current, history, relevance in profiles:
            context = {"active_process": current[0], "window_title": current[1], "last_updated": timestamp,
                       "duration_seconds": 0, "recent_processes": [a for a, _ in history],
                       "recent_windows": [{"app_name": a, "window_title": w} for a, w in history]}
            if generic_editor and role == "current_priority":
                unknown.append({"notification": {**notification, "id": base + "_" + role}, "context": context,
                                "urgency_score": original["urgency"], "relevance_score": None,
                                "context_status": "generic_editor_title_with_unrelated_history", "relevance_training_eligible": False})
                continue
            sample = HistorySample.model_validate({"notification": {**notification, "id": base + "_" + role}, "context": context,
                "label": {"urgency_score": original["urgency"], "relevance_score": relevance, "category": original["category"],
                          "ai_summary_reason": "합성 제안: 현재 창 우선, 목적 불명확 시 최근 창 보완. 긴급도는 알림 내용만 판단."}})
            for app, window in [(current[0], current[1])] + history:
                if window and (app.casefold(), normalize(window)) in reserved_windows:
                    raise ValueError("targeted context overlaps reserved validation window")
            rows.append(sample)
            by_original[(i, role)] = sample
        unknown.append({"notification": {**notification, "id": base + "_insufficient"},
                        "context": {"active_process": "chrome.exe", "window_title": "Instagram", "last_updated": timestamp,
                                    "duration_seconds": 0, "recent_processes": [a for a, _ in vague],
                                    "recent_windows": [{"app_name": a, "window_title": w} for a, w in vague]},
                        "urgency_score": original["urgency"], "relevance_score": None,
                        "context_status": "insufficient_information", "relevance_training_eligible": False})
        lineage.append({"original_id": base, "group": original["group"], "family": original["family"],
                        "scenario": original["scenario"], "review_status": "synthetic_proposal"})
    unique_urgency_samples(rows)
    # Show complete inputs for meaningful boundaries, rather than only easy cases.
    selection = [(4, "unrelated"), (3, "unrelated"), (501, "unrelated"), (506, "unrelated"),
                 (1001, "current_priority"), (1005, "current_matching"), (1006, "current_matching"),
                 (1002, "empty_supported"), (1001, "vague_supported"), (1001, "insufficient")]
    selected = []
    lines = [f"<!-- 용도: 현재 창 우선과 긴급·관련 경계 보강 후보 검수.\n생성일: {date} (Asia/Seoul)\n상태: 미확정 후보, 학습·모델 평가 미사용. -->\n",
             "# 보강 후보 검수 10건\n", "2,000원문은 합성 후보입니다. 이 검수는 아래 표시한 10행만 대상으로 합니다. 최근 창은 보조 정보이며 모두 모호하면 관련도 정답을 만들지 않습니다.\n"]
    for index, (original_index, role) in enumerate(selection, 1):
        if role == "insufficient":
            value = next(item for item in unknown if item["notification"]["id"] == f"targetpool06_{original_index:04d}_insufficient")
            label_text = f"긴급도 {value['urgency_score']}, 관련도 미정(정보 부족). 관련도 학습에서 제외."
        else:
            sample = by_original[(original_index, role)]
            value = sample.model_dump(mode="json")
            label_text = f"긴급도 {sample.label.urgency_score}, 관련도 {sample.label.relevance_score}, {decision_label(sample.label.urgency_score, sample.label.relevance_score)}."
        selected.append(value["notification"]["id"])
        lines += [f"## {index}. {value['notification']['id']}\n", f"알림: {value['notification']['body']}\n",
                  f"현재 창: {value['context']['active_process']} / {value['context']['window_title'] or '(빈 창)'}\n",
                  "최근 창: " + " / ".join(f"{w['app_name']}: {w['window_title']}" for w in value["context"]["recent_windows"]) + "\n",
                  "제안: " + label_text + "\n"]
    output.mkdir(parents=True)
    write_lines(output/"candidates.jsonl", [s.model_dump(mode="json") for s in rows])
    write_lines(output/"unscored_contexts.jsonl", unknown)
    (output/"review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output/"manifest.json", {
        "purpose": "targeted training-only proposals; preserve uncertain relevance without labels", "created_date": date,
        "intended_split": "train", "status": "synthetic_candidates_pending_review", "policy_version": POLICY_VERSION,
        "originals": 2000, "rows": len(rows), "unscored_context_rows": len(unknown),
        "groups": {"urgency_originals": 1000, "relevance_originals": 1000},
        "task_selection_proposal": {"urgency": "1000 urgency-focused originals plus separately audited replay",
                                    "relevance": "5000 relevance-focused context rows plus separately audited replay; do not blindly include 5000 urgency-focus rows"},
        "diversity": {"financial_subjects": 10, "financial_errors": 5, "financial_states": 10,
                      "account_services": 10, "account_actions": 5, "account_states": 10,
                      "reused_work_tasks": 100, "relevance_event_templates": 10,
                      "independently_human_authored_originals": 0,
                      "limitation": "structured combinations; names, state templates and repeated task vocabulary do not establish 2000 independent scenarios"},
        "context_policy": {"version": "current_window_primary_v1", "current_is_primary": True,
                           "recent_windows_are_supporting": True, "all_vague_relevance_label": None,
                           "editor_titles": "filename only; generic filename plus unrelated history has no relevance gold",
                           "incompatible_with_previous_history_prompt": True},
        "displayed_review_ids": selected, "lineage": lineage,
        "human_reviewed_rows": 0, "training_used": False, "model_evaluation_used": False,
        "candidate_sha256": digest(output/"candidates.jsonl"), "unscored_contexts_sha256": digest(output/"unscored_contexts.jsonl"),
        "review_sha256": digest(output/"review.md"),
        "urgency_original_counts": dict(Counter(o["urgency"] for o in originals)),
        "relevance_row_counts": dict(Counter(s.label.relevance_score for s in rows)),
        "protected_sources": [{"path": str(p), "sha256": digest(p)} for p in protected],
        "audit": {"existing_body_overlap": 0, "reserved_validation_window_overlap": 0,
                  "urgency_invariance": "passed", "semantic_independence": "not exhaustively established"},
        "training_gate": ["review proposed score boundaries", "version current-priority prompt", "audit replay labels against current-priority policy", "reserve original validation/test"],
        "files": {"candidates.jsonl": "proposed labelled history contexts; not frozen training data",
                  "unscored_contexts.jsonl": "same originals in vague contexts, no relevance gold or training eligibility",
                  "review.md": "complete ten-case review", "manifest.json": "counts, provenance, safeguards and lineage"}})
    return rows


def generate_relevance_ladder_review(output):
    """Preview one fixed task before expanding the approved 300-task design."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if output.exists():
        raise ValueError("preserve earlier review; choose a fresh output folder")
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    timestamp = date + "T03:00:00Z"
    history = [
        ("chrome.exe", "동점 검색 결과의 순서가 바뀜 · Issue #184 · shop-search - Google Chrome"),
        ("Notion.exe", "검색 정렬 재현 메모 - Notion"),
        ("Code.exe", "review.md — shop-search — Visual Studio Code"),
    ]
    # Use actual-style titles; task descriptions and reasons are not extra model inputs.
    cases = [
        (1, 1, "토요일 풋살 인원 한 명 비는데 같이 할 사람? 장소는 지난번 운동장이야.",
         "운동 약속은 검색 정렬 오류 테스트와 무관함"),
        (2, 1, "검색창 모양 책상 매트 공동구매 열렸어. 사진 보니까 키보드 놓으면 꽤 예쁘더라.",
         "검색 주제 장식 상품이며 오류 확인을 돕는 기능이나 자료가 없음"),
        (3, 1, "검색 정렬 버그 잡다 보면 내 할 일도 우선순위 정렬 좀 해줬으면 싶다ㅋㅋ",
         "검색 정렬에 관한 농담이며 재현이나 검증 정보는 없음"),
        (3, 1, "우리 팀은 검색 쪽 하는 사람이 제일 많은 듯. 오늘 점심에도 정렬 얘기로 시작했네.",
         "같은 작업 주제의 잡담이지만 구체적인 도움은 없음"),
        (3, 2, "다른 서비스는 인기순으로 바꿨다던데 우리 검색에도 언젠가 넣어볼까? 다음 기획 때 얘기해보자.",
         "검색 정렬 기획 아이디어지만 현재 동점 순서 오류 테스트에 직접 쓰일 정보는 없음"),
        (4, 2, "정렬 결과 테스트할 때 값이 같은 항목, 빈 결과, 페이지 경계를 나눠 확인하는 체크리스트 찾았어. 링크 공유할게.",
         "일반 체크리스트지만 현재 정렬 오류 테스트에 직접 활용 가능함"),
        (4, 2, "stable sort가 동점 항목 순서를 어떻게 유지하는지 설명한 문서야. 언어별 예제도 같이 있어.",
         "동점 정렬 이해에 직접 도움이 되지만 해당 오류의 정확한 구현이나 재현 답은 아님"),
        (4, 2, "같은 입력을 여러 번 실행해 항목 ID 순서가 달라지는지 비교하는 테스트 헬퍼 올렸어. 샘플 데이터만 바꾸면 쓸 수 있어.",
         "현재 순서 변동 검증에 쓸 수 있는 일반 도구이며 해당 이슈 전용 결과는 아님"),
        (5, 2, "#184 재현됐어. score가 같은 상품 12개로 인기순 검색하고 2페이지를 새로고침하면 ID 81과 93의 순서가 바뀌어. 요청 JSON이랑 응답 두 개 올려뒀어.",
         "현재 최근 창에서 확인되는 정확한 이슈의 재현 절차와 결과임"),
        (5, 2, "shop-search의 search_sort_test.py에 #184 회귀 케이스 추가한 diff 보냈어. 같은 score에서 item_id 순서가 유지되는지 검사하고, 지금 브랜치에서는 실패해.",
         "현재 파일과 정확한 오류를 대상으로 한 테스트 변경과 결과임"),
    ]
    context = {"active_process": "Code.exe", "window_title": "search_sort_test.py — shop-search — Visual Studio Code",
               "last_updated": timestamp, "duration_seconds": 0,
               "recent_processes": [app for app, _ in history],
               "recent_windows": [{"app_name": app, "window_title": title} for app, title in history]}
    rows = []
    for index, (rel, urgency, body, reason) in enumerate(cases, 1):
        rows.append(HistorySample.model_validate({
            "notification": {"id": f"ladder07_search_sort_{index:02d}", "app_name": "Teams", "sender": "민수",
                             "title": "검색팀", "body": body, "timestamp": timestamp},
            "context": deepcopy(context),
            "label": {"urgency_score": urgency, "relevance_score": rel, "category": "일반 업무",
                      "ai_summary_reason": "검수 전 제안: " + reason}}))
    normalize = lambda value: re.sub(r"\s+", "", value).casefold()
    known_bodies = set()
    checked = []
    for path in [Path("filtering_training/outputs") / folder / filename for folder, filename in (
        ("v3_reviewed_01", "dataset.jsonl"), ("v5_training_pool_01", "candidates.jsonl"),
        ("v5_validation_pool_01", "candidates.jsonl"), ("v6_targeted_pool_01", "candidates.jsonl"),
        ("v6_targeted_pool_01", "revised.jsonl"))]:
        if not path.exists():
            raise ValueError(f"missing overlap audit source: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            known_bodies.add(normalize(json.loads(line)["notification"]["body"]))
        checked.append({"path": str(path), "sha256": digest(path)})
    bodies = [normalize(row.notification.body) for row in rows]
    if len(set(bodies)) != 10 or set(bodies) & known_bodies:
        raise ValueError("review contains a repeated or previously used notification")
    output.mkdir(parents=True)
    write_lines(output / "candidates.jsonl", [row.model_dump(mode="json") for row in rows])
    lines = ["<!-- 용도: 같은 작업의 관련도 1~5 보강 후보 10건 검수.", "생성일: " + date + " -->", "",
             "# 작업별 관련도 보강 — 첫 검수 10건", "",
             "합의 규모: 작업 300종 × 서로 다른 알림 10개 = 원문 3,000개. 문맥 복제 없이 원문당 한 행.",
             "현재 준비: 아래 한 작업의 원문 10개만. 전체 생성·기존 데이터 병합·학습은 아직 실행하지 않음.",
             "작업 설명은 검수 안내이며 모델에는 알림과 수집 가능한 창 정보만 입력.", "",
             "작업: 검색 정렬 오류 테스트", "",
             "현재 창: " + context["active_process"] + " / " + context["window_title"], "",
             "최근 창: " + " / ".join(app + ": " + title for app, title in history), ""]
    for index, row in enumerate(rows, 1):
        lines += [f"## {index}. {row.notification.id}", "", "알림: " + row.notification.body, "",
                  f"제안: 긴급도 {row.label.urgency_score}, 관련도 {row.label.relevance_score}, "
                  + decision_label(row.label.urgency_score, row.label.relevance_score) + ".", "",
                  "근거: " + row.label.ai_summary_reason, ""]
    (output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "manifest.json", {
        "purpose": "one-task relevance ladder preview before 3000-original expansion", "created_date": date,
        "status": "ten_row_preview_pending_review", "intended_split": "train", "policy_version": POLICY_VERSION,
        "originals": 10, "rows": 10, "work_tasks": 1, "human_reviewed_rows": 0,
        "training_used": False, "model_evaluation_used": False,
        "bulk_plan": {"originals": 3000, "work_tasks": 300, "originals_per_task": 10, "context_rows_per_original": 1,
                      "relevance_per_task": {"1": 1, "2": 1, "3": 3, "4": 3, "5": 2}},
        "candidate_sha256": digest(output / "candidates.jsonl"), "review_sha256": digest(output / "review.md"),
        "audit": {"normalized_body_overlap": 0, "checked_sources": checked,
                  "semantic_overlap": "not established by exact-body audit; review required"},
        "training_gate": ["review ten cases", "author diverse additional work tasks", "audit old replay label conflicts",
                          "version current-window-priority prompt", "preserve existing validation and final test"],
        "files": {"candidates.jsonl": "unapproved ten-row preview", "review.md": "complete preview with fixed window context",
                  "manifest.json": "plan, status and hashes"}})
    return rows


def generate_low_relevance_review(output):
    """Append a two-task low-score preview while preserving the approved ladder."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = output / "supplement.jsonl"
    if target.exists() or manifest.get("low_score_supplement"):
        raise ValueError("preserve existing supplement review")
    if manifest.get("human_reviewed_rows") != 10 or digest(output / "candidates.jsonl") != manifest["candidate_sha256"]:
        raise ValueError("approved initial preview must be intact")
    if digest(output / "review.md") != manifest["review_sha256"]:
        raise ValueError("review text changed")
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    timestamp = date + "T03:00:00Z"
    approved = [json.loads(line) for line in (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    source = Path(manifest["old_replay_conflict_audit"]["source"])
    if digest(source) != manifest["old_replay_conflict_audit"]["source_sha256"]:
        raise ValueError("old training source changed")
    training_payloads = [json.loads(json.loads(line)["messages"][1]["content"])
                         for line in source.read_text(encoding="utf-8").splitlines()]
    stay = next(payload for payload in training_payloads
                if payload["context"]["window_title"] == "장기 투숙 요청 정리 - Excel"
                and payload["notification"]["id"].endswith("_matching"))
    contexts = [deepcopy(approved[0]["context"]), deepcopy(stay["context"])]
    contexts[1]["window_title"] = "장기투숙_요청표.xlsx - Excel"
    contexts[1]["recent_windows"] = [
        {"app_name": "chrome.exe", "window_title": "장기 투숙 연장 요청 - Gmail - Google Chrome"},
        {"app_name": "Notion.exe", "window_title": "장기 투숙 요청 정리 - Notion"},
        {"app_name": "explorer.exe", "window_title": "투숙 요청 자료 - 파일 탐색기"}]
    cases = [
        (0, 1, 1, "오늘 퀴즈: 도원결의에서 유비, 관우, 장비가 모인 곳에는 어떤 나무가 있었을까요?",
         "제목은 검색팀이지만 본문은 역사 퀴즈로 현재 테스트와 무관함"),
        (0, 1, 1, "집에 남는 탁상 선풍기 있는데 필요한 사람 가져가. 사진은 저녁에 올릴게.",
         "개인 물품 나눔으로 현재 테스트에 도움 없음"),
        (0, 2, 1, "검색 아이콘이랑 개발자 문구 들어간 스티커 세트 나왔대. 노트북에 붙인 사진 공유해.",
         "개발 분야 장식 상품으로 테스트 기능이나 자료가 아님"),
        (0, 3, 1, "검색 정렬 케이스 이름 또 길어졌네ㅋㅋ 함수 이름 읽다가 점심시간 끝나겠다.",
         "현재 주제의 농담이며 오류 재현이나 검증 정보는 없음"),
        (0, 4, 2, "정렬 결과 테스트용으로 동점 점수가 많은 샘플 상품 CSV를 만들었어. 원하는 개수로 잘라 쓸 수 있고 특정 이슈 전용은 아니야.",
         "동점 순서 오류 테스트에 직접 쓸 수 있는 일반 입력 자료임"),
        (1, 1, 1, "휴대폰 사진 백업할 때 원본 화질로 저장하는 설정 알아? 여행 사진 용량이 너무 커졌어.",
         "개인 사진 백업 질문은 장기 투숙 요청 처리와 무관함"),
        (1, 1, 1, "요즘 어떤 커피 원두 마셔? 집에서 내려 먹을 만한 거 추천 좀 해줘.",
         "개인 기호 질문이며 투숙객의 요청이나 숙소 운영 업무가 아님"),
        (1, 2, 1, "호텔 로비 사진 들어간 휴대폰 배경화면 모음이야. 마음에 드는 거 있으면 써봐.",
         "숙박 분야 이미지이지만 투숙 요청 정리에 쓰일 정보는 없음"),
        (1, 3, 1, "장기 투숙 요청표 보고 있으니까 나도 휴가 길게 잡고 싶네ㅎㅎ 한 달 쉬면 좋겠다.",
         "현재 주제에서 나온 잡담이며 요청 정리에 도움이 되는 내용은 없음"),
        (1, 4, 2, "투숙 연장 문의를 표로 정리할 때 체크인 날짜, 현재 퇴실일, 희망 연장일을 따로 두는 양식 공유할게. 다른 숙소에도 쓰는 공통 양식이야.",
         "현재 요청표 정리에 직접 활용할 수 있는 일반 양식임"),
    ]
    rows = []
    for index, (task, relevance, urgency, body, reason) in enumerate(cases, 1):
        context = deepcopy(contexts[task])
        context["last_updated"] = timestamp
        context["duration_seconds"] = 0
        context["recent_processes"] = [window["app_name"] for window in context["recent_windows"]]
        rows.append(HistorySample.model_validate({
            "notification": {"id": f"low07_preview_{index:02d}", "app_name": "Teams", "sender": "민수",
                             "title": "검색팀" if task == 0 else "숙소 운영팀", "body": body, "timestamp": timestamp},
            "context": context,
            "label": {"urgency_score": urgency, "relevance_score": relevance, "category": "일반 업무",
                      "ai_summary_reason": "검수 전 제안: " + reason}}))
    normalize = lambda value: re.sub(r"\s+", "", value).casefold()
    known = {normalize(row["notification"]["body"]) for row in approved}
    for entry in manifest["audit"]["checked_sources"]:
        path = Path(entry["path"])
        if digest(path) != entry["sha256"]:
            raise ValueError("overlap source changed")
        known.update(normalize(json.loads(line)["notification"]["body"])
                     for line in path.read_text(encoding="utf-8").splitlines())
    bodies = [normalize(row.notification.body) for row in rows]
    if len(set(bodies)) != 10 or set(bodies) & known:
        raise ValueError("supplement overlaps an earlier original")
    write_lines(target, [row.model_dump(mode="json") for row in rows])
    lines = ["", "## " + date + " — 기존 작업의 낮은 관련도 보강 10건", "",
             "기존 높은 점수 사례는 유지. 기존 작업 150종에 새 알림 20개씩 원문 3,000개를 보강하는 계획.",
             "작업당 제안 분포: 관련도 1점 6개·2점 5개·3점 6개·4점 3개. 문맥 복제 없이 원문당 한 행.",
             "아래 10건은 미승인 제안이며 전체 생성·병합·학습은 아직 수행하지 않음.", ""]
    for index, row in enumerate(rows, 1):
        if index in (1, 6):
            lines += ["### 작업: " + ("검색 정렬 오류 테스트" if index == 1 else "장기 투숙 요청 정리"), "",
                      "현재 창: " + row.context.active_process + " / " + row.context.window_title, "",
                      "최근 창: " + " / ".join(w.app_name + ": " + w.window_title for w in row.context.recent_windows), ""]
        lines += [f"### {index}. {row.notification.id}", "", "알림: " + row.notification.body, "",
                  f"제안: 긴급도 {row.label.urgency_score}, 관련도 {row.label.relevance_score}, "
                  + decision_label(row.label.urgency_score, row.label.relevance_score) + ".", "",
                  "근거: " + row.label.ai_summary_reason, ""]
    with (output / "review.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    manifest["low_score_supplement"] = {
        "purpose": "low-score notifications appended to two existing tasks for review", "created_date": date,
        "status": "ten_row_preview_pending_review", "dataset_file": target.name, "dataset_sha256": digest(target),
        "originals": 10, "rows": 10, "human_reviewed_rows": 0,
        "relevance_counts": dict(Counter(row.label.relevance_score for row in rows)),
        "bulk_plan": {"originals": 3000, "existing_work_tasks": 150, "new_originals_per_task": 20,
                      "context_rows_per_original": 1, "relevance_per_task": {"1": 6, "2": 5, "3": 6, "4": 3},
                      "preserve_old_high_score_examples": True},
        "supersedes": "previous bulk_plan of 300 new tasks; initial approved ten-row preview remains preserved",
        "displayed_review_ids": [row.notification.id for row in rows],
        "old_training_source_sha256": digest(source), "normalized_body_overlap": 0,
        "training_used": False, "model_evaluation_used": False}
    manifest["review_sha256"] = digest(output / "review.md")
    manifest["files"][target.name] = "ten unapproved low-score supplement originals; separate from approved initial preview"
    write_json(manifest_path, manifest)
    return rows


LOW_SCORE_RESOURCES = [
    ("검색·입력 테스트", "동점·빈값·페이지 경계를 다루는 웹 테스트 체크리스트", "버튼 클릭과 입력 순서를 기록하는 재현 절차 양식", "기대값과 실제값을 나란히 비교하는 테스트 결과 양식"),
    ("서비스 운영", "시간대별 기록을 구간으로 묶는 집계 방법", "서비스 기록에서 누락·중복을 찾는 확인 절차", "실행 시각과 처리 상태를 함께 정리하는 로그 분류 양식"),
    ("데이터 정리", "빈칸과 표기 차이를 구분하는 데이터 정제 체크리스트", "변환 전후 값을 나란히 놓는 대조표 양식", "원본을 보존하면서 정리 결과를 검산하는 절차"),
    ("사용성 조사", "인터뷰 발언을 행동·이유·불편으로 나누는 코딩 예시", "관찰 사실과 조사자의 해석을 분리하는 기록 양식", "사용자 발언을 익명화해 인용하는 정리 방법"),
    ("화면 디자인", "선택·진행·완료 상태를 구분하는 UI 표시 체크리스트", "문구 길이와 터치 영역을 확인하는 화면 검토 방법", "간격과 대비를 비교하는 공통 디자인 점검표"),
    ("고객 상담", "문의 내용과 고객의 요청을 분리하는 상담 분류 양식", "처리 상태별 답변 항목을 비교하는 상담 기록 방법", "중복 문의를 묶고 예외를 별도 기록하는 절차"),
    ("매장 운영", "위치·수량·담당자를 함께 기록하는 매장 점검표", "고객에게 적용 날짜와 대상 범위를 알리는 공지 양식", "확인 완료와 후속 조치를 구분하는 운영 체크리스트"),
    ("숙박 운영", "객실·날짜·요청 내용을 별도 열로 두는 숙박 기록 양식", "숙박 문의를 종류별로 묶는 접수 분류 예시", "날짜 범위와 중복 배정을 확인하는 숙박 운영 점검 방법"),
    ("행정 회계", "영수증·결제 내역·지출 항목을 대조하는 정산 체크리스트", "금액·날짜·증빙 유무를 분리하는 비용 기록 양식", "원본 증빙을 보존하며 누락 항목을 표시하는 방법"),
    ("인사 운영", "신청 내용·담당자·처리 상태를 분리하는 인사 접수 양식", "개인정보를 가리고 업무 목록을 공유하는 방법", "중복 배정과 미처리 신청을 찾는 인사 업무 점검표"),
    ("교육 운영", "학습자 요청을 내용·시간·대상별로 분류하는 양식", "안내문에서 대상·일정·제출 방법을 확인하는 체크리스트", "교육 운영 기록을 항목별로 묶는 정리 예시"),
    ("도서관", "도서관 문의를 접수 경로와 처리 상태별로 묶는 양식", "이용자 안내의 시간·위치·예외를 점검하는 방법", "도서관 업무 기록에서 중복 접수를 대조하는 절차"),
    ("연구 지원", "연구 지원 문서에서 개인정보와 연락 항목을 점검하는 방법", "연구 지원 신청의 대상·일정·승인 상태를 분리하는 양식", "연구 기록의 항목명과 단위를 일관되게 쓰는 체크리스트"),
    ("행사 운영", "참가자 요청·담당자·처리 여부를 분리하는 행사 운영표", "시간대별 인원과 중복 배정을 확인하는 행사 점검 방법", "접수 순서와 예외 사항을 함께 정리하는 행사 기록 양식"),
    ("물품 대여", "대여·반납·부속품 상태를 분리하는 물품 관리 양식", "인수인계 수량과 기록을 대조하는 체크리스트", "물품별 담당자와 사용 장소를 함께 기록하는 방법"),
    ("교통 안내", "교통 안내에서 적용 구간·날짜·대상을 확인하는 점검표", "이용자 문의의 출발·도착·요청 내용을 분리하는 양식", "기존 안내와 변경 안내를 나란히 비교하는 교정 절차"),
    ("주차 운영", "차량·사용 기간·처리 상태를 분리하는 주차 기록 양식", "주차 안내의 적용 구역과 예외를 확인하는 체크리스트", "출입·신청 기록의 중복과 누락을 찾는 대조 방법"),
    ("주거 관리", "방문·수령·이용 요청을 대상과 날짜별로 정리하는 양식", "공용 공간 안내의 위치와 이용 조건을 점검하는 방법", "주거 관리 접수와 완료 내역을 대조하는 체크리스트"),
    ("돌봄 지원", "돌봄 요청의 대상·시간·담당자를 분리하는 일정 양식", "돌봄 연락 기록에서 미확인 요청을 표시하는 방법", "방문별 인계 사항과 준비 항목을 대조하는 점검표"),
    ("반려동물 용품", "반려동물 용품의 치수·소재·세척 조건을 비교하는 양식", "용품 설명에서 사용 조건과 제한을 확인하는 체크리스트", "용품별 상태와 관리 이력을 기록하는 방법"),
    ("스포츠 운영", "경기·연습 참가자와 시간대를 나눠 적는 운영 양식", "체육 시설의 중복 배정과 인원 제한을 점검하는 방법", "팀·진행 순서·담당자를 함께 정리하는 대회 운영표"),
    ("정원 관리", "정원 작업의 위치·상태·담당자를 구분하는 기록 양식", "정원 물품의 사용·반납·보관 상태를 확인하는 체크리스트", "정원 작업 구역별 일정을 겹치지 않게 정리하는 방법"),
    ("목공", "목공 도면에서 치수·간섭·고정 위치를 확인하는 점검표", "목재 부품의 연결 방향을 비교하는 도면 검토 방법", "목재 치수와 보관 위치를 기록하는 자재 목록 양식"),
    ("금속 가공", "가공 도면의 치수·공차·방향 표시를 확인하는 점검표", "금속 부품의 처리 이력을 구분하는 공정 기록 양식", "금속 자재의 두께·길이·보관 위치를 비교하는 방법"),
    ("제품 검사", "검사 항목·결과·사진을 연결하는 제품 검사 양식", "제품 검사에서 관찰 사실과 판정 기준을 분리하는 방법", "검사 기록의 부품명·사진명·표시 문구를 대조하는 체크리스트"),
    ("물류 상담", "배송 문의에서 주문·수령인·요청 내용을 분리하는 양식", "배송 기록과 고객 문의를 대조하는 확인 절차", "배송 접수 사유와 처리 상태를 함께 분류하는 예시"),
    ("사진 정리", "사진의 촬영 정보·용도·선택 사유를 함께 적는 정리 양식", "사진을 나란히 비교하며 중복과 차이를 찾는 방법", "원본 사진을 보존하면서 후보를 표시하는 관리 절차"),
    ("음성 편집", "녹음 구간의 시작·끝·내용을 나눠 기록하는 양식", "오디오를 반복 청취하며 차이를 표시하는 편집 방법", "원본 음성을 보존하면서 편집 구간을 표시하는 절차"),
    ("개인 일정", "방문 일정의 장소·시간·준비 항목을 나누는 양식", "예약·방문 안내에서 조건과 필요 서류를 확인하는 방법", "개인 일정의 이동 시간과 중복을 비교하는 체크리스트"),
    ("취미 기록", "취미 작업의 단계·재료·결과를 나눠 적는 기록 양식", "비교할 대상을 같은 조건으로 나란히 기록하는 방법", "작업 순서와 선택 이유를 함께 남기는 취미 기록 예시"),
]


def generate_low_relevance_bulk(output):
    """Build a train-only proposal; keep approved rows and old datasets intact."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preview = manifest["low_score_supplement"]
    target = output / "supplement_bulk.jsonl"
    if target.exists() or "bulk_candidates" in preview:
        raise ValueError("preserve existing bulk candidates")
    if preview.get("human_reviewed_rows") != 10 or "approval" not in preview:
        raise ValueError("low-score preview must be reviewed first")
    protected = {output / "candidates.jsonl": manifest["candidate_sha256"],
                 output / "supplement.jsonl": preview["dataset_sha256"],
                 output / "review.md": manifest["review_sha256"]}
    for entry in manifest["audit"]["checked_sources"]:
        protected[Path(entry["path"])] = entry["sha256"]
    source = Path(manifest["old_replay_conflict_audit"]["source"])
    protected[source] = manifest["old_replay_conflict_audit"]["source_sha256"]
    for task_name in ("urgency", "relevance"):
        for split in ("train", "validation"):
            path = source.parent.parent / task_name / (split + ".jsonl")
            protected[path] = digest(path)
    holdout = Path("filtering_training/outputs/v3_reviewed_01/prepared/test.jsonl")
    if not holdout.exists():
        raise ValueError("final test preservation check requires the existing file")
    protected[holdout] = digest(holdout)
    for path, expected in protected.items():
        if digest(path) != expected:
            raise ValueError(f"protected source changed: {path}")
    approved = [json.loads(line) for line in (output / "supplement.jsonl").read_text(encoding="utf-8").splitlines()]
    normalize = lambda text: re.sub(r"\s+", "", text).casefold()
    known = set()
    for entry in manifest["audit"]["checked_sources"]:
        known.update(normalize(json.loads(line)["notification"]["body"])
                     for line in Path(entry["path"]).read_text(encoding="utf-8").splitlines())
    known.update(normalize(json.loads(line)["notification"]["body"])
                 for line in (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines())
    tasks = []
    for domain_index, (domain, process, tool, text) in enumerate(TRAINING_EXPANSION_DOMAINS):
        for goal, item, field in (part.split("|") for part in text.split(";")):
            tasks.append(dict(domain=domain, process=process, tool=tool, goal=goal,
                              item=item, field=field, domain_index=domain_index))
    assert len(tasks) == 150 and len(LOW_SCORE_RESOURCES) == 30
    old_payloads = [json.loads(json.loads(line)["messages"][1]["content"])
                    for line in source.read_text(encoding="utf-8").splitlines()]
    old_windows = {value["context"]["window_title"] for value in old_payloads
                   if value["notification"]["id"].endswith("_matching")}
    assert all(task["goal"] + " - " + task["tool"] in old_windows for task in tasks)
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    timestamp = date + "T03:00:00Z"
    unrelated_templates = [
        "{goal} 하면서 {field} 이야기를 하다가 갑자기 저녁 메뉴 토론이 됐네ㅋㅋ",
        "{item} 작업 화면을 보니 {goal} 처음 맡았을 때 생각나네. 그때 책상이 엄청 좁았어.",
        "{goal} 자료 중 {field} 부분을 다룬 설명을 동료에게 받았어. {item} 담당자에게 전달해 둘게.",
        "{item}의 {field} 기록을 별도 보관했어. {goal} 담당자가 다음에 찾아볼 수 있게 해 뒀어.",
        "{goal} 주제로 만든 장식용 그림이야. {item} 데이터는 없고 {field} 문구만 그림에 들어가 있어.",
        "{item}에서 {field} 내용을 찾을 때 쓸 목차를 정리했어. {goal} 담당자에게 필요하면 보내 줄게.",
    ]
    decorative_templates = [
        "{motif} 그림이 들어간 스티커 팩 구경해 봐. 노트북 겉면에 붙인 사진도 올라왔어.",
        "책상 꾸미기용 {motif} 포스터가 새로 나왔대. 인테리어 사진을 보내 줬어.",
        "{motif} 문구가 인쇄된 머그컵 공동 구매 이야기 나왔어. 관심 있으면 색상만 골라 봐.",
        "휴대폰 배경화면에 쓸 {motif} 일러스트 모음이야. 마음에 드는 그림 있으면 저장해.",
        "{motif} 그림 달린 열쇠고리 사진 공유해. 가방 장식으로 만든 거래.",
    ]
    topic_templates = [
        "{goal} 하다 보니 {field}보다 내 점심 메뉴가 더 고민이다ㅋㅋ",
        "{item} 펼쳐 놓으니까 {field} 이야기보다 커피 먼저 마시고 싶네ㅎㅎ",
        "{goal} 처음 맡았던 때 기억나? 그때는 사무실이 지금보다 훨씬 좁았지.",
        "{item} 담당하는 사람들끼리 {field}라는 이름이 너무 딱딱하다는 얘기만 한참 했어.",
        "나중에 {goal} 소재로 짧은 만화 그리면 재밌겠다. {field}를 캐릭터 이름으로 쓰는 거야.",
        "{goal} 끝내면 다 같이 산책하자는 얘기 나왔어. {item} 내용 말고 쉬는 시간 얘기야.",
    ]
    resource_templates = [
        "{resource}를 정리한 자료를 공유할게. 예시를 보면서 항목을 잡는 데 참고할 수 있어.",
        "자료함에 ‘{resource}’ 자료가 있어. 적용 방법과 예시가 같이 적혀 있으니 필요할 때 봐.",
        "{resource}를 설명한 글을 보내. 여러 상황에 쓰는 공통 절차와 빈 양식이 함께 있어.",
    ]
    rows, task_records, seen = [], [], set()
    levels = [1] * 6 + [2] * 5 + [3] * 6 + [4] * 3
    approved_map = {("검색 결과 정렬 검사", slot): value
                    for slot, value in zip((0, 1, 6, 11, 17), approved[:5])}
    approved_map.update({("장기 투숙 요청 정리", slot): value
                         for slot, value in zip((0, 1, 6, 11, 17), approved[5:])})
    approved_contexts = {"검색 결과 정렬 검사": approved[0]["context"],
                         "장기 투숙 요청 정리": approved[5]["context"]}
    extensions = {"Code.exe": ".py", "EXCEL.EXE": ".xlsx", "WINWORD.EXE": ".docx",
                  "FreeCAD.exe": ".FCStd", "audacity.exe": ".aup3"}
    for task_index, task in enumerate(tasks):
        if task["goal"] in approved_contexts:
            context = deepcopy(approved_contexts[task["goal"]])
        else:
            stem = (task["goal"] if task["process"] == "Code.exe" else task["item"]).replace(" ", "_")
            extension = extensions.get(task["process"], "")
            current = stem + extension + " - " + task["tool"]
            recent = [{"app_name": "chrome.exe", "window_title": task["goal"] + " - Gmail - Google Chrome"},
                      {"app_name": "Notion.exe", "window_title": task["goal"] + " 작업 메모 - Notion"},
                      {"app_name": "explorer.exe", "window_title": task["item"] + " - 파일 탐색기"}]
            context = dict(active_process=task["process"], window_title=current,
                           last_updated=timestamp, duration_seconds=0,
                           recent_processes=[window["app_name"] for window in recent], recent_windows=recent)
        ids = []
        for slot, relevance in enumerate(levels):
            value = approved_map.get((task["goal"], slot))
            if value is not None:
                row = HistorySample.model_validate(value)
            else:
                if relevance == 1:
                    other = tasks[(task_index + 75) % 150]
                    body = unrelated_templates[slot].format(**other)
                    urgency = 1 if slot in (0, 1, 4) else 2
                    reason = "다른 분야의 대상과 행동에 대한 내용이며 현재 목적과 무관함"
                elif relevance == 2:
                    # The same decorative body can be shared across five tasks.
                    # Give each task a different visual subject rather than an ID/name suffix.
                    subjects = ("선으로 그린", "수채화풍", "작은 아이콘 형태의", "만화풍", "픽셀 아트풍")
                    motif = subjects[task_index % 5] + " " + LOW_SCORE_RESOURCES[task["domain_index"]][0]
                    body = decorative_templates[slot - 6].format(motif=motif)
                    urgency, reason = 1, "관련 분야의 장식물로 실제 작업에 필요한 자료나 기능은 아님"
                elif relevance == 3:
                    body = topic_templates[slot - 11].format(**task)
                    urgency, reason = 1, "현재 주제를 언급하지만 잡담·회상·농담으로 직접적인 도움은 없음"
                else:
                    resource = LOW_SCORE_RESOURCES[task["domain_index"]][slot - 16]
                    # Vary a substantive example, without claiming an exact task artifact.
                    scopes = ("처음 기록할 때", "여러 담당자가 함께 정리할 때", "기존 기록을 다시 확인할 때",
                              "새 자료를 기존 자료와 비교할 때", "정리한 결과를 다른 사람에게 전달할 때")
                    body = resource_templates[slot - 17].format(resource=resource) + " " + scopes[task_index % 5] + " 쓸 수 있어."
                    urgency, reason = 2, "현재 작업에 활용할 수 있는 일반 절차·양식이며 정확한 대상의 결과물은 아님"
                row = HistorySample.model_validate({
                    "notification": dict(id=f"low07_bulk_{task_index + 1:03d}_{slot + 1:02d}", app_name="Teams",
                                         sender="동료", title=task["domain"] + " 대화", body=body, timestamp=timestamp),
                    "context": deepcopy(context),
                    "label": dict(urgency_score=urgency, relevance_score=relevance, category="일반 업무",
                                  ai_summary_reason="검수 전 제안: " + reason)})
            key = normalize(row.notification.body)
            if key in seen or key in known:
                raise ValueError(f"duplicate body: {row.notification.id}")
            if row.label.relevance_score != relevance:
                raise ValueError("approved score contradicts task distribution")
            seen.add(key)
            rows.append(row)
            ids.append(row.notification.id)
        task_records.append(dict(task_index=task_index + 1, domain=task["domain"], goal=task["goal"],
                                 ids=ids, relevance_counts=dict(Counter(levels))))
    assert len(rows) == 3000 and len(seen) == 3000
    by_id = {row.notification.id: row.model_dump(mode="json") for row in rows}
    assert all(by_id[value["notification"]["id"]] == value for value in approved)
    review_positions = [(3, 0), (3, 7), (3, 12), (3, 18), (17, 18),
                        (39, 18), (114, 18), (134, 18), (139, 18), (149, 18)]
    review_rows = [rows[task * 20 + slot] for task, slot in review_positions]
    lines = ["", "## " + date + " — 3,000원문 후보 분야별 검수", "",
             "총 3,000원문·150작업·원문당 1행. 아래 10건은 새 미승인 제안이며 기존 승인 10건은 그대로 포함.",
             "구조화된 작업·메시지 조합이며 3,000개 독립 시나리오 작성이나 전량 인간 검수가 아님.", ""]
    for index, row in enumerate(review_rows, 1):
        lines += [f"### {index}. {row.notification.id}", "", "알림: " + row.notification.body, "",
                  "현재 창: " + row.context.active_process + " / " + row.context.window_title, "",
                  "최근 창: " + " / ".join(w.app_name + ": " + w.window_title for w in row.context.recent_windows), "",
                  f"제안: 긴급도 {row.label.urgency_score}, 관련도 {row.label.relevance_score}.", "",
                  "근거: " + row.label.ai_summary_reason, ""]
    # Verify source hashes before writing the candidates or review metadata.
    for path, expected in protected.items():
        if digest(path) != expected:
            raise ValueError(f"source changed during generation: {path}")
    write_lines(target, [row.model_dump(mode="json") for row in rows])
    with (output / "review.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    preview["bulk_candidates"] = {
        "purpose": "train-only low-relevance proposal preserving approved preview", "created_date": date,
        "status": "2990_synthetic_rows_pending_review_not_merged", "dataset_file": target.name,
        "dataset_sha256": digest(target), "originals": 3000, "rows": 3000, "work_tasks": 150,
        "human_reviewed_rows": 10, "approved_ids": list(preview["approval"]["approved_ids"]),
        "relevance_counts": dict(Counter(row.label.relevance_score for row in rows)),
        "urgency_counts": dict(Counter(row.label.urgency_score for row in rows)),
        "context_rows_per_original": 1, "task_records": task_records,
        "diversity": {"domains": 30, "message_families": 20, "general_resource_topics": 90,
                      "limitation": "template combinations; visual styles and repeated wording do not establish independent scenarios"},
        "normalized_body_overlap": 0, "preserved_sources": {str(path): value for path, value in protected.items() if path != output / "review.md"},
        "next_review_ids": [row.notification.id for row in review_rows],
        "training_used": False, "model_evaluation_used": False, "merged_into_training": False,
        "context_policy": "clear current window is primary; no context replication or history-driven relabeling",
        "training_gate": ["review domain applicability of score 4 resources", "audit semantic/template similarity",
                          "review existing history-label conflicts before replay merge", "freeze and group-split only after review"]}
    manifest["files"][target.name] = "3000 low-score originals including ten approved preview rows; not a frozen training set"
    manifest["review_sha256"] = digest(output / "review.md")
    write_json(manifest_path, manifest)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("filtering_training/outputs/v3_reviewed_01"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--batch",choices=("supplement","expansion","urgent","bulk","revise-bulk","expand-urgent","validation-review","history-review","revise-history","validation-pool","training-expansion","refine-training","targeted-review","targeted-pool","relevance-ladder-review","low-relevance-review","low-relevance-bulk"),default="supplement")
    parser.add_argument("--feedback",type=Path)
    parser.add_argument("--approved",type=Path,default=Path("filtering_training/outputs/v3_expansion_review_01_approved"))
    args=parser.parse_args()
    if args.batch == "low-relevance-bulk":
        output = args.output or Path("filtering_training/outputs/v7_relevance_ladder_01")
        samples = generate_low_relevance_bulk(output)
        print('Prepared', len(samples), 'low-score original proposals in', output)
        return
    if args.batch == "low-relevance-review":
        output = args.output or Path("filtering_training/outputs/v7_relevance_ladder_01")
        samples = generate_low_relevance_review(output)
        print('Prepared', len(samples), 'unapproved low-score originals in', output)
        return
    if args.batch == "relevance-ladder-review":
        output = args.output or Path("filtering_training/outputs/v7_relevance_ladder_01")
        samples = generate_relevance_ladder_review(output)
        print('Prepared', len(samples), 'unapproved relevance ladder originals in', output)
        return
    if args.batch == "targeted-pool":
        output = args.output or Path("filtering_training/outputs/v6_targeted_pool_01")
        samples = generate_targeted_pool(args.source, output)
        print('Prepared', len(samples), 'proposed context rows in', output)
        return
    if args.batch == "targeted-review":
        output = args.output or Path("filtering_training/outputs/v6_targeted_review_01")
        samples = generate_targeted_review(args.source, output)
        print('Prepared', len(samples), 'unapproved targeted rows in', output)
        return
    if args.batch == "refine-training":
        print(json.dumps(diversify_training_pool(args.source),ensure_ascii=False))
        return
    if args.batch == "training-expansion":
        output=args.output or Path("filtering_training/outputs/v5_training_pool_01")
        samples=generate_training_expansion(args.source,output,args.approved)
        print('Prepared',len(samples),'training candidate rows in',output)
        return
    if args.batch == "validation-pool":
        output=args.output or Path("filtering_training/outputs/v5_validation_pool_01")
        samples=generate_validation_pool(args.source,output)
        print('Prepared',len(samples),'validation candidate rows in',output)
        return
    if args.batch == "revise-history":
        if args.feedback is None:
            parser.error("revise-history requires --source history folder and --feedback")
        samples = revise_history_review(args.source,args.feedback)
        print('Revised',len(samples),'history rows in',args.source)
        return
    if args.batch == "history-review":
        output = args.output or Path("filtering_training/outputs/v5_history_review_01")
        samples = generate_history_review(args.source,output)
        print('Prepared',len(samples),'unapproved history rows in',output)
        return
    if args.batch == "validation-review":
        output = args.output or Path("filtering_training/outputs/v5_validation_review_01")
        samples = generate_validation_review(args.source,output)
        print('Prepared',len(samples),'reserved validation candidate rows in',output)
        return
    if args.batch == "expand-urgent":
        if args.output is None:
            parser.error("expand-urgent requires --source candidate pool, --approved urgent review and --output")
        samples = expand_urgent_states(args.source,args.approved,args.output)
        print('Combined',len(samples),'rows in',args.output)
        return
    if args.batch == "revise-bulk":
        if args.output is None or args.feedback is None:
            parser.error("revise-bulk requires --source candidate folder, --output and --feedback")
        samples = revise_bulk(args.source, args.output, args.feedback)
        print('Revised', len(samples), 'candidate rows in', args.output)
        return
    output = args.output or Path("filtering_training/outputs/v4_urgent_review_01" if args.batch=="urgent" else "filtering_training/outputs/v4_diverse_5000_candidates_01" if args.batch=="bulk" else "filtering_training/outputs/v3_expansion_review_01" if args.batch=="expansion" else "filtering_training/outputs/v3_score_review_02")
    samples=generate_bulk(args.source,output,args.approved) if args.batch=="bulk" else generate(args.source,output,args.batch=="expansion",args.batch=="urgent")
    print('Prepared',len(samples),'unapproved context rows in',output)


if __name__=="__main__":
    main()
