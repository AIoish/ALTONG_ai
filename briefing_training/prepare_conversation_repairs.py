"""Prepare independent v9 conversation repairs; never overwrite v8 data."""

from __future__ import annotations

import json
from pathlib import Path

from .prepare_category_dataset import DATA_DIR as V8_DIR, validate_records
from .prepare_dataset import write_jsonl
from .smoke_test_model import load_cases


DATA_DIR = Path(__file__).with_name("data") / "category_briefing_v9"


def repair_cases(split: str) -> list[dict]:
    labels = {"train": ["영상", "문헌", "방문", "이용", "주거", "환경", "문화", "교육", "취업", "지역", "건강", "교통"],
              "validation": ["체육", "음악", "관광"], "evaluation": ["독서"]}
    if split not in labels:
        raise ValueError("invalid repair split")
    result = []
    for index, subject in enumerate(labels[split]):
        day = index + (2 if split == "train" else 16 if split == "validation" else 25)
        date = f"11월 {day}일"
        room = f"연구관 {index + 101}호"
        cases = (
            ("security_negation", "시스템/보안",
             [f"{subject} 계정 로그인 알림 떴어요", "부산에서 노트북으로 들어왔대요", "제 로그인은 아니에요", "기기 차단하고 세션 종료됐어요"],
             [f"부산의 노트북에서 감지된 {subject} 계정 로그인을 본인 활동이 아닌 것으로 확인해 기기를 차단하고 세션을 종료했습니다."],
             [[f"{subject} 계정"], ["부산"], ["노트북"], ["본인 활동이 아닌"], ["차단"], ["세션"], ["종료"]],
             ["로그인을 수행", "본인 로그인입니다"]),
            ("incident_subject", "긴급 업무",
             [f"{subject} 야간 배치 있잖아요", "계속 멈춰서 중단됐어요", "지금 재시작 중이에요"],
             [f"{subject} 야간 배치가 반복해서 중단되어 현재 재시작 중입니다."],
             [[subject], ["야간 배치"], ["중단"], ["재시작"]], ["주간 배치", "재확인해야"]),
            ("task_deliverable", "일반 업무",
             [f"{subject} 조사 정리 자료요", "응답 표 빠진 내용 있죠", "그거 추가해서", f"{date} 오후 2시까지 분석 채널에 올려주세요"],
             [f"{subject} 조사 정리 자료에 응답 표를 추가해 {date} 오후 2시까지 분석 채널에 올려야 합니다."],
             [[f"{subject} 조사"], ["응답 표"], ["추가"], [date], ["오후 2시"], ["분석 채널"]], ["예약해"]),
            ("schedule_details", "일정/회의",
             [f"{subject} 연구 참가자들과 연구 결과 중간 점검해요", f"{date} 오후 7시에 만나요", f"{room}로 와 주세요", "실험 순서 확인을 위해 오프라인으로 진행해요"],
             [f"{subject} 연구 참가자들과 연구 결과 중간 점검을 {date} 오후 7시 {room}에서 진행합니다.", "실험 순서 확인을 위해 오프라인으로 진행합니다."],
             [[f"{subject} 연구 참가자"], ["중간 점검"], [date], ["오후 7시"], [room], ["실험 순서 확인"], ["오프라인"]], []),
            ("appointment_korean", "개인 중요",
             [f"{subject} 프로그램 상담 예약이요", f"{date} 오후 4시로 바뀌었어요", f"{room}로 와 주세요"],
             [f"{subject} 프로그램 상담 예약이 {date} 오후 4시로 변경되어 {room}로 방문해야 합니다."],
             [[f"{subject} 프로그램"], ["상담 예약"], [date], ["오후 4시"], [room], ["변경"]], ["reservation", "therapy"]),
            ("delivery_subject", "개인 중요",
             [f"{subject} 교재 주문한 거요", "배송 완료됐어요", "무인 택배함에 뒀어요"],
             [f"주문한 {subject} 교재가 무인 택배함에 배송 완료되었습니다."],
             [[f"{subject} 교재"], ["무인 택배함"], ["배송"], ["완료"]], []),
            ("document_not_event", "일반 업무",
             [f"{subject} 행사 안내문 검토하고 올려 주세요"],
             [f"{subject} 행사 안내문을 검토한 후 올려야 합니다."],
             [[f"{subject} 행사 안내문"], ["검토"], ["올려"]], ["예약", "마감", "오후", "업무 진행 관련"]),
            ("security_final_state", "시스템/보안",
             [f"{subject} 계정 비밀번호 재설정 요청했어요", "재설정 완료됐고 모든 기기에서 로그아웃됐어요"],
             [f"{subject} 계정의 비밀번호 재설정이 완료되어 모든 기기에서 로그아웃되었습니다."],
             [[f"{subject} 계정"], ["비밀번호 재설정"], ["완료"], ["모든 기기"], ["로그아웃"]], ["요청했습니다", "요청되었습니다", "요청했어요"]),
        )
        for kind, category, bodies, target, facts, forbidden in cases:
            case_id = f"{split}_repair_{kind}_{index:03d}"
            record = {"case_id": case_id, "input": {"app_name": "KakaoTalk.exe", "sender": "가상 참가자", "notifications": [
                {"id": f"{case_id}_{n}", "timestamp": f"2026-10-10T09:{n:02d}:00Z", "title": "연구 대화방", "body": body}
                for n, body in enumerate(bodies)]}, "max_summary_lines": 3,
                "metadata": {"split": split, "scenario": f"repair_{kind}", "synthetic": True, "task": "category_briefing_v9"}}
            if split == "evaluation":
                record.update(expected_category=category, reference_summary_lines=target,
                              expected_facts=facts, forbidden_phrases=forbidden,
                              expected_max_summary_lines=len(target), require_korean=True)
            else:
                record["target"] = {"primary_category": category, "summary_lines": target}
            result.append(record)
    return result


def main() -> None:
    from .train_lora import validate_split_separation
    from .evaluate import validate_evaluation_cases

    train = load_cases(V8_DIR / "train_cases.jsonl") + repair_cases("train")
    validation = load_cases(V8_DIR / "validation_cases.jsonl") + repair_cases("validation")
    evaluation = repair_cases("evaluation")
    validate_records(train, expected_split="train")
    validate_records(validation, expected_split="validation")
    validate_split_separation(train, validation)
    validate_evaluation_cases(evaluation)
    eval_records = [{**r, "target": {"primary_category": r["expected_category"], "summary_lines": r["reference_summary_lines"]}} for r in evaluation]
    validate_split_separation(train, eval_records)
    validate_split_separation(validation, eval_records)
    for name, rows in (("train_cases.jsonl", train), ("validation_cases.jsonl", validation), ("conversation_evaluation.jsonl", evaluation)):
        write_jsonl(DATA_DIR / name, rows)
        print(f"Wrote {len(rows)} records to {DATA_DIR / name}")


if __name__ == "__main__":
    main()
