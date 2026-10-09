"""Prepare policy-aligned v8 data, preserving v6 and v7 snapshots."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.briefing.category_prompt import build_messages, parse_response
from src.briefing.clustering import app_identity

from .prepare_dataset import DEFAULT_TRAIN_PATH, DEFAULT_VALIDATION_PATH, write_jsonl
from .smoke_test_model import DEFAULT_CASES_PATH, load_cases


DATA_DIR = Path(__file__).with_name("data") / "category_briefing_v8"
TRAIN_PATH = DATA_DIR / "train_cases.jsonl"
VALIDATION_PATH = DATA_DIR / "validation_cases.jsonl"
EVALUATION_PATH = DATA_DIR / "evaluation_cases.jsonl"

# Explicit scenario review, not predictions used as evaluation ground truth.
POLICY_VERSION = "briefing_v8"
LABEL_OVERRIDES = {
    "assignment_deadline": "일반 업무", "deadline_extended": "일반 업무",
    "notice_corrected": "일정/회의", "orientation_venue_corrected": "일정/회의",
    "fragmented_maintenance_rescheduled": "시스템/보안",
    "maintenance_schedule_corrected": "시스템/보안",
    "delivery_completed": "개인 중요", "delivery_completed_at_new_location": "개인 중요",
    "refund_completed": "개인 중요", "fragmented_refund_final_state": "개인 중요",
    "fragmented_colloquial_delivery": "개인 중요", "colloquial_delivery_chat": "개인 중요",
    "fragmented_subject_delivery_result": "개인 중요",
    "fragmented_subscription_renewal": "개인 중요",
}


def short_request_records(split: str) -> list[dict]:
    """Disjoint subjects per split; teach concise, source-grounded answers."""
    if split not in {"train", "validation", "evaluation"}:
        raise ValueError("invalid short-request split")
    subjects = (["활동 계획서", "운영 안내문", "홍보 초안", "참여 신청서", "진행 자료",
                 "설문 초안", "실습 안내서", "동아리 소개글", "정산 문서", "행사 소개서"]
                if split == "train" else ["봉사 안내서", "신입 안내문"] if split == "validation"
                else ["체험 프로그램 안내문"])
    records = []
    for index, subject in enumerate(subjects):
        obj = subject + ("을" if (ord(subject[-1]) - 0xAC00) % 28 else "를")
        for kind, bodies, summary, category in (
            ("review", [f"{subject} 검토해 주세요."], f"{obj} 검토해야 합니다.", "일반 업무"),
            ("upload", [f"{subject} 검토하고 올려주세요."], f"{obj} 검토한 후 올려야 합니다.", "일반 업무"),
            ("completed", [f"{subject} 업로드 완료했어요."], f"{subject} 업로드가 완료되었습니다.", "일반 업무"),
            ("fragments", [f"{subject}요", "검토하고", "공유 폴더에 올려주세요."],
             f"{obj} 검토한 후 공유 폴더에 올려야 합니다.", "일반 업무"),
            ("security", [f"{subject} 계정 비밀번호 재설정 완료", "모든 기기 로그아웃됐어요."],
             f"{subject} 계정의 비밀번호 재설정이 완료되어 모든 기기에서 로그아웃되었습니다.", "시스템/보안"),
            ("personal", [f"오늘 {subject} 읽어봤어", "재밌더라"],
             f"{obj} 읽고 재미있었다고 전했습니다.", "개인 일반"),
            ("missing_time", [f"{subject} 검토 회의 내일 할게요."],
             f"{subject} 검토 회의는 내일 진행됩니다.", "일정/회의"),
            ("latest", [f"{subject} 올리는 중이에요", "업로드 완료됐어요."],
             f"{subject} 업로드가 완료되었습니다.", "일반 업무"),
        ):
            case_id = f"{split}_short_{kind}_{index:03d}"
            records.append({
                "case_id": case_id,
                "input": {"app_name": "KakaoTalk", "sender": "가상 팀원", "notifications": [
                    {"id": f"{case_id}_{n}", "timestamp": f"2026-10-02T08:{n:02d}:00Z",
                     "title": "자료 모임방", "body": body} for n, body in enumerate(bodies)]},
                # Teach that 3 is a ceiling: still produce just one useful line.
                "max_summary_lines": 3,
                "target": {"primary_category": category, "summary_lines": [summary]},
                "metadata": {"split": split, "scenario": f"short_{kind}", "synthetic": True,
                             "task": "category_briefing_v8", "label_source": POLICY_VERSION},
            })
    return records


def short_evaluation_records() -> list[dict]:
    records = short_request_records("evaluation")
    for record in records:
        target = record.pop("target")
        record["expected_category"] = target["primary_category"]
        record["expected_max_summary_lines"] = 1
        record["reference_summary_lines"] = target["summary_lines"]
        record["expected_facts"] = [["체험 프로그램 안내문"]]
        kind = record["metadata"]["scenario"]
        facts = {
            "short_review": [["검토"]],
            "short_upload": [["검토"], ["올려", "업로드"]],
            "short_completed": [["업로드"], ["완료"]],
            "short_fragments": [["검토"], ["공유 폴더"], ["올려", "업로드"]],
            "short_security": [["비밀번호 재설정"], ["완료"], ["모든 기기"], ["로그아웃"]],
            "short_personal": [["읽"], ["재미", "재밌"]],
            "short_missing_time": [["회의"], ["내일"]],
            "short_latest": [["업로드"], ["완료"]],
        }
        record["expected_facts"].extend(facts[kind])
        record["forbidden_phrases"] = ["자료 모임방", "업무 진행 관련", "마감", "오후", "오전", "서울"]
        if kind != "short_missing_time":
            record["forbidden_phrases"].extend(["내일", "10월"])
        # A concise target must pass even with runtime's three-line ceiling.
        record["metadata"]["manual_review"] = "한 문장으로 충분한지, 원문에 없는 정보가 없는지 확인"
    return records


def migrate_record(source: Mapping[str, Any], *, evaluation: bool = False) -> dict:
    record = deepcopy(dict(source))
    group = record["input"]
    category = group.pop("category")
    scenario = record.get("metadata", {}).get("scenario", record["case_id"])
    category = LABEL_OVERRIDES.get(scenario, category)
    group.pop("urgency_score", None)
    group.pop("relevance_score", None)
    if app_identity(group["app_name"]) == "kakaotalk":
        # Older titles were subjects/senders, not room names. Preserve title-only
        # subject facts in body rather than losing them during migration.
        for notification in group["notifications"]:
            old_title = notification["title"]
            if (old_title != group["sender"] and old_title not in notification["body"]
                    and not old_title.endswith(("채팅", "채팅방", "대화방"))):
                notification["body"] = f"{old_title}. {notification['body']}"
            notification["title"] = "알림 대화방"
    if evaluation:
        record["expected_category"] = category
    else:
        record["target"] = {"primary_category": category, **record["target"]}
        record["metadata"]["task"] = "category_briefing_v8"
        record["metadata"]["label_source"] = POLICY_VERSION
    if evaluation and record["case_id"] == "password_reset_completed":
        record["reference_summary_lines"] = ["비밀번호 재설정이 완료되어 모든 기기에서 로그아웃되었습니다."]
    return record


def training_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = build_messages(record["input"], max_summary_lines=record["max_summary_lines"])
    return [*messages, {"role": "assistant", "content": json.dumps(
        record["target"], ensure_ascii=False, separators=(",", ":"),
    )}]


def validate_records(records: Sequence[Mapping[str, Any]], *, expected_split: str) -> None:
    seen = set()
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("missing or duplicate category dataset case_id")
        seen.add(case_id)
        if record.get("metadata", {}).get("split") != expected_split:
            raise ValueError(f"{case_id}: invalid split")
        if "category" in record["input"]:
            raise ValueError(f"{case_id}: category label must not be visible in input")
        limit = record.get("max_summary_lines")
        build_messages(record["input"], max_summary_lines=limit)
        decision = parse_response(json.dumps(record["target"], ensure_ascii=False))
        if len(decision.summary_lines) > limit:
            raise ValueError(f"{case_id}: target exceeds line limit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    train = [migrate_record(record) for record in load_cases(DEFAULT_TRAIN_PATH)]
    validation = [migrate_record(record) for record in load_cases(DEFAULT_VALIDATION_PATH)]
    train.extend(short_request_records("train"))
    validation.extend(short_request_records("validation"))
    evaluation = [migrate_record(record, evaluation=True) for record in load_cases(DEFAULT_CASES_PATH)]
    short_evaluation = short_evaluation_records()
    validate_records(train, expected_split="train")
    validate_records(validation, expected_split="validation")
    from .train_lora import validate_split_separation
    from .evaluate import validate_evaluation_cases
    validate_split_separation(train, validation)
    validate_evaluation_cases(evaluation)
    validate_evaluation_cases(short_evaluation)
    for name, records in (("train_cases.jsonl", train), ("validation_cases.jsonl", validation),
                          ("evaluation_cases.jsonl", evaluation),
                          ("short_request_evaluation.jsonl", short_evaluation)):
        target = args.output_dir / name
        if target.resolve() in {path.resolve() for path in (DEFAULT_TRAIN_PATH, DEFAULT_VALIDATION_PATH, DEFAULT_CASES_PATH)}:
            raise ValueError("output must not overwrite the original datasets")
        write_jsonl(target, records)
        print(f"Wrote {len(records)} category+summary cases to {target}")


if __name__ == "__main__":
    main()
