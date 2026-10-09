"""Derive a separate v7 classification+summary dataset; preserve v6 sources."""

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


DATA_DIR = Path(__file__).with_name("data") / "category_briefing"
TRAIN_PATH = DATA_DIR / "train_cases.jsonl"
VALIDATION_PATH = DATA_DIR / "validation_cases.jsonl"
EVALUATION_PATH = DATA_DIR / "evaluation_cases.jsonl"


def migrate_record(source: Mapping[str, Any], *, evaluation: bool = False) -> dict:
    record = deepcopy(dict(source))
    group = record["input"]
    category = group.pop("category")
    group.pop("urgency_score", None)
    group.pop("relevance_score", None)
    if app_identity(group["app_name"]) == "kakaotalk":
        # Older titles were subjects/senders, not room names. Preserve title-only
        # subject facts in body rather than losing them during migration.
        for notification in group["notifications"]:
            old_title = notification["title"]
            if old_title != group["sender"] and old_title not in notification["body"]:
                notification["body"] = f"{old_title}. {notification['body']}"
            notification["title"] = "알림 대화방"
    if evaluation:
        record["expected_category"] = category
    else:
        record["target"] = {"primary_category": category, **record["target"]}
        record["metadata"]["task"] = "category_briefing_v7"
        record["metadata"]["label_source"] = "existing_synthetic_category"
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
    evaluation = [migrate_record(record, evaluation=True) for record in load_cases(DEFAULT_CASES_PATH)]
    validate_records(train, expected_split="train")
    validate_records(validation, expected_split="validation")
    from .train_lora import validate_split_separation
    from .evaluate import validate_evaluation_cases
    validate_split_separation(train, validation)
    validate_evaluation_cases(evaluation)
    for name, records in (("train_cases.jsonl", train), ("validation_cases.jsonl", validation),
                          ("evaluation_cases.jsonl", evaluation)):
        target = args.output_dir / name
        if target.resolve() in {path.resolve() for path in (DEFAULT_TRAIN_PATH, DEFAULT_VALIDATION_PATH, DEFAULT_CASES_PATH)}:
            raise ValueError("output must not overwrite the original datasets")
        write_jsonl(target, records)
        print(f"Wrote {len(records)} category+summary cases to {target}")


if __name__ == "__main__":
    main()
