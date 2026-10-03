"""Apply explicit human score edits to a new filtering dataset revision."""

from filtering_training.common.paths import TRAINING_ROOT, LEGACY_OUTPUTS_ROOT

import argparse
import csv
import hashlib
import io
import json
import shutil
from collections import Counter
from pathlib import Path

from filtering_training.quality.audit_dataset import audit_samples
from filtering_training.common.dataset import load_samples
from src.filtering.prompt import CATEGORIES, parse_model_output


ROOT = TRAINING_ROOT
DEFAULT_SOURCE = LEGACY_OUTPUTS_ROOT / "candidates/combined_korean_5000.jsonl"
DEFAULT_FEEDBACK = LEGACY_OUTPUTS_ROOT / "review/first_pass_5000/review_batch_01_resolved.csv"
DEFAULT_OUTPUT = LEGACY_OUTPUTS_ROOT / "review/reviewed_5000_v1.jsonl"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_feedback(path: Path) -> tuple[list[dict[str, str]], str]:
    data = path.read_bytes()
    try:
        content = data.decode("utf-8-sig")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = data.decode("cp949")
        encoding = "cp949"
    rows = list(csv.DictReader(io.StringIO(content)))
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("feedback contains no rows or duplicate sample IDs")
    return rows, encoding


def _score(value: str, field: str) -> int | None:
    if not value.strip():
        return None
    try:
        score = int(value)
    except ValueError as error:
        raise ValueError(f"invalid {field}: {value}") from error
    if not 1 <= score <= 5:
        raise ValueError(f"{field} must be 1..5: {score}")
    return score


def apply_feedback(source: Path, feedback: Path, output: Path) -> dict:
    samples = load_samples(source)
    by_id = {sample.notification.id: sample for sample in samples}
    rows, encoding = _load_feedback(feedback)
    changes = []
    superseded_focus_notes = []
    blank_edits = []
    statuses = Counter()
    for row in rows:
        identifier = row["sample_id"]
        if identifier not in by_id:
            raise ValueError(f"feedback ID is absent from dataset: {identifier}")
        sample = by_id[identifier]
        label = sample.label
        if (row["title"], row["body"]) != (sample.notification.title, sample.notification.body):
            raise ValueError(f"notification text changed in feedback: {identifier}")
        if (row["urgency_score"], row["relevance_score"], row["category"]) != (
            str(label.urgency_score), str(label.relevance_score), label.category
        ):
            raise ValueError(f"original labels changed in feedback: {identifier}")
        status = row["review_status"].strip().upper()
        if status not in {"APPROVE", "EDIT", "REMOVE", "UNSURE"}:
            raise ValueError(f"invalid review status: {identifier}: {status}")
        statuses[status] += 1
        if status == "REMOVE":
            raise ValueError(f"REMOVE requires a separate family-level review: {identifier}")
        updates = {}
        for column, label_field in (
            ("corrected_urgency", "urgency_score"),
            ("corrected_relevance", "relevance_score"),
        ):
            score = _score(row[column], column)
            if score is not None and score != getattr(label, label_field):
                updates[label_field] = score
        category = row["corrected_category"].strip()
        if category:
            if category not in CATEGORIES:
                raise ValueError(f"invalid category: {identifier}: {category}")
            if category != label.category:
                updates["category"] = category
        reason = row["corrected_reason"].strip()
        if reason and reason != label.ai_summary_reason:
            updates["ai_summary_reason"] = reason
        if status == "APPROVE" and updates:
            raise ValueError(f"APPROVE has corrections: {identifier}")
        if status == "EDIT" and not updates:
            if row["review_notes"].strip():
                raise ValueError(f"EDIT without structured correction has notes: {identifier}")
            blank_edits.append(identifier)
        if "비집중" in row["review_notes"]:
            superseded_focus_notes.append(identifier)
        if updates:
            before = label.model_dump()
            sample.label = label.model_copy(update=updates)
            parse_model_output(json.dumps(sample.label.model_dump(), ensure_ascii=False))
            changes.append({"sample_id": identifier, "before": before, "updates": updates})

    audit = audit_samples(samples)
    if audit["contradictory_notification_groups"] or audit["conflicting_identical_inputs"]:
        raise ValueError("feedback introduced contradictory labels")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
    lineage_source = source.with_suffix(".lineage.json")
    lineage_target = output.with_suffix(".lineage.json")
    shutil.copyfile(lineage_source, lineage_target)
    report = {
        "source_sha256": _sha256(source.read_bytes()),
        "feedback_sha256": _sha256(feedback.read_bytes()),
        "feedback_encoding": encoding,
        "output_sha256": _sha256(output.read_bytes()),
        "reviewed_rows": len(rows),
        "review_status_counts": dict(statuses),
        "explicit_changes": changes,
        "blank_edits_treated_as_no_change": blank_edits,
        "focus_notes_superseded_by_active_mode_rule": superseded_focus_notes,
        "focus_mode_rule": "while focus mode is on, empty window context does not imply idle or change PASS/BLOCK policy",
        "audit": audit,
        "status": "provisional; only explicit score edits applied; no family-wide propagation",
    }
    output.with_suffix(".feedback.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = apply_feedback(args.source, args.feedback, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "audit"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
