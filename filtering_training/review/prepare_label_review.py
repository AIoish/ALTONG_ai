"""Create a blind label-review sheet and a separate, prediction-aware error queue."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from filtering_training.modeling.compare_evaluations import compare
from filtering_training.common.dataset import load_samples
from filtering_training.common.paths import resolve_existing_path
from src.filtering.policy import should_pass


BLIND_COLUMNS = ("notification_id", "app_name", "sender", "title", "body",
                 "active_process", "window_title", "duration_seconds",
                 "recent_processes", "reviewed_urgency", "reviewed_relevance",
                 "reviewed_category", "reviewer_note")
TRIAGE_COLUMNS = ("priority", "notification_id", "title", "body", "window_title",
                  "gold_urgency", "gold_relevance", "gold_category", "gold_pass",
                  "first_urgency", "first_relevance", "first_category", "first_pass",
                  "second_urgency", "second_relevance", "second_category", "second_pass",
                  "first_policy_error", "second_policy_error")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(dataset: Path, first: tuple[Path, Path], second: tuple[Path, Path],
            output_dir: Path) -> dict:
    dataset = resolve_existing_path(dataset)
    first = tuple(resolve_existing_path(path) for path in first)
    second = tuple(resolve_existing_path(path) for path in second)
    result = compare(dataset, [first, second])
    samples = load_samples(dataset)
    by_id = {sample.notification.id: sample for sample in samples}
    blind_rows = []
    for sample in samples:
        alert, ctx = sample.notification, sample.context
        blind_rows.append({"notification_id": alert.id, "app_name": alert.app_name,
                           "sender": alert.sender, "title": alert.title,
                           "body": alert.body, "active_process": ctx.active_process,
                           "window_title": ctx.window_title,
                           "duration_seconds": ctx.duration_seconds,
                           "recent_processes": ", ".join(ctx.recent_processes),
                           "reviewed_urgency": "", "reviewed_relevance": "",
                           "reviewed_category": "", "reviewer_note": ""})
    random.Random(42).shuffle(blind_rows)
    first_predictions = {item["notification_id"]: item["model_output"] for item in
                         json.loads(first[1].read_text(encoding="utf-8"))["prediction_examples"]}
    second_predictions = {item["notification_id"]: item["model_output"] for item in
                          json.loads(second[1].read_text(encoding="utf-8"))["prediction_examples"]}
    triage_rows = []
    for identifier, sample in by_id.items():
        gold = sample.label
        first_label = first_predictions[identifier]
        second_label = second_predictions[identifier]
        expected = should_pass(gold.urgency_score, gold.relevance_score)
        first_pass = should_pass(first_label["urgency_score"], first_label["relevance_score"]) if first_label else None
        second_pass = should_pass(second_label["urgency_score"], second_label["relevance_score"]) if second_label else None
        first_error, second_error = first_pass != expected, second_pass != expected
        priority = 0 if second_error and gold.urgency_score >= 4 else (1 if second_error else (2 if first_error else 3))
        row = {"priority": priority, "notification_id": identifier,
               "title": sample.notification.title, "body": sample.notification.body,
               "window_title": sample.context.window_title,
               "gold_urgency": gold.urgency_score,
               "gold_relevance": gold.relevance_score,
               "gold_category": gold.category, "gold_pass": expected,
               "first_urgency": first_label["urgency_score"] if first_label else "",
               "first_relevance": first_label["relevance_score"] if first_label else "",
               "first_category": first_label["category"] if first_label else "",
               "first_pass": first_pass,
               "second_urgency": second_label["urgency_score"] if second_label else "",
               "second_relevance": second_label["relevance_score"] if second_label else "",
               "second_category": second_label["category"] if second_label else "",
               "second_pass": second_pass,
               "first_policy_error": first_error, "second_policy_error": second_error}
        triage_rows.append(row)
    triage_rows.sort(key=lambda row: (row["priority"], row["notification_id"]))
    blind_path, triage_path = output_dir / "blind_review.csv", output_dir / "error_triage.csv"
    write_csv(blind_path, BLIND_COLUMNS, blind_rows)
    write_csv(triage_path, TRIAGE_COLUMNS, triage_rows)
    manifest = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "count": len(samples), "blind_sheet": str(blind_path),
                "error_triage": str(triage_path),
                "priority_0_urgent_second_model_error": sum(row["priority"] == 0 for row in triage_rows),
                "priority_1_other_second_model_error": sum(row["priority"] == 1 for row in triage_rows),
                "guidance": "Complete blind_review.csv before opening error_triage.csv; keep all cases out of training."}
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", action="append", nargs=2, type=Path, required=True,
                        metavar=("REPORT", "PREDICTIONS"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        parser.error("exactly two --run pairs are required")
    print(json.dumps(prepare(args.dataset, tuple(args.run[0]), tuple(args.run[1]), args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
