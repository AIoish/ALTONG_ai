"""Select a reproducible, local-only pool of external notification texts.

These records have no ALTONG labels or contexts. They must be rewritten in Korean,
screened for personal information, and relabeled before use in training.
"""

from filtering_training.common.paths import LEGACY_OUTPUTS_ROOT

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


SOURCE_URL = "https://huggingface.co/datasets/charlesfeng1/notifai-dataset"
SOURCE_PATH = LEGACY_OUTPUTS_ROOT / "external" / "notifai" / "training_data.jsonl"
OUTPUT_PATH = LEGACY_OUTPUTS_ROOT / "external" / "notifai" / "selected_3000.jsonl"
FOLDERS = ("Work", "Personal", "Alerts", "Promotions")
SENSITIVE_PATTERN = re.compile(r"https?://|www\.|[\w.+-]+@[\w.-]+|\b\d{7,}\b", re.I)


def text_key(title: str, body: str) -> tuple[str, str]:
    return tuple(" ".join(value.split()).casefold() for value in (title, body))


def eligible_record(row: object) -> tuple[dict | None, str]:
    if not isinstance(row, dict):
        return None, "invalid_record"
    notification = row.get("notification")
    classification = row.get("classification")
    if not isinstance(notification, dict) or not isinstance(classification, dict):
        return None, "missing_fields"
    app = notification.get("app_display_name")
    title = notification.get("title")
    body = notification.get("body")
    folder = classification.get("folder")
    priority = classification.get("priority")
    if (not all(isinstance(value, str) for value in (app, title, body))
            or not isinstance(row.get("id"), str)
            or folder not in FOLDERS
            or type(priority) is not int or priority not in range(1, 6)):
        return None, "missing_fields"
    app, title, body = app.strip(), title.strip(), body.strip()
    if not app or not 4 <= len(title) <= 80 or not 8 <= len(body) <= 180:
        return None, "length"
    if any(char.isalpha() and not char.isascii() for char in title + body):
        return None, "non_english_script"
    if SENSITIVE_PATTERN.search(title + " " + body):
        return None, "sensitive_pattern"
    return {
        "source_id": row["id"], "app": app, "title": title, "body": body,
        "source_folder": folder, "source_priority": priority,
    }, "eligible"


def select_candidates(source: Path, output: Path, total: int = 3000,
                      max_per_app: int = 300) -> dict:
    if total < 1 or total % len(FOLDERS):
        raise ValueError("total must be positive and divisible by four folders")
    if max_per_app < 1:
        raise ValueError("max_per_app must be positive")
    rejection_counts: Counter[str] = Counter()
    unique: dict[tuple[str, str], dict] = {}
    source_count = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            source_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at source line {line_number}") from error
            candidate, status = eligible_record(row)
            if candidate is None:
                rejection_counts[status] += 1
                continue
            key = text_key(candidate["title"], candidate["body"])
            if key in unique:
                rejection_counts["duplicate_text"] += 1
                continue
            unique[key] = candidate

    by_folder = {folder: [] for folder in FOLDERS}
    for candidate in unique.values():
        by_folder[candidate["source_folder"]].append(candidate)
    # The hash-based order is stable even if the source file is reordered.
    for folder in FOLDERS:
        by_folder[folder].sort(key=lambda item: (
            hashlib.sha256(item["source_id"].encode("utf-8")).hexdigest(),
            item["source_id"],
        ))
    quota = total // len(FOLDERS)
    app_counts: Counter[str] = Counter()
    selected: list[dict] = []
    for folder in FOLDERS:
        folder_count = 0
        for candidate in by_folder[folder]:
            if folder_count == quota:
                break
            app_key = candidate["app"].casefold()
            if app_counts[app_key] >= max_per_app:
                continue
            selected.append(candidate)
            app_counts[app_key] += 1
            folder_count += 1
        if folder_count != quota:
            raise ValueError(f"not enough eligible {folder} notifications for quota {quota}")

    if len({row["source_id"] for row in selected}) != len(selected):
        raise ValueError("selected source IDs are not unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for candidate in selected:
            stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    report = {
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_count": source_count,
        "eligible_unique_count": len(unique),
        "rejected_counts": dict(rejection_counts),
        "selected_count": len(selected),
        "selected_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "selected_folder_counts": dict(Counter(row["source_folder"] for row in selected)),
        "selected_priority_counts": dict(sorted(Counter(row["source_priority"] for row in selected).items())),
        "distinct_apps": len(app_counts),
        "max_app_count": max(app_counts.values()),
        "review_status": "external text pool only; no ALTONG labels or contexts",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--total", type=int, default=3000)
    parser.add_argument("--max-per-app", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(select_candidates(args.source, args.output, args.total,
                                       args.max_per_app), ensure_ascii=False))


if __name__ == "__main__":
    main()
