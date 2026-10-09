"""Validate an independent synthetic holdout and prepare an evaluation-only manifest."""

from filtering_training.common.paths import TRAINING_ROOT, LEGACY_OUTPUTS_ROOT

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from filtering_training.common.dataset import load_samples
from filtering_training.preparation.prepare_dataset import DATASET_PATH
from src.filtering.prompt import CATEGORIES
from src.filtering.schema import FilteringSample


HOLDOUT_PATH = TRAINING_ROOT / "data" / "evaluation_notifications.jsonl"
OUTPUT_DIR = LEGACY_OUTPUTS_ROOT / "holdout"


def notification_key(sample: FilteringSample) -> tuple[str, ...]:
    item = sample.notification
    return tuple(" ".join(value.split()).casefold() for value in
                 (item.app_name, item.sender, item.title, item.body))


def validate_holdout(
    holdout: list[FilteringSample], training: list[FilteringSample],
    minimum_per_category: int = 3,
) -> dict:
    if minimum_per_category < 1:
        raise ValueError("minimum_per_category must be positive")
    holdout_ids = {item.notification.id for item in holdout}
    training_ids = {item.notification.id for item in training}
    if holdout_ids & training_ids:
        raise ValueError("holdout IDs overlap with training data")
    holdout_keys = [notification_key(item) for item in holdout]
    training_keys = {notification_key(item) for item in training}
    if len(holdout_keys) != len(set(holdout_keys)):
        raise ValueError("holdout contains duplicate notification text")
    if set(holdout_keys) & training_keys:
        raise ValueError("holdout notification text overlaps with training data")
    category_counts = Counter(item.label.category for item in holdout)
    below_minimum = [
        name for name in CATEGORIES if category_counts[name] < minimum_per_category
    ]
    if below_minimum:
        raise ValueError(f"holdout category coverage is too low: {below_minimum}")
    return {
        "count": len(holdout),
        "category_counts": {name: category_counts[name] for name in CATEGORIES},
        "distinct_notification_texts": len(holdout_keys),
    }


def prepare_holdout(
    holdout_path: Path = HOLDOUT_PATH,
    training_path: Path = DATASET_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    holdout = load_samples(holdout_path)
    training = load_samples(training_path)
    coverage = validate_holdout(holdout, training)
    manifest = {
        "source_sha256": hashlib.sha256(holdout_path.read_bytes()).hexdigest(),
        "training_source_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "purpose": "provisional synthetic holdout; never use for training",
        "review_status": "pending independent label review",
        **coverage,
        "splits": {
            "test": {
                "count": len(holdout),
                "notification_ids": [item.notification.id for item in holdout],
            }
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_PATH)
    parser.add_argument("--training-data", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    manifest = prepare_holdout(args.holdout, args.training_data, args.output_dir)
    print(json.dumps(
        {"count": manifest["count"], "category_counts": manifest["category_counts"],
         "source_sha256": manifest["source_sha256"]}, ensure_ascii=False
    ))


if __name__ == "__main__":
    main()