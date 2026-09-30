"""Summarize filtering label coverage and detect contradictory synthetic samples."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from filtering_training import TRAINING_ROOT

from filtering_training.datasets.prepare_dataset import DATASET_PATH, load_samples
from src.filtering.policy import should_pass
from src.filtering.prompt import CATEGORIES
from src.filtering.schema import FilteringSample


DEFAULT_REPORT = TRAINING_ROOT / "outputs" / "audit" / "dataset_audit.json"


def _notification_key(sample: FilteringSample) -> tuple[str, ...]:
    notification = sample.notification
    return tuple(" ".join(value.split()).casefold() for value in (
        notification.app_name, notification.sender, notification.title, notification.body
    ))


def audit_samples(samples: list[FilteringSample], minimum_per_category: int = 5) -> dict:
    if not samples:
        raise ValueError("dataset contains no samples")
    category_counts = Counter(sample.label.category for sample in samples)
    urgency_counts = Counter(sample.label.urgency_score for sample in samples)
    relevance_counts = Counter(sample.label.relevance_score for sample in samples)
    recent_counts = Counter(len(sample.context.recent_processes) for sample in samples)
    grouped = defaultdict(list)
    exact_inputs = defaultdict(list)
    for sample in samples:
        grouped[_notification_key(sample)].append(sample)
        exact_key = (
            _notification_key(sample), sample.context.active_process.casefold(),
            sample.context.window_title.casefold(), sample.context.duration_seconds,
            tuple(name.casefold() for name in sample.context.recent_processes),
        )
        exact_inputs[exact_key].append(sample)

    contradictions = []
    for group in grouped.values():
        if len(group) > 1 and (len({sample.label.urgency_score for sample in group}) > 1
                               or len({sample.label.category for sample in group}) > 1):
            contradictions.append([sample.notification.id for sample in group])
    conflicting_inputs = []
    for group in exact_inputs.values():
        if len(group) > 1 and len({sample.label.model_dump_json() for sample in group}) > 1:
            conflicting_inputs.append([sample.notification.id for sample in group])
    report = {
        "count": len(samples),
        "category_counts": {name: category_counts[name] for name in CATEGORIES},
        "urgency_counts": {str(score): urgency_counts[score] for score in range(1, 6)},
        "relevance_counts": {str(score): relevance_counts[score] for score in range(1, 6)},
        "policy_counts": {
            "pass": sum(should_pass(sample.label.urgency_score, sample.label.relevance_score)
                        for sample in samples),
            "block": sum(not should_pass(sample.label.urgency_score, sample.label.relevance_score)
                         for sample in samples),
        },
        "recent_process_count": {str(count): recent_counts[count] for count in range(4)},
        "duration_zero_count": sum(sample.context.duration_seconds == 0 for sample in samples),
        "empty_context_count": sum(
            not sample.context.active_process and not sample.context.window_title
            for sample in samples
        ),
        "notification_variant_groups": sum(len(group) > 1 for group in grouped.values()),
        "contradictory_notification_groups": contradictions,
        "conflicting_identical_inputs": conflicting_inputs,
        "categories_below_minimum": [
            name for name in CATEGORIES if category_counts[name] < minimum_per_category
        ],
        "minimum_per_category": minimum_per_category,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--minimum-per-category", type=int, default=5)
    args = parser.parse_args()
    if args.minimum_per_category < 1:
        parser.error("--minimum-per-category must be positive")
    report = audit_samples(load_samples(args.dataset), args.minimum_per_category)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
