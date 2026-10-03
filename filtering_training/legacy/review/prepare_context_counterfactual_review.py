"""Create a small counterfactual relevance review batch with reused windows."""

from filtering_training.common.paths import LEGACY_OUTPUTS_ROOT

import argparse
import csv
from pathlib import Path

from filtering_training.modeling.evaluate import load_split_samples
from src.filtering.prompt import CATEGORIES


DEFAULT_DATASET = LEGACY_OUTPUTS_ROOT / "review/reviewed_5000_v2.jsonl"
DEFAULT_PREPARED = LEGACY_OUTPUTS_ROOT / "review/prepared_reviewed_5000_v2"
DEFAULT_OUTPUT = LEGACY_OUTPUTS_ROOT / "review/context_counterfactual_batch_01.csv"


def build_rows(dataset: Path, prepared: Path):
    samples = load_split_samples(dataset, prepared, "train")
    by_category = {category: [] for category in CATEGORIES}
    for sample in samples:
        by_category[sample.label.category].append(sample)
    rows = []
    for index, category in enumerate(CATEGORIES):
        matched = next(
            sample for sample in by_category[category]
            if sample.label.relevance_score >= 4 and sample.context.window_title
        )
        donor_category = CATEGORIES[(index + 4) % len(CATEGORIES)]
        donor = next(
            sample for sample in by_category[donor_category]
            if sample.notification.title != matched.notification.title
        )
        for role, notification_sample, proposed in (
            ("original_match", matched, matched.label.relevance_score),
            ("swapped_notification", donor, 1),
        ):
            n = notification_sample.notification
            c = matched.context
            rows.append({
                "pair_id": f"context_{index + 1:02d}",
                "role": role,
                "notification_source_id": n.id,
                "context_source_id": matched.notification.id,
                "category": notification_sample.label.category,
                "urgency_score": notification_sample.label.urgency_score,
                "app_name": n.app_name,
                "title": n.title,
                "body": n.body,
                "active_process": c.active_process,
                "window_title": c.window_title,
                "recent_processes": ", ".join(c.recent_processes),
                "duration_seconds": c.duration_seconds,
                "proposed_relevance": proposed,
                "reviewed_relevance": "",
                "review_note": "",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = build_rows(args.dataset, args.prepared_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows in {args.output}")


if __name__ == "__main__":
    main()
