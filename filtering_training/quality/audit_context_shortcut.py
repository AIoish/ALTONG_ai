"""Audit whether a context window alone reveals relevance across data splits."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from filtering_training.modeling.evaluate import load_split_samples


def audit(dataset: Path, prepared_dir: Path):
    train = load_split_samples(dataset, prepared_dir, "train")
    validation = load_split_samples(dataset, prepared_dir, "validation")
    by_context = defaultdict(set)
    for sample in train:
        context = sample.context
        by_context[(context.active_process, context.window_title)].add(
            sample.label.relevance_score
        )
    seen = [
        sample for sample in validation
        if (sample.context.active_process, sample.context.window_title) in by_context
    ]
    deterministic = [
        sample for sample in seen
        if len(by_context[(sample.context.active_process, sample.context.window_title)]) == 1
    ]
    correct = [
        sample for sample in deterministic
        if sample.label.relevance_score in
        by_context[(sample.context.active_process, sample.context.window_title)]
    ]
    return {
        "train_count": len(train),
        "validation_count": len(validation),
        "train_context_keys": len(by_context),
        "validation_seen_context_rows": len(seen),
        "validation_deterministic_context_rows": len(deterministic),
        "validation_context_only_correct_rows": len(correct),
        "validation_context_only_coverage": len(correct) / len(validation),
        "key": ["active_process", "window_title"],
        "interpretation": "A repeated window can reveal relevance without comparing notification meaning.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.dataset, args.prepared_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
