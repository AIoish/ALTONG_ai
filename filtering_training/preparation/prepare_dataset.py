"""Validate and convert filtering samples into reproducible SFT JSONL splits."""

from filtering_training.common.paths import TRAINING_ROOT, LEGACY_OUTPUTS_ROOT, resolve_existing_path

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from filtering_training.common.dataset import load_samples, to_sft_record
from src.filtering.schema import FilteringSample


DATASET_PATH = TRAINING_ROOT / "data" / "sample_notifications.jsonl"
OUTPUT_DIR = LEGACY_OUTPUTS_ROOT / "prepared"




def _group_key(sample: FilteringSample) -> tuple[str, ...]:
    """Keep the same notification text in one split when context varies."""
    notification = sample.notification
    return tuple(
        " ".join(value.split()).casefold()
        for value in (
            notification.app_name,
            notification.sender,
            notification.title,
            notification.body,
        )
    )


def split_samples(
    samples: list[FilteringSample], seed: int = 42
) -> dict[str, list[FilteringSample]]:
    groups: dict[tuple[str, ...], list[FilteringSample]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)
    shuffled = list(sorted(groups.items()))
    random.Random(seed).shuffle(shuffled)

    # Small splits check the pipeline; reserve urgent groups to exercise miss metrics.
    target = max(1, round(len(samples) * 0.1))
    splits: dict[str, list[FilteringSample]] = {
        "train": [], "validation": [], "test": []
    }
    urgent_keys = [
        key for key, group in shuffled
        if any(sample.label.urgency_score >= 4 for sample in group)
    ]
    reserved = {}
    if len(urgent_keys) >= 2:
        reserved = {urgent_keys[0]: "test", urgent_keys[1]: "validation"}
    for key, group in shuffled:
        if key in reserved:
            splits[reserved[key]].extend(group)
    for key, group in shuffled:
        if key in reserved:
            continue
        if len(splits["test"]) < target:
            name = "test"
        elif len(splits["validation"]) < target:
            name = "validation"
        else:
            name = "train"
        splits[name].extend(group)
    if not splits["train"]:
        raise ValueError("not enough independent notification groups for training")
    return splits




def prepare_dataset(dataset: Path, output_dir: Path, seed: int = 42) -> dict:
    dataset = resolve_existing_path(dataset)
    samples = load_samples(dataset)
    splits = split_samples(samples, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "seed": seed,
        "purpose": "pipeline smoke test; synthetic examples require broader review before quality claims",
        "splits": {},
    }
    for name, items in splits.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as destination:
            for sample in items:
                destination.write(json.dumps(to_sft_record(sample), ensure_ascii=False))
                destination.write("\n")
        manifest["splits"][name] = {
            "count": len(items),
            "notification_ids": [sample.notification.id for sample in items],
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_dataset(args.dataset, args.output_dir, args.seed)
    print(json.dumps({name: info["count"] for name, info in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
