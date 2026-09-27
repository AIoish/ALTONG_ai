"""Prepare provisional rapid candidates with scenario-family isolation."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from filtering_training.generate_rapid_dataset import OUTPUT_PATH
from filtering_training.prepare_dataset import load_samples, to_sft_record
from src.filtering.prompt import CATEGORIES


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "prepared_rapid"


def prepare(dataset: Path = OUTPUT_PATH, output_dir: Path = OUTPUT_DIR,
            seed: int = 42) -> dict:
    samples = load_samples(dataset)
    lineage_path = dataset.with_suffix(".lineage.json")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    entries = lineage["items"]
    if len(entries) != len(samples):
        raise ValueError("lineage count does not match dataset")
    family_by_id = {}
    for entry in entries:
        identifier, family = entry["id"], entry["scenario"]
        if identifier in family_by_id:
            raise ValueError("duplicate lineage ID")
        family_by_id[identifier] = family
    if set(family_by_id) != {sample.notification.id for sample in samples}:
        raise ValueError("lineage IDs do not match dataset")
    grouped = defaultdict(list)
    category_families = defaultdict(set)
    for sample in samples:
        family = family_by_id[sample.notification.id]
        if not family.startswith(sample.label.category + ":"):
            raise ValueError("lineage category does not match label")
        grouped[family].append(sample)
        category_families[sample.label.category].add(family)
    if set(category_families) != set(CATEGORIES):
        raise ValueError("dataset categories are incomplete")

    assignments = {}
    for category in CATEGORIES:
        families = sorted(category_families[category])
        if len(families) < 3:
            raise ValueError(f"at least three families required for {category}")
        random.Random(f"{seed}:{category}").shuffle(families)
        assignments[families[0]] = "validation"
        assignments[families[1]] = "test"
        assignments.update({family: "train" for family in families[2:]})

    splits = {name: [] for name in ("train", "validation", "test")}
    for sample in samples:
        splits[assignments[family_by_id[sample.notification.id]]].append(sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "lineage_sha256": hashlib.sha256(lineage_path.read_bytes()).hexdigest(),
        "seed": seed,
        "status": "provisional; review wording and labels before model training",
        "split_rule": "one scenario family per category for validation and test; remaining families for training",
        "splits": {},
    }
    for name, items in splits.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as destination:
            for sample in items:
                destination.write(json.dumps(to_sft_record(sample), ensure_ascii=False) + "\n")
        manifest["splits"][name] = {
            "count": len(items),
            "families": sorted({family_by_id[item.notification.id] for item in items}),
            "category_counts": {category: sum(item.label.category == category for item in items)
                                for category in CATEGORIES},
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.dataset, args.output_dir, args.seed)
    print(json.dumps({name: info["count"] for name, info in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
