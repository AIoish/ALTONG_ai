"""Select a small, balanced first batch from the filtering review sheet."""

from filtering_training.common.paths import LEGACY_OUTPUTS_ROOT

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.filtering.prompt import CATEGORIES


ROOT = LEGACY_OUTPUTS_ROOT / "review/first_pass_5000"


def create_first_batch(source: Path, destination: Path) -> list[dict[str, str]]:
    with source.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("review sheet has no header")
        rows = list(reader)
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    if set(by_category) != set(CATEGORIES):
        raise ValueError("review sheet categories are incomplete")

    selected = []
    for category in CATEGORIES:
        category_rows = by_category[category]
        singles = [row for row in category_rows if row["selection_reason"] == "family representative"]
        pairs = defaultdict(list)
        for row in category_rows:
            if row["pair_id"]:
                pairs[row["pair_id"]].append(row)
        if not singles or not pairs or any(len(pair) != 2 for pair in pairs.values()):
            raise ValueError(f"review cases are incomplete for {category}")

        def pair_rank(item):
            pair_id, pair = item
            return (
                0 if any("가스 점검" in row["title"] for row in pair) else 1,
                0 if len({row["policy_decision"] for row in pair}) == 2 else 1,
                0 if any(not row["active_process"] and not row["window_title"] for row in pair) else 1,
                0 if pair[0]["urgency_score"] in {"3", "4"} else 1,
                pair_id,
            )

        pair_id, pair = min(pairs.items(), key=pair_rank)
        pair_decisions = {row["policy_decision"] for row in pair}
        single = min(
            singles,
            key=lambda row: (
                0 if row["policy_decision"] not in pair_decisions else 1,
                0 if row["urgency_score"] in {"3", "4"} else 1,
                row["sample_id"],
            ),
        )
        selected.append(single)
        selected.extend(sorted(pair, key=lambda row: row["sample_id"]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "review_sheet.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "review_batch_01.csv")
    args = parser.parse_args()
    rows = create_first_batch(args.source, args.output)
    print(f"Created {len(rows)} review rows in {args.output}")


if __name__ == "__main__":
    main()
