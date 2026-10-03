"""Create a reproducible first-pass human review sheet for filtering labels."""

from filtering_training.common.paths import TRAINING_ROOT, LEGACY_OUTPUTS_ROOT

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from filtering_training.quality.audit_dataset import audit_samples
from filtering_training.common.dataset import load_samples
from src.filtering.policy import decision_label
from src.filtering.prompt import CATEGORIES


ROOT = TRAINING_ROOT
DEFAULT_DATASET = LEGACY_OUTPUTS_ROOT / "candidates/combined_korean_5000.jsonl"
DEFAULT_MANIFEST = LEGACY_OUTPUTS_ROOT / "prepared_targeted_5000/manifest.json"
DEFAULT_OUTPUT = LEGACY_OUTPUTS_ROOT / "review/first_pass_5000"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_csv(value: object) -> str:
    """Keep notification text from becoming a spreadsheet formula."""
    text = str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def _rank(sample) -> tuple:
    label = sample.label
    boundary = label.urgency_score in (3, 4)
    uncertain_relevance = label.relevance_score in (3, 4)
    missing_context = not sample.context.active_process and not sample.context.window_title
    return (
        int(not boundary),
        int(not uncertain_relevance),
        int(missing_context),
        sample.notification.id,
    )


def select_review_cases(samples, lineage, split_by_id):
    by_id = {sample.notification.id: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("duplicate sample ID")
    entries = lineage["items"]
    if len(entries) != len(samples):
        raise ValueError("lineage count does not match dataset")
    entry_by_id = {entry["id"]: entry for entry in entries}
    if len(entry_by_id) != len(entries) or set(entry_by_id) != set(by_id):
        raise ValueError("lineage IDs do not match dataset")
    if set(split_by_id) != set(by_id):
        raise ValueError("manifest IDs do not match dataset")

    families = defaultdict(list)
    for identifier, entry in entry_by_id.items():
        family = entry["scenario"]
        if not family.startswith(by_id[identifier].label.category + ":"):
            raise ValueError(f"lineage category mismatch: {identifier}")
        families[family].append(by_id[identifier])

    selected = []
    for family in sorted(families):
        members = families[family]
        if len({split_by_id[s.notification.id] for s in members}) != 1:
            raise ValueError(f"family crosses splits: {family}")
        pairs = defaultdict(list)
        for sample in members:
            pair_id = entry_by_id[sample.notification.id].get("pair_id")
            if pair_id is not None:
                pairs[pair_id].append(sample)
        if not pairs:
            selected.append((family, "family representative", "", min(members, key=_rank)))
            continue
        candidates = []
        for pair_id, pair in pairs.items():
            if len(pair) != 2:
                raise ValueError(f"context pair must contain two rows: {pair_id}")
            left, right = pair
            if (left.notification.title, left.notification.body) != (right.notification.title, right.notification.body):
                raise ValueError(f"context pair has different notification text: {pair_id}")
            if left.label.urgency_score != right.label.urgency_score or left.label.category != right.label.category:
                raise ValueError(f"context pair has conflicting urgency or category: {pair_id}")
            if left.label.relevance_score == right.label.relevance_score:
                raise ValueError(f"context pair has unchanged relevance: {pair_id}")
            decisions_differ = decision_label(left.label.urgency_score, left.label.relevance_score) != decision_label(right.label.urgency_score, right.label.relevance_score)
            rank = (
                int(not decisions_differ),
                *_rank(left)[:1],
                -abs(left.label.relevance_score - right.label.relevance_score),
                pair_id,
            )
            candidates.append((rank, pair_id, pair))
        _, pair_id, pair = min(candidates)
        for sample in sorted(pair, key=lambda s: s.notification.id):
            selected.append((family, "context pair", pair_id, sample))
    return selected


def prepare_review_sheet(dataset: Path, manifest_path: Path, output_dir: Path) -> dict:
    samples = load_samples(dataset)
    lineage_path = dataset.with_suffix(".lineage.json")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != _sha256(dataset) or manifest["lineage_sha256"] != _sha256(lineage_path):
        raise ValueError("manifest hashes do not match the source dataset and lineage")
    split_by_id = {}
    for split, info in manifest["splits"].items():
        for identifier in info["notification_ids"]:
            if identifier in split_by_id:
                raise ValueError(f"sample crosses splits: {identifier}")
            split_by_id[identifier] = split
    selected = select_review_cases(samples, lineage, split_by_id)
    audit = audit_samples(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "review_group", "selection_reason", "split", "scenario_family", "pair_id", "sample_id",
        "app_name", "sender", "title", "body", "timestamp", "active_process",
        "window_title", "duration_seconds", "recent_processes", "urgency_score",
        "relevance_score", "category", "ai_summary_reason", "policy_decision",
        "review_status", "corrected_urgency", "corrected_relevance", "corrected_category",
        "corrected_reason", "review_notes",
    ]
    sheet = output_dir / "review_sheet.csv"
    group_by_family = {family: index for index, family in enumerate(sorted({item[0] for item in selected}), 1)}
    with sheet.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for family, selection_reason, pair_id, sample in selected:
            notification, context, label = sample.notification, sample.context, sample.label
            row = {
                "review_group": group_by_family[family],
                "selection_reason": selection_reason,
                "split": split_by_id[notification.id],
                "scenario_family": family,
                "pair_id": pair_id,
                "sample_id": notification.id,
                "app_name": notification.app_name,
                "sender": notification.sender,
                "title": notification.title,
                "body": notification.body,
                "timestamp": notification.timestamp.isoformat(),
                "active_process": context.active_process,
                "window_title": context.window_title,
                "duration_seconds": context.duration_seconds,
                "recent_processes": ", ".join(context.recent_processes),
                "urgency_score": label.urgency_score,
                "relevance_score": label.relevance_score,
                "category": label.category,
                "ai_summary_reason": label.ai_summary_reason,
                "policy_decision": decision_label(label.urgency_score, label.relevance_score),
            }
            writer.writerow({key: _safe_csv(row.get(key, "")) for key in columns})
    report = {
        "source_dataset": str(dataset),
        "source_sha256": _sha256(dataset),
        "lineage_sha256": _sha256(lineage_path),
        "manifest_sha256": _sha256(manifest_path),
        "total_rows": len(samples),
        "total_families": len({entry["scenario"] for entry in lineage["items"]}),
        "review_rows": len(selected),
        "review_families": len({family for family, _, _, _ in selected}),
        "review_pairs": len({pair_id for _, _, pair_id, _ in selected if pair_id}),
        "review_categories": dict(Counter(sample.label.category for _, _, _, sample in selected)),
        "review_splits": dict(Counter(split_by_id[sample.notification.id] for _, _, _, sample in selected)),
        "review_policy": dict(Counter(decision_label(sample.label.urgency_score, sample.label.relevance_score) for _, _, _, sample in selected)),
        "structural_audit": audit,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = prepare_review_sheet(args.dataset, args.manifest, args.output_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "structural_audit"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
