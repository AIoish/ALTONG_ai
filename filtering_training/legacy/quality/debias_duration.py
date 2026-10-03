"""Remove synthetic duration shortcuts without changing filtering labels."""

from filtering_training.common.paths import TRAINING_ROOT, LEGACY_OUTPUTS_ROOT

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

from sklearn.metrics import mutual_info_score

from filtering_training.common.dataset import load_samples


ROOT = TRAINING_ROOT
DEFAULT_SOURCE = LEGACY_OUTPUTS_ROOT / "review/reviewed_5000_v1.jsonl"
DEFAULT_OUTPUT = LEGACY_OUTPUTS_ROOT / "review/reviewed_5000_v2.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _duration_relevance_mi_bits(samples) -> float:
    active = [
        sample for sample in samples
        if sample.context.active_process or sample.context.window_title
    ]
    return mutual_info_score(
        [sample.context.duration_seconds for sample in active],
        [sample.label.relevance_score for sample in active],
    ) / math.log(2)


def debias_duration(source: Path, output: Path, seed: int = 42) -> dict:
    samples = load_samples(source)
    before_mi = _duration_relevance_mi_bits(samples)
    active = [
        sample for sample in samples
        if sample.context.active_process or sample.context.window_title
    ]
    durations = [sample.context.duration_seconds for sample in active]
    short = [duration for duration in durations if duration <= 120]
    long = [duration for duration in durations if duration > 120]
    rng = random.Random(seed)
    rng.shuffle(short)
    rng.shuffle(long)
    needs_recent_switch = [
        sample for sample in active
        if set(sample.context.recent_processes) - {sample.context.active_process}
    ]
    switch_ids = {sample.notification.id for sample in needs_recent_switch}
    single_app = [sample for sample in active if sample.notification.id not in switch_ids]
    if len(short) < len(needs_recent_switch):
        raise ValueError("not enough short durations for recent app switches")
    assignments = {
        sample.notification.id: duration
        for sample, duration in zip(needs_recent_switch, short[:len(needs_recent_switch)])
    }
    single_durations = short[len(needs_recent_switch):] + long
    rng.shuffle(single_durations)
    assignments.update({
        sample.notification.id: duration
        for sample, duration in zip(single_app, single_durations)
    })
    changed = 0
    for sample in active:
        duration = assignments[sample.notification.id]
        changed += sample.context.duration_seconds != duration
        sample.context.duration_seconds = duration
    after_mi = _duration_relevance_mi_bits(samples)
    if after_mi >= before_mi:
        raise ValueError("duration shuffle did not reduce relevance shortcut")
    if any(
        sample.context.duration_seconds > 120
        and set(sample.context.recent_processes) - {sample.context.active_process}
        for sample in active
    ):
        raise ValueError("duration conflicts with recent app switches")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
    shutil.copyfile(source.with_suffix(".lineage.json"), output.with_suffix(".lineage.json"))
    report = {
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(output),
        "seed": seed,
        "count": len(samples),
        "active_context_count": len(active),
        "changed_duration_count": changed,
        "recent_app_duration_conflicts_after": 0,
        "duration_values_preserved": dict(sorted(Counter(durations).items())),
        "duration_relevance_mi_bits_before": before_mi,
        "duration_relevance_mi_bits_after": after_mi,
        "other_fields_unchanged": True,
        "status": "provisional; duration values redistributed among active contexts only",
    }
    output.with_suffix(".duration_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(debias_duration(args.source, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
