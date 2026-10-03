"""Shared validated dataset loading and SFT conversion."""

import json
from pathlib import Path

from filtering_training.common.paths import resolve_existing_path
from src.filtering.prompt import build_messages, parse_model_output
from src.filtering.schema import FilteringSample


def load_samples(path: Path) -> list[FilteringSample]:
    path = resolve_existing_path(path)
    samples = []
    seen_ids = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                sample = FilteringSample.model_validate_json(line)
                target = json.dumps(sample.label.model_dump(), ensure_ascii=False)
                parse_model_output(target)
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid sample at line {line_number}") from error
            if sample.notification.id in seen_ids:
                raise ValueError(f"duplicate notification id at line {line_number}")
            seen_ids.add(sample.notification.id)
            samples.append(sample)
    if not samples:
        raise ValueError("dataset contains no samples")
    return samples

def to_sft_record(sample: FilteringSample) -> dict[str, list[dict[str, str]]]:
    response = json.dumps(
        sample.label.model_dump(), ensure_ascii=False, separators=(",", ":")
    )
    parse_model_output(response)
    return {
        "messages": [
            *build_messages(sample.notification, sample.context),
            {"role": "assistant", "content": response},
        ]
    }
