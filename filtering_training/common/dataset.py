"""Shared validated dataset loading and SFT conversion."""

import json
from filtering_training.common.validation_data import load_samples
from src.filtering.prompt import build_messages, parse_model_output
from src.filtering.schema import FilteringSample



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
