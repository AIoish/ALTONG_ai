"""Run one sample through the base Qwen model and validate its JSON response."""

import argparse
import json
from pathlib import Path

from src.filtering.policy import should_pass
from src.filtering.prompt import build_messages, parse_model_output
from src.filtering.schema import FilteringSample


DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "sample_notifications.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")

    with args.dataset.open(encoding="utf-8") as source:
        lines = [line for line in source if line.strip()]
    if args.sample_index >= len(lines):
        parser.error(f"--sample-index must be less than {len(lines)}")
    sample = FilteringSample.model_validate_json(lines[args.sample_index])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    inputs = tokenizer.apply_chat_template(
        build_messages(sample.notification, sample.context),
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    try:
        label = parse_model_output(output)
    except ValueError as error:
        print(f"Invalid model output: {error}")
        raise SystemExit(1) from error

    result = {
        "notification_id": sample.notification.id,
        "is_passed": should_pass(label.urgency_score, label.relevance_score),
        **label.model_dump(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
