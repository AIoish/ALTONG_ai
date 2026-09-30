"""Run filtering inference on a sample, JSON file, or newline-delimited JSON stream."""

import argparse
import json
import sys
from pathlib import Path

from src.filtering.policy import should_pass
from src.filtering.prompt import build_messages, parse_model_output
from src.filtering.schema import CurrentContext, FilteringSample, RawNotification


DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "sample_notifications.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


def parse_input(payload: dict) -> tuple[RawNotification, CurrentContext]:
    return (RawNotification.model_validate(payload["notification"]),
            CurrentContext.model_validate(payload["context"]))


def load_model(model_name: str, adapter: Path | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto"
    )
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def predict(notification: RawNotification, context: CurrentContext, tokenizer, model) -> dict:
    import torch
    inputs = tokenizer.apply_chat_template(
        build_messages(notification, context),
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
        generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    label = parse_model_output(output)
    return {"notification_id": notification.id,
            "is_passed": should_pass(label.urgency_score, label.relevance_score),
            **label.model_dump()}


def run_stream(tokenizer, model) -> None:
    """One JSON input line yields one JSON result line; the model remains loaded."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            notification, context = parse_input(payload)
            result = predict(notification, context, tokenizer, model)
        except (ValueError, KeyError, TypeError):
            result = {"error": "invalid_input_or_model_output"}
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--input-json", type=Path,
                        help="JSON object containing notification and context; no gold label needed")
    parser.add_argument("--stream", action="store_true",
                        help="keep the model loaded and read one notification/context JSON per stdin line")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.stream and args.input_json is not None:
        parser.error("--stream and --input-json cannot be combined")
    if args.stream:
        tokenizer, model = load_model(args.model, args.adapter)
        run_stream(tokenizer, model)
        return
    if args.input_json is not None:
        payload = json.loads(args.input_json.read_text(encoding="utf-8-sig"))
        notification, context = parse_input(payload)
    else:
        with args.dataset.open(encoding="utf-8") as source:
            lines = [line for line in source if line.strip()]
        if args.sample_index >= len(lines):
            parser.error(f"--sample-index must be less than {len(lines)}")
        sample = FilteringSample.model_validate_json(lines[args.sample_index])
        notification, context = sample.notification, sample.context
    tokenizer, model = load_model(args.model, args.adapter)
    try:
        result = predict(notification, context, tokenizer, model)
    except ValueError as error:
        print(f"Invalid model output: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
