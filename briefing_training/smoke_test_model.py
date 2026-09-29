"""Run one synthetic group through the local Qwen summary prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from .prompts import (
    MAX_SUMMARY_LINES,
    MODEL_NAME,
    build_messages,
    parse_summary_response,
)


DEFAULT_CASES_PATH = Path(__file__).with_name("data") / "evaluation_cases.jsonl"


def load_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"case at line {line_number} must be an object")
            cases.append(case)
    if not cases:
        raise ValueError("evaluation data must contain at least one case")
    return cases


def attach_adapter(model: Any, adapter_path: str | Path):
    """Attach a trained PEFT adapter to an already loaded base model."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_path))


def load_model(
    model_name: str = MODEL_NAME,
    *,
    adapter_path: str | Path | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = attach_adapter(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_summary(
    *,
    tokenizer: Any,
    model: Any,
    group: Mapping[str, Any],
    max_summary_lines: int = MAX_SUMMARY_LINES,
    seed: int = 42,
) -> tuple[str, float]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = tokenizer.apply_chat_template(
        build_messages(group, max_summary_lines=max_summary_lines),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    started_at = perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed_seconds = perf_counter() - started_at

    output_ids = generated_ids[0][model_inputs["input_ids"].shape[-1] :]
    raw_response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return raw_response, elapsed_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="local path to a trained PEFT/LoRA adapter",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if not 0 <= args.case_index < len(cases):
        raise SystemExit(f"case index must be between 0 and {len(cases) - 1}")

    case = cases[args.case_index]
    group = case.get("input")
    if not isinstance(group, Mapping):
        raise SystemExit("selected case does not contain an input object")
    max_summary_lines = case.get("max_summary_lines", MAX_SUMMARY_LINES)
    if (
        isinstance(max_summary_lines, bool)
        or not isinstance(max_summary_lines, int)
        or not 1 <= max_summary_lines <= MAX_SUMMARY_LINES
    ):
        raise SystemExit("selected case has an invalid max_summary_lines value")

    print(f"Model: {MODEL_NAME}")
    print(f"Case: {case.get('case_id', args.case_index)}")
    if args.adapter_path is not None:
        print(f"Adapter: {args.adapter_path}")
    print("Loading tokenizer and model...")
    tokenizer, model = load_model(adapter_path=args.adapter_path)
    raw_response, elapsed_seconds = generate_summary(
        tokenizer=tokenizer,
        model=model,
        group=group,
        max_summary_lines=max_summary_lines,
    )

    print("\n=== Raw response ===")
    print(raw_response)
    print(f"\nLatency: {elapsed_seconds:.2f}s")

    try:
        parsed = parse_summary_response(raw_response)
    except ValueError as exc:
        raise SystemExit(f"Invalid structured response: {exc}") from exc

    print("\n=== Parsed summary ===")
    print(json.dumps({"summary_lines": list(parsed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
