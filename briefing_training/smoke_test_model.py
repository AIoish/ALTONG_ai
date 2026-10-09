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

from src.briefing.category_prompt import MAX_NEW_TOKENS, build_messages as build_briefing_messages, parse_response


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


def generation_options(*, do_sample: bool, task: str = "summary") -> dict[str, Any]:
    """Explicitly override sampling defaults stored in the base model config."""
    return {
        "max_new_tokens": MAX_NEW_TOKENS if task == "briefing" else 160,
        "do_sample": do_sample,
        "num_beams": 1,
        "temperature": 0.7 if do_sample else None,
        "top_p": 0.8 if do_sample else None,
        "top_k": 20 if do_sample else None,
    }


def render_generation_prompt(
    tokenizer: Any, group: Mapping[str, Any], *,
    max_summary_lines: int = MAX_SUMMARY_LINES, prompt_style: str = "runtime",
    task: str = "summary",
) -> str:
    if task not in {"briefing", "summary"}:
        raise ValueError("task must be briefing or summary")
    builder = build_briefing_messages if task == "briefing" else build_messages
    messages = builder(group, max_summary_lines=max_summary_lines)
    if task == "briefing":
        if prompt_style not in {"runtime", "training"}:
            raise ValueError("prompt_style must be runtime or training")
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    if prompt_style == "training":
        messages[-1]["content"] += "\n/no_think"
        # Match train_lora.render_prompt_completion, including its template defaults.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    if prompt_style != "runtime":
        raise ValueError("prompt_style must be runtime or training")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate_summary(
    *,
    tokenizer: Any,
    model: Any,
    group: Mapping[str, Any],
    max_summary_lines: int = MAX_SUMMARY_LINES,
    seed: int = 42,
    do_sample: bool = True,
    prompt_style: str = "runtime",
    task: str = "summary",
) -> tuple[str, float]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = render_generation_prompt(
        tokenizer, group, max_summary_lines=max_summary_lines, prompt_style=prompt_style, task=task,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    started_at = perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **generation_options(do_sample=do_sample, task=task),
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed_seconds = perf_counter() - started_at

    output_ids = generated_ids[0][model_inputs["input_ids"].shape[-1] :]
    raw_response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return raw_response, elapsed_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--task", choices=("briefing", "summary"), default="summary")
    parser.add_argument("--cases", type=Path)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="local path to a trained PEFT/LoRA adapter",
    )
    args = parser.parse_args()
    args.cases = args.cases or (DEFAULT_CASES_PATH.parent / "category_briefing" / "evaluation_cases.jsonl"
                               if args.task == "briefing" else DEFAULT_CASES_PATH)

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
        task=args.task,
    )

    print("\n=== Raw response ===")
    print(raw_response)
    print(f"\nLatency: {elapsed_seconds:.2f}s")

    try:
        if args.task == "briefing":
            decision = parse_response(raw_response)
            payload = {"primary_category": decision.primary_category, "summary_lines": list(decision.summary_lines)}
        else:
            payload = {"summary_lines": list(parse_summary_response(raw_response))}
    except ValueError as exc:
        raise SystemExit(f"Invalid structured response: {exc}") from exc

    print("\n=== Parsed summary ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
