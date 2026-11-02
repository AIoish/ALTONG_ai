"""Merge a local briefing adapter for subsequent official GGUF conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.briefing.category_prompt import build_messages
from src.briefing.category_correction import correction_manifest
from src.briefing.qwen_prompt import MODEL_NAME


def validate_export(adapter: Path, output: Path) -> str:
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise ValueError(f"missing adapter file: {adapter / name}")
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base = config.get("base_model_name_or_path")
    if base != MODEL_NAME:
        raise ValueError(f"expected {MODEL_NAME}; adapter declares {base}")
    if output.exists():
        raise ValueError("output already exists; choose a new directory (no overwrite)")
    if adapter.resolve() == output.resolve() or adapter.resolve() in output.resolve().parents:
        raise ValueError("output must not be inside the source adapter")
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    base_name = validate_export(args.adapter_path, args.output_dir)
    if args.validate_only:
        print(f"Ready to merge {base_name}; source adapter will not be modified")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Non-quantized CPU merge; not NF4, not a training run. Needs system RAM.
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32,
                                               device_map="cpu", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    adapter = PeftModel.from_pretrained(base, str(args.adapter_path))
    merged = adapter.merge_and_unload(safe_merge=True)
    merged.eval()
    merged = merged.to(dtype=torch.float16)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "category_correction_rules.json").write_text(
        json.dumps(correction_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    # Freeze exactly what the client should send; GGUF stores weights, not the
    # Python pipeline. Explicit rendering avoids enabling Qwen3 thinking.
    group = {"app_name": "KakaoTalk.exe", "sender": "가상 팀장", "notifications": [
        {"id": "example", "title": "프로젝트방", "body": "내일 오후 1시에 회의합니다.",
         "timestamp": "2026-10-10T09:00:00Z"}]}
    messages = build_messages(group)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
    (args.output_dir / "briefing_prompt_example.json").write_text(
        json.dumps({"messages": messages, "rendered_prompt": rendered,
                    "generation": {"do_sample": False, "max_new_tokens": 256},
                    "model_output_fields": ["primary_category", "summary_lines"],
                    "note": "Group notifications first; apply category_correction_rules.json in application code. Schedule extraction and dashboard envelope are also application code."},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved merged Hugging Face model: {args.output_dir}")
    print("This is not GGUF yet. Convert to F16 GGUF, then run llama-quantize Q4_K_M.")


if __name__ == "__main__":
    main()
