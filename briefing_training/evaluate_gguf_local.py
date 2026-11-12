"""Evaluate a GGUF with a local llama-completion process (no network server)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from . import evaluate
from .smoke_test_model import render_generation_prompt


def build_llama_command(executable: Path, model: Path, prompt: Path, threads: int,
                        schema: Path | None) -> list[str]:
    command = [str(executable), "-m", str(model), "-f", os.path.relpath(prompt),
               "-n", "256", "--temp", "0", "-ngl", "0", "-c", "4096", "-t", str(threads),
               "-no-cnv", "--no-display-prompt", "--simple-io"]
    if schema is not None:
        command.extend(["--json-schema-file", str(schema)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--llama-completion", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--json-schema-file", type=Path,
                        help="optional JSON schema for constrained generation")
    args = parser.parse_args()
    for path in (args.gguf_path, args.llama_completion, args.cases):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if args.json_schema_file is not None and not args.json_schema_file.is_file():
        parser.error(f"JSON schema file not found: {args.json_schema_file}")
    for path in (args.output, args.review_output):
        if path.exists():
            parser.error(f"output already exists: {path}; choose a new path")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    case_count = len(evaluate.load_cases(args.cases))
    current = 0

    def generate(**options):
        nonlocal current
        current += 1
        prompt = render_generation_prompt(
            tokenizer, options["group"], max_summary_lines=options["max_summary_lines"],
            task="briefing", prompt_style="runtime",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", dir=args.output.parent, delete=False) as tmp:
            tmp.write(prompt)
            prompt_path = Path(tmp.name)
        try:
            cmd = build_llama_command(args.llama_completion, args.gguf_path, prompt_path,
                                      args.threads, args.json_schema_file)
            print(f"Evaluating case {current}/{case_count}...", file=sys.stderr, flush=True)
            start = time.perf_counter()
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode:
                raise RuntimeError(f"llama-completion failed ({result.returncode}): "
                                   + result.stderr.decode("utf-8", errors="replace")[-3000:])
            elapsed = time.perf_counter() - start
            text = result.stdout.decode("utf-8", errors="replace").strip()
            # Strip only llama-completion's end marker; leave malformed output
            # intact so the evaluator reports invalid JSON rather than repairing it.
            text = text.removesuffix("[end of text]").strip()
            print(f"Finished case {current}/{case_count} in {elapsed:.1f}s", file=sys.stderr, flush=True)
            return text, elapsed
        finally:
            prompt_path.unlink(missing_ok=True)

    # Avoid ambiguous provenance: the adapter was merged into this GGUF.
    sys.argv = ["evaluate", "--task", "briefing", "--greedy", "--model-name", "Qwen/Qwen3-1.7B",
                "--cases", str(args.cases), "--output", str(args.output),
                "--review-output", str(args.review_output)]
    evaluate.main(model_loader=lambda **_: (tokenizer, None), summary_generator=generate,
                  runtime_metadata={"runtime": "llama-completion CPU", "gguf_path": str(args.gguf_path),
                                    "latency_includes_model_load": True, "threads": args.threads,
                                    "json_schema_file": str(args.json_schema_file) if args.json_schema_file else None})


if __name__ == "__main__":
    main()
