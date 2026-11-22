"""Run a batch of ALTONG briefing groups on one local Gemma GGUF server."""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import subprocess
from copy import deepcopy
from pathlib import Path
from time import monotonic, perf_counter, sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.briefing.category_prompt import MAX_NEW_TOKENS, build_messages, parse_response


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "outputs/gemma4-e2b/gemma-4-E2B-it-Q4_0.gguf"
DEFAULT_SYSTEM_PROMPT = ROOT / "briefing_training/experiments/2026-11-15/experimental-system-prompt.txt"
SCHEMA_PATH = Path(__file__).with_name("briefing_response_schema.json")
CONTEXT_SIZE = 4096


def run_batch(args: argparse.Namespace) -> None:
    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("The input JSONL must contain at least one briefing group.")
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    model = args.gguf_path.resolve(strict=True)
    executable = args.llama_server.resolve(strict=True)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    key = secrets.token_urlsafe(24)

    def request(endpoint: str, payload=None, *, timeout: int = 180):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{port}{endpoint}",
            data=data,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".server.log")
    command = [
        str(executable), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--api-key", key, "-c", str(CONTEXT_SIZE), "-ngl", "0", "-np", "1",
        "-t", str(args.threads), "--reasoning", "off", "--no-mmproj",
    ]
    with args.output.open("x", encoding="utf-8") as output, log_path.open("wb") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = monotonic() + 120
            while monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"llama-server exited. See {log_path}")
                try:
                    if request("/health", timeout=2).get("status") == "ok":
                        break
                except (URLError, TimeoutError):
                    pass
                sleep(1)
            else:
                raise TimeoutError(f"llama-server did not become ready. See {log_path}")

            def render_tokens(messages, *, stop_before=None):
                prompt = request("/apply-template", {
                    "messages": messages, "chat_template_kwargs": {"enable_thinking": False},
                })["prompt"]
                if "<|turn>model" not in prompt:
                    raise ValueError("The server did not render a Gemma chat template.")
                if stop_before is not None:
                    prompt, separator, _ = prompt.partition(stop_before)
                    if not separator:
                        raise ValueError("The Gemma template dropped the warmup marker.")
                return request("/tokenize", {
                    "content": prompt, "add_special": False, "parse_special": True,
                })["tokens"]

            warmup_seconds = None
            if args.warm_system_prompt:
                marker = "ALTONG_WARMUP_PLACEHOLDER"
                prefix = render_tokens([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": marker},
                ], stop_before=marker)
                if not 0 < len(prefix) < CONTEXT_SIZE:
                    raise ValueError("The system prompt exceeds the warmup context budget.")
                started = perf_counter()
                request("/completion", {
                    "prompt": prefix, "n_predict": 1, "temperature": 0.0,
                    "cache_prompt": True, "stream": False,
                })
                warmup_seconds = round(perf_counter() - started, 2)
                print(f"System prompt warmup: {warmup_seconds:.2f}s", flush=True)

            for index, case in enumerate(cases):
                limit = case["max_summary_lines"]
                messages = build_messages(case["input"], max_summary_lines=limit)
                messages[0]["content"] = system_prompt
                tokens = render_tokens(messages)
                if len(tokens) + MAX_NEW_TOKENS > CONTEXT_SIZE:
                    raise ValueError(f"{case['case_id']}: input exceeds the context budget.")
                case_schema = deepcopy(schema)
                case_schema["properties"]["summary_lines"]["maxItems"] = limit
                started = perf_counter()
                response = request("/completion", {
                    "prompt": tokens, "n_predict": MAX_NEW_TOKENS,
                    "temperature": 0.0, "samplers": ["temperature"], "seed": 42,
                    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                    "cache_prompt": not args.no_cache, "stream": False, "json_schema": case_schema,
                })
                elapsed = round(perf_counter() - started, 2)
                raw = response["content"]
                try:
                    valid = len(parse_response(raw.strip()).summary_lines) <= limit
                except (ValueError, TypeError):
                    valid = False
                row = dict(
                    case_id=case["case_id"], model=model.name, raw=raw, valid=valid,
                    seconds=elapsed, cache_prompt=not args.no_cache, timings=response.get("timings", {}),
                )
                if index == 0 and warmup_seconds is not None:
                    row["warmup_seconds"] = warmup_seconds
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{row['case_id']}: {elapsed:.2f}s, valid={valid}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warm-system-prompt", action="store_true",
                        help="Precompute the shared prompt on the same server before the first group")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable prompt caching for a baseline run")
    args = parser.parse_args()
    if args.no_cache and args.warm_system_prompt:
        parser.error("--warm-system-prompt requires prompt caching")
    run_batch(args)


if __name__ == "__main__":
    main()
