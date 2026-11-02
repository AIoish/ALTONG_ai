"""Evaluate GGUF through a local llama-server with the exact HF prompt tokens.

No JSON grammar, summary repair, or reference answers are sent to the model.
Uses the same scoring and category postprocessing as the Transformers evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import evaluate
from .smoke_test_model import render_generation_prompt


def validate_server_url(url: str) -> str:
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("GGUF validation requires a local loopback HTTP server")
    return url.rstrip("/")


def request_json(url: str, payload=None):
    request = Request(url, data=None if payload is None else json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


class GGUFGenerator:
    def __init__(self, server_url: str):
        self.server_url = validate_server_url(server_url)

    def __call__(self, *, tokenizer, model, group, max_summary_lines=3,
                 seed=42, do_sample=False, prompt_style="runtime", task="briefing"):
        if do_sample:
            raise ValueError("GGUF comparison requires --greedy")
        prompt = render_generation_prompt(tokenizer, group, max_summary_lines=max_summary_lines,
                                          prompt_style=prompt_style, task=task)
        return self.complete(tokenizer, prompt, max_new_tokens=256 if task == "briefing" else 160, seed=seed)

    def complete(self, tokenizer, prompt, *, max_new_tokens=256, seed=42):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        native_tokens = request_json(self.server_url + "/tokenize", {
            "content": prompt, "add_special": False, "parse_special": True,
        })["tokens"]
        if tokens != native_tokens:
            raise ValueError("HF and GGUF tokenization differ; comparison would not be valid")
        started = perf_counter()
        response = request_json(self.server_url + "/completion", {
            "prompt": tokens, "n_predict": max_new_tokens,
            "temperature": 0.0, "samplers": ["temperature"], "seed": seed,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "cache_prompt": False, "stream": False,
        })
        content = response.get("content")
        if not isinstance(content, str):
            raise ValueError("llama-server did not return completion content")
        return content.strip(), perf_counter() - started


class GGUFQwenBackend:
    """Diagnostic adapter for the existing Python pipeline, not a C# integration."""

    def __init__(self, *, gguf_path, tokenizer_path, server_url="http://127.0.0.1:18089"):
        self.generator = GGUFGenerator(server_url)
        gguf_path = Path(gguf_path).resolve()
        if not gguf_path.is_file():
            raise FileNotFoundError(gguf_path)
        props = request_json(self.generator.server_url + "/props")
        if Path(props.get("model_path", "")).resolve() != gguf_path:
            raise ValueError("llama-server is not running the requested GGUF model")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(list(messages), tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
        return self.generator.complete(self.tokenizer, prompt)[0]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gguf-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18089")
    args, scoring_args = parser.parse_known_args()
    if not args.gguf_path.is_file():
        raise FileNotFoundError(args.gguf_path)
    server_url = validate_server_url(args.server_url)
    props = request_json(server_url + "/props")
    if Path(props.get("model_path", "")).resolve() != args.gguf_path.resolve():
        raise ValueError("llama-server is not running the requested GGUF model")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    with args.gguf_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {"backend": "llama.cpp /completion", "gguf_path": str(args.gguf_path),
                "gguf_sha256": digest, "tokenizer_path": str(args.tokenizer_path),
                "prompt_tokens_verified": True, "thinking_enabled": False,
                "sampling": False, "json_grammar": False, "cache_prompt": False,
                "latency_scope": "local HTTP completion including prompt processing; excludes loading/token verification"}
    if "--greedy" not in scoring_args:
        scoring_args.append("--greedy")
    sys.argv = [sys.argv[0]] + scoring_args
    evaluate.main(model_loader=lambda **kwargs: (tokenizer, None),
                  summary_generator=GGUFGenerator(server_url), runtime_metadata=metadata)


if __name__ == "__main__":
    main()
