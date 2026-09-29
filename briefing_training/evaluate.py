"""Evaluate structured output and source-fact coverage on synthetic cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .prompts import MODEL_NAME, parse_summary_response
from .smoke_test_model import (
    DEFAULT_CASES_PATH,
    generate_summary,
    load_cases,
    load_model,
)


def _expected_facts(case: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    facts = case.get("expected_facts", [])
    if not isinstance(facts, list):
        raise ValueError("expected_facts must be an array")

    normalized: list[tuple[str, ...]] = []
    for fact in facts:
        if not isinstance(fact, list) or not all(
            isinstance(alternative, str) and alternative.strip()
            for alternative in fact
        ):
            raise ValueError(
                "each expected fact must be an array of alternative phrases"
            )
        if not fact:
            raise ValueError("each expected fact must contain an alternative")
        normalized.append(tuple(alternative.strip() for alternative in fact))
    return tuple(normalized)


def _forbidden_phrases(case: Mapping[str, Any]) -> tuple[str, ...]:
    phrases = case.get("forbidden_phrases", [])
    if not isinstance(phrases, list) or not all(
        isinstance(phrase, str) and phrase.strip() for phrase in phrases
    ):
        raise ValueError("forbidden_phrases must be an array of non-empty strings")
    return tuple(phrase.strip() for phrase in phrases)


def _max_summary_lines(case: Mapping[str, Any]) -> int:
    value = case.get("max_summary_lines", 3)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3:
        raise ValueError("max_summary_lines must be an integer from 1 to 3")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="local path to a trained PEFT/LoRA adapter",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the JSON evaluation report",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    tokenizer, model = load_model(adapter_path=args.adapter_path)
    structured_count = 0
    fact_hits = 0
    fact_total = 0
    passed_case_count = 0
    latencies: list[float] = []
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        group = case.get("input")
        if not isinstance(group, Mapping):
            raise ValueError(f"case {index} does not contain an input object")

        max_summary_lines = _max_summary_lines(case)
        raw_response, elapsed_seconds = generate_summary(
            tokenizer=tokenizer,
            model=model,
            group=group,
            max_summary_lines=max_summary_lines,
            seed=42 + index,
        )
        latencies.append(elapsed_seconds)
        expected_facts = _expected_facts(case)
        forbidden_phrases = _forbidden_phrases(case)
        fact_total += len(expected_facts)

        try:
            summary_lines = parse_summary_response(raw_response)
            structured_count += 1
            combined = " ".join(summary_lines)
            hits = sum(
                any(alternative in combined for alternative in alternatives)
                for alternatives in expected_facts
            )
            forbidden_hits = tuple(
                phrase for phrase in forbidden_phrases if phrase in combined
            )
            line_limit_passed = len(summary_lines) <= max_summary_lines
            fact_hits += hits
            error = None
        except ValueError as exc:
            summary_lines = ()
            hits = 0
            forbidden_hits = ()
            line_limit_passed = False
            error = str(exc)

        passed = (
            error is None
            and hits == len(expected_facts)
            and not forbidden_hits
            and line_limit_passed
        )
        if passed:
            passed_case_count += 1

        results.append(
            {
                "case_id": case.get("case_id", index),
                "passed": passed,
                "structured_output": error is None,
                "fact_hits": hits,
                "fact_total": len(expected_facts),
                "forbidden_phrase_hits": list(forbidden_hits),
                "max_summary_lines": max_summary_lines,
                "line_limit_passed": line_limit_passed,
                "latency_seconds": round(elapsed_seconds, 2),
                "summary_lines": list(summary_lines),
                "error": error,
                "raw_response": raw_response,
            }
        )

    report = {
        "model": MODEL_NAME,
        "adapter_path": str(args.adapter_path)
        if args.adapter_path is not None
        else None,
        "case_count": len(cases),
        "structured_output_rate": round(structured_count / len(cases), 4),
        "case_pass_rate": round(passed_case_count / len(cases), 4),
        "fact_coverage": round(fact_hits / fact_total, 4)
        if fact_total
        else None,
        "average_latency_seconds": round(sum(latencies) / len(latencies), 2),
        "results": results,
    }
    serialized_report = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized_report + "\n", encoding="utf-8")
    print(serialized_report)


if __name__ == "__main__":
    main()
