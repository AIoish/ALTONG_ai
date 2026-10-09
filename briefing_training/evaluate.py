"""Evaluate structured output and source-fact coverage on synthetic cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.briefing.schema import FILTER_CATEGORIES
from src.briefing.category_prompt import parse_response
from .prepare_category_dataset import EVALUATION_PATH

from .prompts import MODEL_NAME, parse_summary_response_with_metadata
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


def _reference_summary_lines(case: Mapping[str, Any]) -> tuple[str, ...]:
    lines = case.get("reference_summary_lines", [])
    if not isinstance(lines, list) or not lines or not all(
        isinstance(line, str) and line.strip() for line in lines
    ):
        raise ValueError(
            "reference_summary_lines must be a non-empty array of strings"
        )
    normalized = tuple(line.strip() for line in lines)
    if len(normalized) > _expected_max_summary_lines(case):
        raise ValueError("reference_summary_lines exceeds max_summary_lines")
    return normalized


def _expected_max_summary_lines(case: Mapping[str, Any]) -> int:
    """Separate concision assessment from the model-visible three-line ceiling."""
    ceiling = _max_summary_lines(case)
    value = case.get("expected_max_summary_lines", ceiling)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
        raise ValueError("expected_max_summary_lines must be within max_summary_lines")
    return value


def validate_evaluation_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    """Validate the human-reviewed evaluation contract without loading a model."""

    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case {index} must have a non-empty case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)

        group = case.get("input")
        if not isinstance(group, Mapping):
            raise ValueError(f"{case_id} must contain an input object")
        if case.get("expected_category", group.get("category")) not in FILTER_CATEGORIES:
            raise ValueError(f"{case_id} has an invalid category")

        notifications = group.get("notifications")
        if not isinstance(notifications, list) or not notifications:
            raise ValueError(f"{case_id} must contain notifications")
        for notification in notifications:
            if not isinstance(notification, Mapping) or not all(
                isinstance(notification.get(field), str)
                and notification[field].strip()
                for field in ("timestamp", "title", "body")
            ):
                raise ValueError(f"{case_id} contains an invalid notification")

        _expected_facts(case)
        forbidden_phrases = _forbidden_phrases(case)
        _max_summary_lines(case)
        reference_lines = _reference_summary_lines(case)
        reference_text = " ".join(reference_lines)
        if any(
            not any(alternative in reference_text for alternative in alternatives)
            for alternatives in _expected_facts(case)
        ):
            raise ValueError(f"{case_id} reference omits an expected fact")
        if any(phrase in reference_text for phrase in forbidden_phrases):
            raise ValueError(f"{case_id} reference contains a forbidden phrase")


def _render_review_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact report for a human to compare source and model output."""

    lines = [
        "# Qwen 브리핑 평가 검토",
        "",
        "## 전체 결과",
        "",
        f"- 모델: `{report['model']}`",
        f"- 평가 사례: {report['case_count']}개",
        f"- 구조화 출력 성공률: {report['structured_output_rate']}",
        f"- 원본 JSON 계약 준수율: {report['raw_contract_compliance_rate']}",
        f"- 안전 형식 보정률: {report['format_repair_rate']}",
        f"- 사례 통과율: {report['case_pass_rate']}",
        f"- 카테고리 정확도: {report.get('category_accuracy', '분류 미평가')}",
        f"- 사실 정보 포함률: {report['fact_coverage']}",
        "",
        "아래 데이터는 모두 합성 사례입니다. 자동 점수만 보지 말고 입력 알림, "
        "사람 기준 요약, 모델 요약을 직접 비교해 주세요.",
    ]

    for index, result in enumerate(report["results"], start=1):
        status = "PASS" if result["passed"] else "FAIL"
        lines.extend(
            [
                "",
                f"## {index}. {result['case_id']} — {status}",
                "",
                f"- 카테고리: {result['category']}",
                f"- 예측 카테고리: {result.get('predicted_category') or '분류 미평가'}",
                f"- 핵심 사실: {result['fact_hits']}/{result['fact_total']}",
                f"- 줄 수 제한 통과: {result['line_limit_passed']}",
                f"- 출력 형식 자동 보정: {result['format_repaired']}",
                f"- 누락된 핵심 사실: "
                f"{result['missing_expected_facts'] or '없음'}",
                f"- 금지 표현 검출: "
                f"{', '.join(result['forbidden_phrase_hits']) or '없음'}",
                "",
                "### 입력 알림",
                "",
            ]
        )
        for notification in result["notifications"]:
            lines.append(
                f"- `{notification['timestamp']}` **{notification['title']}** — "
                f"{notification['body']}"
            )
        lines.extend(["", "### 사람 기준 요약", ""])
        lines.extend(
            f"- {line}" for line in result["reference_summary_lines"]
        )
        lines.extend(["", "### 모델 요약", ""])
        if result["summary_lines"]:
            lines.extend(f"- {line}" for line in result["summary_lines"])
        else:
            lines.append(f"- 오류: {result['error']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--task", choices=("briefing", "summary"), default="summary")
    parser.add_argument(
        "--prompt-style", choices=("runtime", "training"), default="runtime",
        help="compare current runtime formatting with the exact training prompt formatting",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="disable sampling and use greedy decoding for a controlled comparison",
    )
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
    parser.add_argument(
        "--review-output",
        type=Path,
        help="optional path for a human-readable Markdown review report",
    )
    args = parser.parse_args()
    args.cases = args.cases or (EVALUATION_PATH if args.task == "briefing" else DEFAULT_CASES_PATH)

    cases = load_cases(args.cases)
    validate_evaluation_cases(cases)
    tokenizer, model = load_model(adapter_path=args.adapter_path)
    structured_count = 0
    raw_contract_count = 0
    format_repair_count = 0
    fact_hits = 0
    fact_total = 0
    passed_case_count = 0
    category_hits = 0
    latencies: list[float] = []
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        group = case.get("input")
        if not isinstance(group, Mapping):
            raise ValueError(f"case {index} does not contain an input object")

        max_summary_lines = _max_summary_lines(case)
        reference_summary_lines = _reference_summary_lines(case)
        raw_response, elapsed_seconds = generate_summary(
            tokenizer=tokenizer,
            model=model,
            group=group,
            max_summary_lines=max_summary_lines,
            seed=42 + index,
            do_sample=not args.greedy,
            prompt_style=args.prompt_style,
            task=args.task,
        )
        latencies.append(elapsed_seconds)
        expected_facts = _expected_facts(case)
        forbidden_phrases = _forbidden_phrases(case)
        fact_total += len(expected_facts)

        try:
            predicted_category = None
            expected_category = case.get("expected_category", group.get("category"))
            category_passed = True
            if args.task == "briefing":
                decision = parse_response(raw_response)
                summary_lines = decision.summary_lines
                predicted_category = decision.primary_category
                category_passed = predicted_category == expected_category
                category_hits += int(category_passed)
                format_repaired = False
            else:
                parsed_response = parse_summary_response_with_metadata(raw_response)
                summary_lines = parsed_response.summary_lines
                format_repaired = parsed_response.format_repaired
            structured_count += 1
            if format_repaired:
                format_repair_count += 1
            else:
                raw_contract_count += 1
            combined = " ".join(summary_lines)
            hits = sum(
                any(alternative in combined for alternative in alternatives)
                for alternatives in expected_facts
            )
            missing_expected_facts = [
                list(alternatives)
                for alternatives in expected_facts
                if not any(
                    alternative in combined for alternative in alternatives
                )
            ]
            forbidden_hits = tuple(
                phrase for phrase in forbidden_phrases if phrase in combined
            )
            line_limit_passed = len(summary_lines) <= _expected_max_summary_lines(case)
            fact_hits += hits
            error = None
        except ValueError as exc:
            predicted_category = None
            category_passed = args.task != "briefing"
            summary_lines = ()
            hits = 0
            missing_expected_facts = [
                list(alternatives) for alternatives in expected_facts
            ]
            forbidden_hits = ()
            line_limit_passed = False
            error = str(exc)
            format_repaired = False

        passed = (
            error is None
            and hits == len(expected_facts)
            and not forbidden_hits
            and line_limit_passed
            and category_passed
        )
        if passed:
            passed_case_count += 1

        results.append(
            {
                "case_id": case.get("case_id", index),
                "category": case.get("expected_category", group.get("category")),
                "predicted_category": predicted_category,
                "category_passed": category_passed if args.task == "briefing" else None,
                "passed": passed,
                "structured_output": error is None,
                "raw_contract_compliant": error is None and not format_repaired,
                "format_repaired": format_repaired,
                "fact_hits": hits,
                "fact_total": len(expected_facts),
                "missing_expected_facts": missing_expected_facts,
                "forbidden_phrase_hits": list(forbidden_hits),
                "max_summary_lines": max_summary_lines,
                "expected_max_summary_lines": _expected_max_summary_lines(case),
                "line_limit_passed": line_limit_passed,
                "latency_seconds": round(elapsed_seconds, 2),
                "summary_lines": list(summary_lines),
                "reference_summary_lines": list(reference_summary_lines),
                "notifications": [
                    {
                        "timestamp": notification.get("timestamp"),
                        "title": notification.get("title"),
                        "body": notification.get("body"),
                    }
                    for notification in group.get("notifications", [])
                ],
                "error": error,
                "raw_response": raw_response,
            }
        )

    report = {
        "task": args.task,
        "category_accuracy": round(category_hits / len(cases), 4) if args.task == "briefing" else None,
        "generation_mode": "greedy" if args.greedy else "sampling",
        "prompt_style": args.prompt_style,
        "model": MODEL_NAME,
        "adapter_path": str(args.adapter_path)
        if args.adapter_path is not None
        else None,
        "case_count": len(cases),
        "structured_output_rate": round(structured_count / len(cases), 4),
        "raw_contract_compliance_rate": round(
            raw_contract_count / len(cases), 4
        ),
        "format_repair_rate": round(format_repair_count / len(cases), 4),
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
    if args.review_output is not None:
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        args.review_output.write_text(
            _render_review_markdown(report),
            encoding="utf-8",
        )
    print(serialized_report)


if __name__ == "__main__":
    main()
