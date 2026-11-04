"""용도: 다른 프롬프트 군의 동일 검증 자료와 예측 완전성을 확인해 비교한다.
생성일: 2026-10-03
"""

import argparse
import json
from pathlib import Path

from filtering_training.common.score_tasks import score_metrics, prompt_digest, parse_scores
from filtering_training.modeling.evaluate import load_split_samples
from filtering_training.preparation.prepare_score_experiment import digest, write_json
from src.filtering.policy import POLICY_VERSION


def compare(source, reports, output):
    if output.exists():
        raise ValueError("comparison already exists")
    samples = load_split_samples(source/"dataset.jsonl", source/"prepared", "validation")
    ids = [s.notification.id for s in samples]
    results = {}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        variant = report["variant"]
        if variant not in ("A", "B", "C") or variant in results:
            raise ValueError("unknown or repeated variant")
        if (report["evaluated_ids"] != ids or report["sampling"] is not None
                or report["dataset_sha256"] != digest(source/"dataset.jsonl")
                or report["source_manifest_sha256"] != digest(source/"prepared/manifest.json")
                or report["policy_version"] != POLICY_VERSION or report["split"] != "validation"):
            raise ValueError("reports do not cover the same fixed validation split")
        expected_tasks = {"A": ("full",), "B": ("scores",), "C": ("urgency", "relevance")}[variant]
        if report["prompt_hashes"] != {task: prompt_digest(task) for task in expected_tasks}:
            raise ValueError("recorded prompt differs from current task")
        prediction_path = path.with_suffix(".predictions.jsonl")
        if report["predictions_sha256"] != digest(prediction_path):
            raise ValueError("predictions changed")
        rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
        if [row["notification_id"] for row in rows] != ids:
            raise ValueError("missing, duplicated or unordered predictions")
        for sample, row in zip(samples, rows):
            if (row["gold"] != sample.label.model_dump()
                    or row["notification"] != sample.notification.model_dump(mode="json")
                    or row["context"] != sample.context.model_dump(mode="json")):
                raise ValueError("input or gold label differs from source")
            responses = row["responses"]
            if set(responses) != set(expected_tasks):
                raise ValueError("missing task response")
            parsed, invalid = {}, False
            for task in expected_tasks:
                response = responses[task]
                try:
                    values = parse_scores(response["raw"], task)
                except ValueError:
                    if not response["error"]:
                        raise ValueError("invalid raw output was accepted")
                    invalid = True
                else:
                    if response["error"]:
                        if response["error"] != "generation reached token limit without EOS":
                            raise ValueError("valid raw output has an unexplained error")
                        invalid = True
                    else:
                        parsed.update(values)
            if row["prediction"] != (None if invalid else parsed):
                raise ValueError("prediction does not match raw task outputs")
        metrics = score_metrics(samples, [row["prediction"] for row in rows])
        if metrics != report["metrics"]:
            raise ValueError("recorded metrics differ from predictions")
        errors = []
        from src.filtering.policy import should_pass
        for sample, row in zip(samples, rows):
            p = row["prediction"]
            decision = should_pass(**p) if p else None
            expected = should_pass(sample.label.urgency_score, sample.label.relevance_score)
            if decision != expected:
                errors.append({"id": sample.notification.id, "body": sample.notification.body,
                               "window": sample.context.window_title, "gold": sample.label.model_dump(),
                               "prediction": p, "decision": decision})
        results[variant] = {"report_sha256": digest(path), "metrics": metrics,
                            "performance": report["performance"], "errors": errors}
    if set(results) != {"A", "B", "C"}:
        raise ValueError("all three variants required")
    summary = {"purpose": "validated comparison; not a final model selection",
               "created_date": "2026-10-03", "source_sha256": digest(source/"dataset.jsonl"),
               "evaluated_ids": ids, "variants": results}
    write_json(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("filtering_training/outputs/v3_reviewed_01"))
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compare(args.source, args.reports, args.output)
    for variant, result in summary["variants"].items():
        m = result["metrics"]
        print(variant, m["policy_correct"], "/", m["count"], "urgent failures", m["urgent_failures"],
              "mean seconds", round(result["performance"]["mean_seconds"], 3))


if __name__ == "__main__":
    main()
