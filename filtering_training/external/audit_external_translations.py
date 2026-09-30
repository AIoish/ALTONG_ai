"""Audit public machine-translated notification drafts before any relabeling."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from filtering_training import TRAINING_ROOT

from filtering_training.external.translate_external_candidates import OUTPUT_PATH as INPUT_PATH

REPORT_PATH = TRAINING_ROOT / "outputs" / "audit" / "external_translation_audit.json"
DIGITS = re.compile(r"\d+")
HANGUL = re.compile(r"[가-힣]")


def audit(input_path: Path) -> dict:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    flags = {}
    keys = Counter((row["title"].strip().casefold(), row["body"].strip().casefold()) for row in rows)
    for row in rows:
        reasons = []
        source = row["source_title"] + " " + row["source_body"]
        translated = row["title"] + " " + row["body"]
        if not row["title"].strip() or not row["body"].strip():
            reasons.append("empty_field")
        if not HANGUL.search(translated):
            reasons.append("no_hangul")
        if set(DIGITS.findall(source)) - set(DIGITS.findall(translated)):
            reasons.append("source_number_missing")
        if keys[(row["title"].strip().casefold(), row["body"].strip().casefold())] > 1:
            reasons.append("duplicate_translation")
        if len(translated.strip()) < 0.22 * len(source.strip()):
            reasons.append("very_short_translation")
        if reasons:
            flags[row["source_id"]] = reasons
    counts = Counter(reason for reasons in flags.values() for reason in reasons)
    return {
        "count": len(rows), "flagged_count": len(flags),
        "flag_counts": dict(sorted(counts.items())),
        "source_folder_counts": dict(Counter(row["source_folder"] for row in rows)),
        "flagged_source_ids": flags,
        "status": "heuristic review queue; unflagged translations are not automatically approved",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "flagged_source_ids"}, ensure_ascii=False))


if __name__ == "__main__":
    main()