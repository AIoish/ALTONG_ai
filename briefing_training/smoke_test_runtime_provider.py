"""Run the application briefing pipeline with a local Qwen adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from src.briefing import QwenBriefingProvider, SessionBriefingService
from src.briefing.reports import category_report, render_category_markdown


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_DIR = REPOSITORY_ROOT / "data" / "sample" / "briefing"


def load_json_array(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"expected an array of objects: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--rule-based", action="store_true", help="offline structure test; does not assess model quality")
    parser.add_argument("--contract", choices=("briefing", "summary"), default="briefing")
    parser.add_argument("--output", type=Path, help="dashboard session_id + groups JSON")
    parser.add_argument("--report-output", type=Path, help="category-organized Markdown report")
    parser.add_argument("--category-output", type=Path, help="optional category-organized JSON projection")
    parser.add_argument(
        "--notifications",
        type=Path,
        default=DEFAULT_SAMPLE_DIR / "raw_notifications.json",
    )
    parser.add_argument(
        "--filter-results",
        type=Path,
        default=DEFAULT_SAMPLE_DIR / "filter_results.json",
    )
    parser.add_argument("--session-id", default="runtime_qwen_smoke")
    args = parser.parse_args()

    if not args.rule_based and args.adapter_path is None:
        parser.error("provide --adapter-path or --rule-based")
    provider = None if args.rule_based else QwenBriefingProvider(
        adapter_path=args.adapter_path, allow_fallback=False, contract=args.contract,
    )
    service = SessionBriefingService(provider=provider)
    briefing = service.build(
        session_id=args.session_id,
        notifications=load_json_array(args.notifications),
        filter_results=load_json_array(args.filter_results),
    )
    encoded = json.dumps(briefing.to_dict(), ensure_ascii=False, indent=2)
    for path, content in (
        (args.output, encoded + "\n"),
        (args.report_output, render_category_markdown(briefing)),
        (args.category_output, json.dumps(category_report(briefing), ensure_ascii=False, indent=2) + "\n"),
    ):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
