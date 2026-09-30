"""Run the application briefing pipeline with a local Qwen adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from src.briefing import QwenBriefingProvider, SessionBriefingService


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
    parser.add_argument("--adapter-path", type=Path, required=True)
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

    provider = QwenBriefingProvider(
        adapter_path=args.adapter_path,
        allow_fallback=False,
    )
    service = SessionBriefingService(provider=provider)
    encoded = service.build_json(
        session_id=args.session_id,
        notifications=load_json_array(args.notifications),
        filter_results=load_json_array(args.filter_results),
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
