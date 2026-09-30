"""Compatibility exports for the shared runtime Qwen prompt contract."""

from src.briefing.qwen_prompt import (
    MAX_LINE_LENGTH,
    MAX_SUMMARY_LINES,
    MODEL_NAME,
    SYSTEM_PROMPT,
    SummaryResponseError,
    build_messages,
    parse_summary_response,
)

__all__ = [
    "MAX_LINE_LENGTH",
    "MAX_SUMMARY_LINES",
    "MODEL_NAME",
    "SYSTEM_PROMPT",
    "SummaryResponseError",
    "build_messages",
    "parse_summary_response",
]
