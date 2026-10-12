"""Compatibility exports for the shared runtime Qwen prompt contract."""

from src.briefing.qwen_prompt import (
    MAX_LINE_LENGTH,
    MAX_SUMMARY_LINES,
    MODEL_NAME,
    ParsedSummaryResponse,
    SYSTEM_PROMPT,
    SummaryResponseError,
    build_messages,
    parse_summary_response,
    parse_summary_response_with_metadata,
)

__all__ = [
    "MAX_LINE_LENGTH",
    "MAX_SUMMARY_LINES",
    "MODEL_NAME",
    "ParsedSummaryResponse",
    "SYSTEM_PROMPT",
    "SummaryResponseError",
    "build_messages",
    "parse_summary_response",
    "parse_summary_response_with_metadata",
]
