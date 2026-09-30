"""Notification briefing pipeline.

The models in this package are local adapter models for the summary module.  They
are intentionally kept separate from the future team-owned models in ``common``.
"""

from .pipeline import SessionBriefingService
from .qwen_provider import QwenBriefingProvider, TransformersQwenBackend

__all__ = [
    "QwenBriefingProvider",
    "SessionBriefingService",
    "TransformersQwenBackend",
]
