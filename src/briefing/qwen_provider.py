"""Local Qwen summary provider for the application briefing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .categorization import categorize_group
from .qwen_prompt import (
    MAX_SUMMARY_LINES,
    MODEL_NAME,
    build_messages,
    parse_summary_response,
)
from .schema import BriefingItem, format_timestamp
from .summarize import BriefingProvider, RuleBasedBriefingProvider


LOGGER = logging.getLogger(__name__)


class QwenTextBackend(Protocol):
    """Boundary used to keep heavyweight model code out of provider tests."""

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str: ...


class TransformersQwenBackend:
    """Lazily load a local Qwen model and optional PEFT adapter."""

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 160,
        seed: int = 42,
    ) -> None:
        self._model_name = model_name
        self._adapter_path = (
            Path(adapter_path).expanduser() if adapter_path is not None else None
        )
        self._max_new_tokens = max_new_tokens
        self._seed = seed
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def adapter_path(self) -> Path | None:
        return self._adapter_path

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        if self._adapter_path is not None and not self._adapter_path.is_dir():
            raise FileNotFoundError(
                f"Qwen adapter directory does not exist: {self._adapter_path}"
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        if self._adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self._adapter_path))

        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        tokenizer, model = self._ensure_loaded()

        import torch

        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

        prompt = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][model_inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def build_group_context(items: Sequence[BriefingItem]) -> dict[str, object]:
    """Map validated application records to the shared Qwen prompt contract."""

    if not items:
        raise ValueError("cannot summarize an empty briefing group")

    ordered = sorted(items, key=lambda item: item.notification.timestamp)
    urgency_values = [item.filter_result.urgency_score for item in ordered]
    relevance_values = [item.filter_result.relevance_score for item in ordered]
    category = categorize_group(list(ordered)).primary_category
    first = ordered[0].notification

    def score(values: list[int]) -> dict[str, int | float]:
        return {
            "min": min(values),
            "max": max(values),
            "average": round(sum(values) / len(values), 2),
        }

    return {
        "app_name": first.app_name,
        "sender": first.sender,
        "category": category,
        "urgency_score": score(urgency_values),
        "relevance_score": score(relevance_values),
        "notifications": [
            {
                "id": item.notification.id,
                "timestamp": format_timestamp(item.notification.timestamp),
                "title": item.notification.title,
                "body": item.notification.body,
            }
            for item in ordered
        ],
    }


class QwenBriefingProvider:
    """Generate strict local-model summaries with a rule-based safety fallback."""

    def __init__(
        self,
        *,
        adapter_path: str | Path | None = None,
        model_name: str = MODEL_NAME,
        backend: QwenTextBackend | None = None,
        fallback_provider: BriefingProvider | None = None,
        max_summary_lines: int = MAX_SUMMARY_LINES,
        allow_fallback: bool = True,
    ) -> None:
        self._backend = backend or TransformersQwenBackend(
            model_name=model_name,
            adapter_path=adapter_path,
        )
        self._fallback_provider = fallback_provider or RuleBasedBriefingProvider()
        self._max_summary_lines = max_summary_lines
        self._allow_fallback = allow_fallback
        self._disabled = False

    def summarize(self, items: Sequence[BriefingItem]) -> tuple[str, ...]:
        if not items:
            return ()
        if self._disabled:
            return self._fallback_provider.summarize(items)

        try:
            group_context = build_group_context(items)
            messages = build_messages(
                group_context,
                max_summary_lines=self._max_summary_lines,
            )
            raw_response = self._backend.generate(messages)
            return parse_summary_response(raw_response)
        except Exception:
            # The briefing must remain available when model dependencies, model
            # files, GPU memory, or generated JSON fail at this optional boundary.
            if not self._allow_fallback:
                raise
            self._disabled = True
            LOGGER.exception("Qwen summary failed; using rule-based fallback")
            return self._fallback_provider.summarize(items)
