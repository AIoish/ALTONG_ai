"""용도: 분리 점수의 다섯 숫자 후보 입력과 조건부 분류 loss.
생성일: 2026-10-04
"""

import hashlib
import json
from filtering_training.common.score_tasks import URGENCY_RULE, RELEVANCE_RULE, HISTORY_RELEVANCE_RULE


def system_prompt(task, history=False):
    rule = {"urgency": URGENCY_RULE, "relevance": HISTORY_RELEVANCE_RULE if history else RELEVANCE_RULE}[task]
    return "점수 1~5 중 숫자 하나를 선택하세요. " + rule + " 입력에 없는 정보를 추측하지 마세요. 설명을 출력하지 마세요."


def prompt_digest(task, history=False):
    return hashlib.sha256(system_prompt(task,history).encode("utf-8")).hexdigest()


def replace_prompt(messages, task):
    history = task == "relevance" and HISTORY_RELEVANCE_RULE in messages[0]["content"]
    result = [{"role": "system", "content": system_prompt(task,history)}, dict(messages[1])]
    if task == "relevance":
        payload = json.loads(result[1]["content"])
        payload["notification"].pop("id", None)
        result[1]["content"] = json.dumps(payload, ensure_ascii=False)
    return result


def digit_loss(logits, labels, digit_ids, threshold_weight=0.0):
    import math
    import torch
    import torch.nn.functional as F
    if len(digit_ids) != 5 or len(set(digit_ids)) != 5:
        raise ValueError("five distinct digit tokens required")
    if labels.ndim != 1 or not bool(((labels >= 0) & (labels < 5)).all()):
        raise ValueError("labels must be zero-based score classes")
    if not math.isfinite(threshold_weight) or threshold_weight < 0:
        raise ValueError("threshold weight must be finite and nonnegative")
    scores = logits[:, -1, digit_ids].float()
    loss = F.cross_entropy(scores, labels)
    if threshold_weight:
        # Sum candidate probability mass for scores 1..3 versus 4..5.
        grouped = torch.stack((scores[:, :3].logsumexp(-1), scores[:, 3:].logsumexp(-1)), dim=-1)
        loss = loss + threshold_weight * F.cross_entropy(grouped, (labels >= 3).long())
    return loss
