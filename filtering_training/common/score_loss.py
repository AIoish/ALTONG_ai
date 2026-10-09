"""용도: 단일 점수 completion의 숫자 토큰만 가중하는 causal loss와 균등 샘플링.
생성일: 2026-10-04
"""

from collections import defaultdict
import random


def balanced_indices(scores, seed=42):
    groups = defaultdict(list)
    for index, score in enumerate(scores):
        if type(score) is not int or score not in range(1, 6):
            raise ValueError("invalid gold score")
        groups[score].append(index)
    if set(groups) != set(range(1, 6)):
        raise ValueError("balancing requires all five gold scores")
    rng = random.Random(seed)
    quota, extra = divmod(len(scores), 5)
    chosen = []
    for position, score in enumerate(range(1, 6)):
        chosen.extend(rng.choice(groups[score]) for _ in range(quota + (position < extra)))
    rng.shuffle(chosen)
    return chosen


def weighted_score_loss(logits, labels, digit_ids, numeric_weight=7.0):
    """Shift labels once; masked prompt/padding contributes zero weight and gradient."""
    import torch
    import torch.nn.functional as F
    if numeric_weight < 1:
        raise ValueError("numeric weight must be at least one")
    shifted_labels = labels[..., 1:]
    shifted_logits = logits[..., :-1, :].float()
    valid = shifted_labels != -100
    numeric = torch.zeros_like(valid)
    for token_id in digit_ids:
        numeric |= shifted_labels == token_id
    if not valid.any() or (numeric.sum(-1) != 1).any():
        raise ValueError("each single-score completion must have exactly one supervised digit")
    losses = F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.size(-1)),
                             shifted_labels.reshape(-1), ignore_index=-100,
                             reduction="none").reshape_as(shifted_labels)
    weights = valid.to(losses.dtype) * torch.where(numeric, numeric_weight, 1.0)
    return (losses * weights).sum() / weights.sum()
