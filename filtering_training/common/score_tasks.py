"""용도: 오프라인 점수 실험의 입력 분리, 엄격한 출력 검증과 정책 지표.
생성일: 2026-10-03
"""

import hashlib
import json
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.filtering.schema import CurrentContext, FilteringSample

from src.filtering.policy import should_pass
from src.filtering.prompt import build_messages, parse_model_output, SYSTEM_PROMPT


TASKS = ("full", "scores", "urgency", "relevance")


class RecentWindow(BaseModel):
    """Offline history input; no timestamps are assumed to be available."""
    model_config = ConfigDict(extra="forbid")
    app_name: str = Field(min_length=1, pattern=r"\S")
    window_title: str


class HistoryContext(CurrentContext):
    recent_windows: list[RecentWindow] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def consistent_recent_apps(self):
        if self.recent_processes != [window.app_name for window in self.recent_windows]:
            raise ValueError("recent app names must match the supplied windows in order")
        return self


class HistorySample(FilteringSample):
    """Offline schema extension; existing app schema remains unchanged."""
    context: HistoryContext


def load_score_samples(path):
    """Preserve optional history fields instead of silently dropping them."""
    samples, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        schema = HistorySample if "recent_windows" in value.get("context", {}) else FilteringSample
        sample = schema.model_validate(value)
        if sample.notification.id in seen:
            raise ValueError("duplicate notification id")
        parse_model_output(json.dumps(sample.label.model_dump(), ensure_ascii=False))
        seen.add(sample.notification.id)
        samples.append(sample)
    if not samples:
        raise ValueError("dataset contains no samples")
    return samples
FIELDS = {"scores": ("urgency_score", "relevance_score"),
          "urgency": ("urgency_score",), "relevance": ("relevance_score",)}
URGENCY_RULE = (
    "긴급도는 알림 자체의 시간 민감도, 놓쳤을 때 영향, 행동 필요성으로 판단하세요. "
    "1 언제 확인해도 무방, 2 나중에 확인 가능, 3 비교적 빠른 확인, "
    "4 현재 작업을 중단하고 확인할 가치, 5 즉시 확인 또는 대응 필요. "
    "선택적인 광고 마감이나 발신자만으로 긴급도를 높이지 마세요."
)
RELEVANCE_RULE = (
    "관련도는 현재 목적에 주는 도움으로 판단하세요. "
    "1 무관, 2 같은 조직/분야지만 거의 무관, 3 같은 주제지만 도움 불분명, "
    "4 현재 목적에 직접 도움, 5 정확한 작업 대상이나 행동과 연결. "
    "같은 주제의 농담은 3 이하입니다. 빈 창에서는 현재 작업을 추측하지 마세요. "
    "집중 모드 ON입니다. 빈 창을 쉬는 상태로 추정하지 마세요. "
    "체류 시간과 최근 앱만으로 작업 주제나 중요도를 단정하지 마세요."
)
HISTORY_RELEVANCE_RULE = (
    "관련도는 현재 창과 최근 최대 3개 앱의 창 제목에서 확인되는 작업 흐름에 알림이 주는 도움으로 판단하세요. "
    "먼저 작업 대상과 행동을 확인한 뒤 알림이 그 목적을 돕는지 판단하세요. "
    "1 무관, 2 같은 분야지만 거의 무관, 3 같은 주제지만 도움 불분명, "
    "4 작업 목적에 직접 도움, 5 정확한 작업 대상이나 행동과 연결. "
    "최근 창들이 다른 목적의 작업 흐름을 일관되게 보여주면 현재 창 제목과 단어가 같다는 이유만으로 높은 관련도를 고정하지 마세요. "
    "현재 창이 비어 있어도 최근 창 제목들이 알림의 정확한 작업 대상과 행동을 보여주면 관련도 4 또는 5가 가능합니다. "
    "현재 창과 최근 창이 비어 있거나 앱 이름만 있고 목적을 확인할 수 없다면 작업을 추측하지 마세요. "
    "관련 작업 분야의 장식용 상품 광고는 실제 작업을 돕는 기능이나 자료로 취급하지 마세요. "
    "최근 앱의 개수만 세어 목적을 결정하거나 최근 기록을 항상 현재 창보다 우선하지 마세요. "
    "집중 모드 ON이며 빈 창을 쉬는 상태로 추정하지 마세요. 입력에 없는 시각·체류 시간·작업 전환을 만들어 내지 마세요."
)


def system_prompt(task, history=False):
    if task == "full":
        return SYSTEM_PROMPT
    if task not in FIELDS:
        raise ValueError("unknown score task")
    fields = ", ".join(FIELDS[task])
    rules = (URGENCY_RULE if task != "relevance" else "") + " " + (
        (HISTORY_RELEVANCE_RULE if history else RELEVANCE_RULE) if task != "urgency" else "")
    return (f"JSON 객체 하나만 출력하세요. 필드는 {fields}이며 각각 1~5 정수입니다. "
            + rules + " 입력에 없는 정보를 추측하지 마세요. 설명이나 마크다운을 출력하지 마세요.")


def prompt_digest(task):
    return hashlib.sha256(system_prompt(task).encode("utf-8")).hexdigest()


def messages(sample, task, history_mode="titles"):
    if history_mode not in ("none", "apps", "titles"):
        raise ValueError("unknown history input mode")
    if task == "full":
        return build_messages(sample.notification, sample.context)
    payload = {"notification": sample.notification.model_dump(mode="json")}
    # Urgency never receives the context, even an empty context or its timestamp.
    if task == "urgency":
        # Variant IDs contain related/unrelated/empty suffixes and leak context.
        payload["notification"].pop("id")
    else:
        payload["context"] = sample.context.model_dump(mode="json")
        if isinstance(sample.context, HistoryContext):
            if history_mode != "titles":
                payload["context"].pop("recent_windows", None)
            if history_mode == "none":
                payload["context"]["recent_processes"] = []
    return [{"role": "system", "content": system_prompt(task,history=isinstance(sample.context,HistoryContext))},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def parse_scores(raw, task):
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if task == "full":
        # Check duplicate keys first, then reuse the production contract validator.
        label = parse_model_output(raw)
        return {key: getattr(label, key) for key in FIELDS["scores"]}
    if task not in FIELDS or not isinstance(value, dict) or set(value) != set(FIELDS[task]):
        raise ValueError("unexpected score fields")
    if any(type(v) is not int or not 1 <= v <= 5 for v in value.values()):
        raise ValueError("scores must be integers from 1 to 5")
    return value


def sft_record(sample, task):
    target = sample.label.model_dump() if task == "full" else {
        key: getattr(sample.label, key) for key in FIELDS[task]}
    raw = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    parse_scores(raw, task)
    return {"messages": messages(sample, task) + [{"role": "assistant", "content": raw}]}


def unique_urgency_samples(samples):
    groups = {}
    for sample in samples:
        notification = sample.notification.model_dump(mode="json")
        notification.pop("id")
        key = json.dumps(notification, ensure_ascii=False, sort_keys=True)
        previous = groups.get(key)
        if previous and (previous.label.urgency_score, previous.label.category) != (
                sample.label.urgency_score, sample.label.category):
            raise ValueError("context variants disagree on urgency or category")
        groups.setdefault(key, sample)
    return list(groups.values())


def score_metrics(samples, predictions):
    if not samples or len(samples) != len(predictions):
        raise ValueError("predictions must cover all samples")
    for prediction in predictions:
        if prediction is not None:
            parse_scores(json.dumps(prediction), "scores")
    valid = [(s, p) for s, p in zip(samples, predictions) if p is not None]
    urgent = blocked = failed = invalid_urgent = false_pass = correct = gold_block = 0
    groups = defaultdict(list)
    by_context = defaultdict(lambda: {"urgent": 0, "urgent_failures": 0})
    for s, p in zip(samples, predictions):
        expected = should_pass(s.label.urgency_score, s.label.relevance_score)
        decision = should_pass(**p) if p is not None else None
        correct += decision == expected
        gold_block += not expected
        false_pass += not expected and decision is True
        is_urgent = s.label.urgency_score >= 4
        urgent += is_urgent
        blocked += is_urgent and decision is False
        failed += is_urgent and decision is not True
        invalid_urgent += is_urgent and p is None
        role = s.notification.id.rsplit("_", 1)[-1]
        by_context[role]["urgent"] += is_urgent
        by_context[role]["urgent_failures"] += is_urgent and decision is not True
        groups[s.notification.id.rsplit("_", 1)[0]].append(p)
    invariance = {}
    for group, items in groups.items():
        values = [p["urgency_score"] for p in items if p is not None]
        invariance[group] = {"count": len(items), "valid": len(values),
                             "range": max(values)-min(values) if len(values) == len(items) else None}
    metrics = {"count": len(samples), "valid_outputs": len(valid),
               "policy_correct": correct, "policy_accuracy": correct/len(samples),
               "urgent_count": urgent, "urgent_false_blocks": blocked,
               "urgent_invalid": invalid_urgent, "urgent_failures": failed,
               "gold_block_count": gold_block, "false_passes": false_pass,
               "by_context": dict(by_context), "urgency_invariance": invariance,
               "fully_invariant_groups": sum(x["count"] > 1 and x["range"] == 0 for x in invariance.values())}
    for field in FIELDS["scores"]:
        errors = [abs(getattr(s.label, field)-p[field]) for s, p in valid]
        metrics[field] = {"exact_accuracy": sum(e == 0 for e in errors)/len(samples),
                          "within_one_accuracy": sum(e <= 1 for e in errors)/len(samples),
                          "mae_on_valid": sum(errors)/len(errors) if errors else None}
    return metrics
