"""용도: Qwen 프롬프트 없이 원본 스키마·승인·해시를 확인하고 고정 검증74행을 읽는다.
생성일: 2026-11-04
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from filtering_training.common.artifact_io import ROOT, digest, load_lines, read
from filtering_training.common.encoder_input import (
    INPUT_SCHEMA, WINDOWS_INPUT_SCHEMA, LEGACY_INPUT_SCHEMA, TASKS, input_text, windows_only_state,
)
from filtering_training.common.paths import resolve_existing_path
from src.filtering.schema import CATEGORIES, CurrentContext, FilterLabel, FilteringSample

SOURCE = ROOT / 'filtering_training/outputs/v3_reviewed_01'
VALIDATION_POOL = ROOT / 'filtering_training/outputs/v5_validation_pool_01'
EVALUATION_SOURCES = {
    'safety_development': ROOT / 'filtering_training/outputs/v10_safety_supplement_01/development.jsonl',
    'personal100': ROOT / 'filtering_training/outputs/v9_personal_use_eval_01/dataset.jsonl',
    'boundary14': ROOT / 'filtering_training/outputs/v11_boundary_supplement_01/dataset.jsonl',
}


class RecentWindow(BaseModel):
    model_config = ConfigDict(extra='forbid')
    app_name: str = Field(min_length=1, pattern=r'\S')
    window_title: str


class HistoryContext(CurrentContext):
    """Frozen source schema, including the old redundant field for approval verification."""
    recent_windows: list[RecentWindow] = Field(default_factory=list, max_length=3)

    @model_validator(mode='after')
    def consistent_recent_apps(self):
        if self.recent_processes != [window.app_name for window in self.recent_windows]:
            raise ValueError('recent app names must match the supplied windows in order')
        return self


class HistorySample(FilteringSample):
    context: HistoryContext


def parse_label(value):
    if not isinstance(value, dict) or set(value) != set(FilterLabel.model_fields):
        raise ValueError('model output must contain exactly the four label fields')
    label = FilterLabel.model_validate(value, strict=True)
    if label.category not in CATEGORIES:
        raise ValueError('category is not in the labeling guideline')
    if not label.ai_summary_reason.strip():
        raise ValueError('ai_summary_reason must be non-empty')
    return label


def _load_samples(path, *, history):
    samples, seen = [], set()
    for number, line in enumerate(resolve_existing_path(Path(path)).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('sample must be a JSON object')
            schema = HistorySample if history and 'recent_windows' in value.get('context', {}) else FilteringSample
            sample = schema.model_validate(value)
            parse_label(sample.label.model_dump())
        except ValueError as error:
            raise ValueError(f'invalid sample at line {number}') from error
        if sample.notification.id in seen:
            raise ValueError(f'duplicate notification id at line {number}')
        seen.add(sample.notification.id)
        samples.append(sample)
    if not samples:
        raise ValueError('dataset contains no samples')
    return samples


def load_samples(path):
    return _load_samples(path, history=False)


def load_score_samples(path):
    return _load_samples(path, history=True)


def load_split_samples(dataset, prepared_dir, split):
    dataset = resolve_existing_path(Path(dataset))
    manifest = read(resolve_existing_path(Path(prepared_dir)) / 'manifest.json')
    if digest(dataset) != manifest['source_sha256']:
        raise ValueError('prepared split does not match the current dataset')
    ids = manifest['splits'][split]['notification_ids']
    if len(ids) != len(set(ids)):
        raise ValueError('prepared split contains duplicate IDs')
    by_id = {sample.notification.id: sample for sample in load_samples(dataset)}
    try:
        return [by_id[item] for item in ids]
    except KeyError as error:
        raise ValueError('prepared split references an unknown sample') from error


def load_validation_for_evaluation(validation_pool=None):
    legacy = load_split_samples(SOURCE / 'dataset.jsonl', SOURCE / 'prepared', 'validation')
    groups = {'legacy_24': [sample.notification.id for sample in legacy]}
    if validation_pool is None:
        return legacy, groups, None
    validation_pool = Path(validation_pool)
    manifest = read(validation_pool / 'manifest.json')
    subset = manifest.get('evaluation_set', {})
    if manifest.get('intended_split') != 'validation' or not subset or subset.get('training_used'):
        raise ValueError('approved validation subset required')
    if (digest(validation_pool / subset['dataset_file']) != subset['dataset_sha256']
            or digest(validation_pool / 'approval.json') != subset['preview_approval_sha256']):
        raise ValueError('approved validation subset changed')
    approved = {}
    for source in subset['seed_sources']:
        folder = Path(source['folder'])
        if not folder.is_absolute():
            folder = ROOT / folder
        if (digest(folder / 'dataset.jsonl') != source['dataset_sha256']
                or digest(folder / 'approval.json') != source['approval_sha256']):
            raise ValueError('validation approval evidence changed')
        for sample in load_score_samples(folder / 'dataset.jsonl'):
            approved[sample.notification.id] = sample.model_dump(mode='json')
    for item in read(validation_pool / 'approval.json')['approved_cases']:
        approved[item['notification']['id']] = item
    extra = load_score_samples(validation_pool / subset['dataset_file'])
    ids = {sample.notification.id for sample in extra}
    if ids != set(subset['approved_ids']) or ids != set(approved):
        raise ValueError('unreviewed validation rows in evaluation subset')
    for sample in extra:
        if sample.model_dump(mode='json') != approved[sample.notification.id]:
            raise ValueError('validation label differs from approval')
    if set(groups['legacy_24']) & set(approved):
        raise ValueError('duplicate evaluation ids')
    groups['approved_expansion'] = [sample.notification.id for sample in extra]
    return legacy + extra, groups, subset


def validation(*, schema=INPUT_SCHEMA):
    samples, groups, _ = load_validation_for_evaluation(VALIDATION_POOL)
    texts = [[input_text(sample.model_dump(mode='json'), task, schema=schema) for task in TASKS]
             for sample in samples]
    return samples, groups, texts


def evaluation_sets(*, schema=INPUT_SCHEMA):
    """Keep frozen sources unchanged and apply one explicit model-input contract to every split."""
    samples, _, _ = validation(schema=schema)
    result = {'validation': [sample.model_dump(mode='json') for sample in samples]}
    result.update({name: load_lines(path) for name, path in EVALUATION_SOURCES.items()})
    if schema in (INPUT_SCHEMA, WINDOWS_INPUT_SCHEMA):
        return {name: [windows_only_state(row, schema=schema) for row in rows] for name, rows in result.items()}
    if schema != LEGACY_INPUT_SCHEMA:
        raise ValueError('unknown evaluation input schema')
    return result
