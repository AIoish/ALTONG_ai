"""용도: Encoder의 창 기반 최신 입력과 기존 모델 재현용 입력을 명시적으로 구분한다.
생성일: 2026-11-04
"""

from copy import deepcopy
import json

INPUT_SCHEMA = 'notification-previous-windows-v3'
WINDOWS_INPUT_SCHEMA = 'notification-windows-v2'
LEGACY_INPUT_SCHEMA = 'notification-history-v1'
TASKS = ('urgency', 'relevance')
RECENT_WINDOWS_POLICY = 'exclude-current-newest-first-max3'


def previous_windows(context):
    """Match the app's case-sensitive Where(...).Take(3), preserving supplied recency order."""
    current = (context.get('active_process', ''), context.get('window_title', ''))
    return [window for window in context.get('recent_windows', [])
            if (window.get('app_name', ''), window.get('window_title', '')) != current][:3]


def windows_only_state(state, *, schema=INPUT_SCHEMA):
    if schema not in (INPUT_SCHEMA, WINDOWS_INPUT_SCHEMA):
        raise ValueError('unknown windows input schema')
    result = deepcopy(state)
    if 'context' in result:
        result['context'].pop('recent_processes', None)
        if schema == INPUT_SCHEMA:
            result['context']['recent_windows'] = previous_windows(result['context'])
    return result


def input_text(state, task, *, schema=INPUT_SCHEMA):
    if task not in TASKS:
        raise ValueError('unknown task')
    if schema not in (INPUT_SCHEMA, WINDOWS_INPUT_SCHEMA, LEGACY_INPUT_SCHEMA):
        raise ValueError('unknown encoder input schema')
    value = {'notification': {k: v for k, v in state['notification'].items() if k != 'id'}}
    if task == 'relevance':
        context = state['context']
        value['current_window'] = {k: context.get(k, '') for k in ('active_process', 'window_title')}
        windows = previous_windows(context) if schema == INPUT_SCHEMA else context.get('recent_windows', [])
        value['recent_windows'] = [{k: w.get(k, '') for k in ('app_name', 'window_title')} for w in windows]
        if schema == LEGACY_INPUT_SCHEMA:
            value['recent_processes'] = context.get('recent_processes', [])
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def legacy_input_text(state, task):
    """Only for unchanged models trained with the old serialized field."""
    return input_text(state, task, schema=LEGACY_INPUT_SCHEMA)


def input_schema_for_run(config):
    schema = config.get('input_schema', LEGACY_INPUT_SCHEMA)
    if schema not in (INPUT_SCHEMA, WINDOWS_INPUT_SCHEMA, LEGACY_INPUT_SCHEMA):
        raise ValueError('unknown run input schema')
    return schema


def encode_checked(tokenizer, texts):
    encoded = tokenizer(texts, truncation=False, padding=False, return_token_type_ids=False)
    lengths = [len(ids) for ids in encoded['input_ids']]
    if max(lengths, default=0) > 512:
        raise ValueError(f'input exceeds 512; no silent truncation: {max(lengths)}')
    return [{'input_ids': ids, 'attention_mask': mask}
            for ids, mask in zip(encoded['input_ids'], encoded['attention_mask'])], lengths
