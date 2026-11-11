"""용도: 관련도 전체 행을 검수한 문장 유형·작업 대상·개별 예외와 대조하고 행별 수정 근거를 기록한다.
생성일: 2026-11-08
"""

import argparse
import ast
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import input_text
from filtering_training.preparation.curate_relevance import OUTPUT, core_body, key, write_lines
from filtering_training.quality.audit_relevance_diversity import GENERATOR, authored_templates

POLICY = Path(__file__).with_name('relevance_review_policy.json')
TOPIC_EVENTS = {'joke_focus', 'joke_break', 'community_news', 'color_option', 'theme'}


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def source_definitions(path=GENERATOR):
    values = {}
    for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8'))):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id in {'TRAINING_EXPANSION_DOMAINS', 'TRAINING_EVENT_PHRASINGS', 'BULK_EVENTS',
                             'TRAINING_URGENT_SCENARIOS', 'TRAINING_URGENT_GOALS', 'LOW_SCORE_RESOURCES',
                             'unrelated_templates', 'decorative_templates', 'topic_templates', 'resource_templates'}:
                values[target.id] = ast.literal_eval(node.value)
    return values


class Reviewer:
    def __init__(self, policy_path=POLICY):
        self.policy = read(policy_path)
        required = {'same_topic_without_direct_help': 2, 'same_artifact_archive_location': 3,
                    'task_specific_reference_link': 3, 'same_artifact_file_format_information': 3}
        if any(self.policy.get('confirmed_user_rules', {}).get(name) != value for name, value in required.items()):
            raise ValueError('review implementation must be updated for the changed user policy')
        self.definitions = source_definitions()
        self.templates = authored_templates()
        self.kinds = {entry[4]: entry[1] for entry in self.definitions['BULK_EVENTS']}
        for event, phrases in self.definitions['TRAINING_EVENT_PHRASINGS'].items():
            self.kinds.update({phrase: event for phrase in phrases})
        for name in ('unrelated_templates', 'decorative_templates', 'topic_templates', 'resource_templates'):
            self.kinds.update({phrase: name for phrase in self.definitions[name]})
        self.tasks = []
        for index, (domain, process, tool, text) in enumerate(self.definitions['TRAINING_EXPANSION_DOMAINS']):
            for entry in text.split(';'):
                goal, item, field = entry.split('|')
                self.tasks.append({'domain': domain, 'domain_index': index, 'process': process,
                                   'goal': goal, 'item': item, 'field': field})
        self._task_terms = [(normalize(t['goal']), normalize(t['item'])) for t in self.tasks]
        self.urgent = {body: (family, goal) for (family, body), goal in zip(
            self.definitions['TRAINING_URGENT_SCENARIOS'], self.definitions['TRAINING_URGENT_GOALS'])}

    @lru_cache(maxsize=50000)
    def targets(self, text):
        text = normalize(text)
        targets = {i for i, (goal, item) in enumerate(self._task_terms) if goal in text or item in text}
        if 'shopsearch' in text:
            targets.update(i for i, task in enumerate(self.tasks) if task['goal'] == '검색 결과 정렬 검사')
        return targets

    @lru_cache(maxsize=50000)
    def matching_template(self, body):
        return next((t for t, p in self.templates if p.fullmatch(body)), None)

    def decide(self, row):
        n, c = row['state']['notification'], row['state']['context']
        body = core_body(n['body'])
        manual = self.policy['manual_cases'].get(row['id'])
        if manual:
            if key(json.loads(input_text(row['state'], 'relevance'))) != manual['input_sha256']:
                return None, 'manual_input_changed', '개별 검수한 입력과 달라 재검수 필요', 'unresolved'
            method = 'user_case_confirmation' if manual.get('human_reviewed') else 'individual_case_read'
            return manual['score'], 'individual_case', manual['reason'], method
        if body in self.urgent:
            family, goal = self.urgent[body]
            if normalize(goal) in normalize(c['window_title']):
                return 3, family, '현재 창이 해당 안전·필수 대응의 작업 대상을 명시함; 긴급성과 별개로 관련성 확인', 'authored_scenario_context_check'
            if not c['window_title'] and any(normalize(goal) in normalize(w['window_title']) for w in c['recent_windows']):
                return 3, family, '현재 창이 비어 있으나 최근 창에 정확한 대응 작업 근거가 있음', 'authored_scenario_context_check'
            if not c['window_title'] and not c['recent_windows']:
                return 1, family, '긴급한 알림이더라도 현재 작업 정보가 전혀 없으므로 관련도 근거 없음', 'authored_scenario_context_check'
            return None, family, '검수한 안전 시나리오와 다른 창 조합이므로 개별 대조 필요', 'unresolved'
        template = self.matching_template(body)
        if template is None:
            return None, 'unmatched', '검수한 문장 유형이나 개별 예외에 속하지 않음', 'unresolved'
        kind = self.kinds[template]
        current = self.targets(c['window_title'])
        history = set().union(*(self.targets(w['window_title']) for w in c['recent_windows'])) if c['recent_windows'] else set()
        body_targets = self.targets(body)
        target_evidence = 'explicit_goal_or_item'
        if not body_targets and '{field}' in template:
            # Field-only jokes need both an actual field name and a domain cue; do not infer from app.
            body_targets = {i for i, t in enumerate(self.tasks)
                            if normalize(t['field']) in normalize(body) and
                            normalize(t['domain']) in normalize(n.get('title', '') + n.get('sender', ''))}
            target_evidence = 'field_and_notification_domain'
        if kind == 'decorative_templates':
            domains = {index for index, resources in enumerate(self.definitions['LOW_SCORE_RESOURCES'])
                       if normalize(resources[0]) in normalize(body)}
            body_targets = {i for i, t in enumerate(self.tasks) if t['domain_index'] in domains}
            target_evidence = 'decorative_topic_domain'
        if kind == 'resource_templates':
            return None, kind, '일반 양식이 현재 구체적 행동에 얼마나 유용한지 별도 확인 필요', 'unresolved'
        if not body_targets:
            return None, kind, '문장 틀은 같지만 작업 대상 치환을 입력에서 확인하지 못함', 'unresolved'
        if current:
            related = bool(current & body_targets)
            basis = '현재 창의 작업 대상과 대조; 상충하는 최근 창으로 덮어쓰지 않음'
        elif not c['window_title']:
            if not c['recent_windows']:
                return 1, kind, '현재 창·최근 창 모두 작업 정보가 없음', 'template_substitution_and_context_check'
            if not history:
                return None, kind, '최근 창에도 확인 가능한 작업 대상이 없어 개별 대조 필요', 'unresolved'
            related = bool(history & body_targets)
            basis = '현재 창은 비어 있으므로 최근 창의 구체적 작업 대상을 보조 근거로 사용'
        elif history & body_targets:
            related = True
            basis = '제목만으로 작업을 식별하기 어려운 현재 창을 최근 창의 구체적 단서로 보완'
        else:
            return None, kind, '현재 창을 특정 작업으로 해석하지 못함; 프로그램 종류만으로 라벨을 추정하지 않음', 'unresolved'
        if not related:
            return 1, kind, basis + '; 알림의 대상과 다른 작업', 'template_substitution_and_context_check'
        if kind in TOPIC_EVENTS or kind in {'topic_templates', 'decorative_templates'}:
            return 2, kind, basis + '; 같은 대상·주제의 잡담 또는 선택적 꾸미기이며 직접 도움 불분명', 'template_substitution_and_context_check'
        if kind == 'unrelated_templates':
            return None, kind, '다른 작업용으로 생성한 문장이 현재 작업과 겹쳐 생성 가정 재검수 필요', 'unresolved'
        if target_evidence != 'explicit_goal_or_item':
            return None, kind, '직접 유용한 알림으로 분류하려면 정확한 대상 근거를 추가 검수해야 함', 'unresolved'
        return 3, kind, basis + '; 같은 작업 대상의 요청·자료·보관 위치·참고 링크·파일 정보로 유용', 'template_substitution_and_context_check'


def review(folder=OUTPUT, batch='full-pass-01', policy_path=POLICY):
    folder = Path(folder).resolve()
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', batch):
        raise ValueError('invalid batch name')
    output = folder / 'review' / batch
    if output.exists():
        raise ValueError('choose a new review batch; prior decisions are preserved')
    parent = read(folder / 'manifest.json')
    source = folder / 'relevance/retained.jsonl'
    expected = parent['files']['relevance/retained.jsonl']['sha256']
    if digest(source) != expected:
        raise ValueError('curated source hash changed')
    protected = {**parent['protected_hashes'], str(source): expected,
                 str(Path(policy_path).resolve()): digest(policy_path), str(GENERATOR): digest(GENERATOR)}
    if any(digest(Path(path)) != h for path, h in protected.items()):
        raise ValueError('protected source changed')
    reviewer = Reviewer(policy_path)
    records, decisions, proposed = load_lines(source), [], []
    for row in records:
        score, rule, reason, method = reviewer.decide(row)
        status = 'needs_user_review' if score is None else ('proposed_keep' if score == row['score'] else 'proposed_change')
        decision = {'id': row['id'], 'group_id': row['group_id'], 'old_score': row['score'],
                    'proposed_score': score, 'status': status, 'rule': rule, 'reason': reason,
                    'method': method, 'input_sha256': key(json.loads(input_text(row['state'], 'relevance'))),
                    'human_reviewed': method == 'user_case_confirmation'}
        decisions.append(decision)
        changed = deepcopy(row)
        if score is not None:
            changed['score'] = score
        changed['review_status'] = status
        changed['review'] = decision
        changed['training_eligible'] = False
        proposed.append(changed)
    output.mkdir(parents=True)
    write(output / 'policy-snapshot.json', reviewer.policy)
    write_lines(output / 'decisions.jsonl', decisions)
    write_lines(output / 'changes.jsonl', [d for d in decisions if d['status'] == 'proposed_change'])
    write_lines(output / 'unresolved.jsonl', [d for d in decisions if d['status'] == 'needs_user_review'])
    write_lines(output / 'proposed.jsonl', proposed)
    if any(digest(Path(path)) != h for path, h in protected.items()):
        raise ValueError('source changed during review')
    summary = {'purpose': 'Complete template-assisted pass with individually read exceptions; not a human-certified dataset',
               'created_date': created_date(), 'source_rows': len(records), 'decision_rows': len(decisions),
               'coverage': dict(Counter(d['status'] for d in decisions)),
               'methods': dict(Counter(d['method'] for d in decisions)),
               'changes_by_rule': dict(Counter(d['rule'] for d in decisions if d['status'] == 'proposed_change')),
               'unresolved_by_rule': dict(Counter(d['rule'] for d in decisions if d['status'] == 'needs_user_review')),
               'protected_hashes': protected, 'original_inputs_unchanged': True, 'training_ready': False,
               'human_reviewed_rows': sum(d['human_reviewed'] for d in decisions),
               'policy_snapshot_sha256': digest(output / 'policy-snapshot.json'),
               'limitations': ['Template-level semantic assessment plus row-wise target/context validation, not 9813 independent human readings.',
                               'App/history distribution bias remains a separate dataset-construction problem.',
                               'No model fitting or deployment performed.']}
    write(output / 'manifest.json', summary)
    write(folder / 'review/current-review.json', {'purpose': 'Pointer to the latest full relevance review',
          'created_date': created_date(), 'batch': batch,
          'source_sha256': expected, 'manifest': str((output / 'manifest.json').relative_to(folder)),
          'proposed': str((output / 'proposed.jsonl').relative_to(folder)),
          'proposed_sha256': digest(output / 'proposed.jsonl'), 'training_ready': False})
    print(json.dumps({k: summary[k] for k in ('source_rows', 'coverage', 'methods', 'changes_by_rule', 'unresolved_by_rule')}, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--batch', default='full-pass-01')
    args = parser.parse_args()
    review(args.folder, args.batch)
