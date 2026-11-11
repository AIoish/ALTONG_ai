"""용도: 원문을 유지한 채 다양한 단문·맥락·알림 앱에서 관련도 모델의 점수 변화를 비교한다.
생성일: 2026-11-12
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path

from filtering_training.common.artifact_io import created_date, digest, load_lines, write
from filtering_training.common.encoder_input import encode_checked, input_text
from filtering_training.modeling.relevance_only import load_run, predictions
from filtering_training.preparation.curate_relevance import key, write_lines


# Display categories are not equivalence rules or input-normalization rules.
EXPRESSIONS = {
    'single_jamo': ('ㅇ', 'ㅁ', 'ㄱ', 'ㄴ', 'ㅋ', 'ㅠ', 'ㅜ', 'ㅅ'),
    'acknowledgment': ('ㅇㅇ', 'ㅇㅋ', 'ㅇㅋㅇㅋ', '네', '넵', '넹', '예', '응', '알겠어', '오케이', 'ok', 'OK'),
    'rejection': ('ㄴㄴ', '아니', '아뇨', '싫어', '안 돼', '취소'),
    'laughter': ('ㅋㅋ', 'ㅎㅎ', 'ㅋㅋㅋ', '하하'),
    'emotion': ('ㅠㅠ', 'ㅜㅜ', '헐', '와', '아', '음'),
    'punctuation': ('?', '??', '!', '…', '...', ''),
    'emoji': ('👍', '👎', '😂', '😭', '이모티콘을 보냈습니다.'),
    'numeric': ('1', '0'),
}
APP_BODIES = {'ㅇ', 'ㅁ', '네', '아니', 'ㅋㅋ', 'ㅠㅠ', '👍', '취소'}
APPS = ('KakaoTalk.exe', 'Slack', 'Discord.exe')


def authored_contexts():
    code = ('Code.exe', 'invoice_table.py — 정산 도구 — Visual Studio Code')
    design = ('Illustrator.exe', '행사 포스터 — 출력 여백 조정.ai')
    travel = ('chrome.exe', '오사카 숙소 무료 취소 조건 비교')
    specs = [
        ('project_code', code, '정산 도구 개발', [], 2),
        ('generic_code', code, '새 메시지', [], 1),
        ('other_current', travel, '정산 도구 개발', [code], 1),
        ('ambiguous_with_history', ('Code.exe', 'main.py'), '정산 도구 개발', [code], 2),
        ('ambiguous_without_history', ('Code.exe', 'main.py'), '정산 도구 개발', [], None),
        ('approval_target', ('Code.exe', '정산 도구 배포 승인 대기'), '정산 도구 배포 승인', [], None),
        ('project_design', design, '행사 포스터 작업', [], 2),
        ('generic_design', design, '새 메시지', [], 1),
    ]
    contexts = []
    for identifier, current, title, history, neutral in specs:
        state = {'notification': {'app_name': APPS[0], 'sender': '협업 담당', 'title': title,
                                  'body': '', 'timestamp': '2026-11-12T00:00:00Z'},
                 'context': {'active_process': current[0], 'window_title': current[1],
                             'recent_windows': [{'app_name': p, 'window_title': t} for p, t in history]}}
        controls = []
        if identifier in ('project_code', 'generic_code', 'ambiguous_with_history'):
            controls = [('정산 도구의 합계 중복 오류 수정본 올렸어. 재현 테스트도 통과했어.', 3),
                        ('요즘 정산 도구 만드는 사람들 정말 많더라ㅋㅋ', 2), ('이번 주말에 등산 갈 사람?', 1)]
        elif identifier in ('project_design', 'generic_design'):
            controls = [('행사 포스터 출력 여백을 늘린 파일 보냈어. 잘림 없이 인쇄돼.', 3),
                        ('요즘 행사 포스터 글씨가 점점 커지는 것 같아', 2), ('이번 주말에 등산 갈 사람?', 1)]
        elif identifier == 'other_current':
            controls = [('정산 도구의 합계 중복 오류 수정본 올렸어.', 1),
                        ('비교하던 오사카 숙소 B는 전날까지 취소 수수료가 없대.', 3)]
        contexts.append({'id': identifier, 'scope': 'authored_development_probe', 'state': state,
                         'neutral_score_proposal': neutral, 'controls': controls,
                         'confirmed_examples': ({'네': 2, '아니': 2} if identifier == 'approval_target' else
                            {body: 1 for body in ('ㅇㅇ', 'ㅇㅋ', '네', '아니', 'ㅋㅋ', 'ㅠㅠ')} if identifier == 'project_code' else {}),
                         'confirmed_app': APPS[0] if identifier == 'project_code' else None})
    return contexts


def make_probes(contexts):
    if len({c['id'] for c in contexts}) != len(contexts):
        raise ValueError('duplicate context ID')
    probes = []
    for context in contexts:
        bodies = [(category, body, None) for category, expressions in EXPRESSIONS.items() for body in expressions]
        bodies += [('explicit_control', body, score) for body, score in context.get('controls', [])]
        for category, body, control in bodies:
            original_app = context['state']['notification']['app_name']
            apps = tuple(dict.fromkeys((original_app, *APPS))) if body in APP_BODIES else (original_app,)
            for app in apps:
                state = deepcopy(context['state'])
                state['notification'].update(body=body, app_name=app)
                score, status = None, 'policy_pending'
                if control is not None:
                    score, status = control, 'rule_applied_proposal'
                elif body in context.get('confirmed_examples', {}) and context.get('confirmed_app') in (None, app):
                    score, status = context['confirmed_examples'][body], 'user_confirmed_example'
                elif body in ('ㅇ', 'ㅁ') and context.get('neutral_score_proposal') is not None:
                    score, status = context['neutral_score_proposal'], 'rule_applied_proposal'
                if score is not None and (type(score) is not int or score not in (1, 2, 3)):
                    raise ValueError('expected relevance 1-3')
                fingerprint = key(json.loads(input_text(state, 'relevance')))
                probes.append({'id': 'reaction_' + key([context['id'], category, body, app])[:20],
                    'context_id': context['id'], 'scope': context['scope'], 'category': category,
                    'body': body, 'app': app, 'state': state, 'input_sha256': fingerprint,
                    'reference_score': score, 'label_status': status,
                    'minimal_pair_comparable': body in ('ㅇ', 'ㅁ') and context.get('neutral_score_proposal') is not None})
    if len({p['id'] for p in probes}) != len(probes):
        raise ValueError('duplicate probe ID')
    return probes


def summarize(probes, scores):
    if len(scores) != len(probes) or any(type(s) is not int or s not in (1, 2, 3) for s in scores):
        raise ValueError('one valid score per probe required')
    labels, categories, app_groups, minimal = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for probe, score in zip(probes, scores):
        labels[probe['label_status']].append((probe, score))
        categories[(probe['context_id'], probe['category'])].append(score)
        app_groups[(probe['context_id'], probe['body'])].append((probe['app'], score))
        if probe['minimal_pair_comparable']:
            minimal[(probe['context_id'], probe['app'])].append((probe['body'], score))
    scopes = {p['context_id']: p['scope'] for p in probes}
    pairs = [{'context_id': ctx, 'scope': scopes[ctx], 'app': app, 'scores': dict(values),
              'same_score': len({s for _, s in values}) == 1}
             for (ctx, app), values in minimal.items() if {b for b, _ in values} == {'ㅇ', 'ㅁ'}]
    app_variation = [{'context_id': ctx, 'body': body, 'scores': dict(values)}
                     for (ctx, body), values in app_groups.items() if len(values) > 1 and len({s for _, s in values}) > 1]
    return {'rows': len(probes), 'score_counts': dict(Counter(scores)),
            'scope_counts': dict(Counter(p['scope'] for p in probes)),
            'reference_agreement': {status: {'rows': len(values),
                'matching': sum(p['reference_score'] == s for p, s in values)}
                for status, values in labels.items() if status != 'policy_pending'},
            'unjudged_rows': len(labels['policy_pending']), 'minimal_pairs': pairs,
            'minimal_pair_disagreements': sum(not p['same_score'] for p in pairs),
            'app_variant_groups': sum(len(v) > 1 for v in app_groups.values()),
            'app_variant_disagreements': app_variation,
            'category_score_spreads': [{'context_id': ctx, 'category': category, 'min': min(values), 'max': max(values)}
                                      for (ctx, category), values in categories.items()],
            'interpretation': 'Category spreads are observations, not errors or equivalence judgments.'}


def write_first_review(output, probes):
    selected = []
    for body in ('ㅇㅇ', 'ㅇㅋ', '네', '아니', 'ㅋㅋ', 'ㅠㅠ'):
        selected.append(next(p for p in probes if p['context_id'] == 'project_code' and p['body'] == body and p['app'] == APPS[0]))
    state = selected[0]['state']
    lines = ['<!-- 용도: 모델 예측 없이 프로젝트 방 단문 6개의 개별 검수 상태를 기록한다.',
             '생성일: ' + created_date() + ' -->', '', '# 단문 개별 검수 — 1차', '',
             '앞선 대화는 입력에 없다. 아래 여섯 알림은 본문 외 조건이 모두 같다.', '',
             '- 현재 창: ' + state['context']['active_process'] + ' / ' + state['context']['window_title'],
             '- 알림 앱·발신자: ' + state['notification']['app_name'] + ' / ' + state['notification']['sender'],
             '- 알림 제목(프로젝트 방): ' + state['notification']['title'], '- 최근 창: 없음', '',
             '| 카드 | 본문 | 관련도 1~3 또는 보류 |', '| --- | --- | --- |']
    for card, p in zip('ABCDEF', selected):
        score = p['reference_score'] if p['label_status'].startswith('user_confirmed') else ''
        lines.append(f"| {card} | {p['body']} | {score} |")
    lines += ['', '답변 예: `A=2, B=2, C=보류, D=2, E=1, F=2` — 형식 예시이며 제안 점수가 아니다.',
              '알림 제목이 구체적인 승인 대상인 별도 맥락의 네·아니=2점 결정은 이 프로젝트 방 사례에 자동 적용하지 않는다.']
    (output / 'first-review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(output / 'first-review-inputs.json', {'purpose': 'Bind six review cards and confirmed scores to exact model inputs',
        'created_date': created_date(), 'cards': {card: {'probe_id': p['id'], 'input_sha256': p['input_sha256'],
            'score': p['reference_score'] if p['label_status'].startswith('user_confirmed') else None}
            for card, p in zip('ABCDEF', selected)}})


def audit(runs, output, *, extra_contexts=None, device='cuda'):
    output = Path(output)
    if output.exists():
        raise ValueError('preserve earlier audit; choose a new folder')
    contexts = authored_contexts()
    source_hash = None
    if extra_contexts:
        source_hash = digest(extra_contexts)
        contexts += load_lines(extra_contexts)
    probes = make_probes(contexts)
    output.mkdir(parents=True)
    write_lines(output / 'probes.jsonl', probes)
    reports = {}
    for number, run in enumerate(runs, 1):
        model, tokenizer, config = load_run(run, device=device)
        encoded, lengths = encode_checked(tokenizer, [input_text(p['state'], 'relevance') for p in probes])
        scores, probabilities = predictions(model, tokenizer, encoded, device=device, batch_size=4)
        report = summarize(probes, scores)
        name = f'model-{number}'
        report.update(run=str(Path(run).resolve()), checkpoint_sha256=config['checkpoint_sha256'], max_tokens=max(lengths))
        reports[name] = report
        write(output / f'{name}.summary.json', report)
        write_lines(output / f'{name}.predictions.jsonl', [dict(p, predicted_score=s, probabilities_1_2_3=q)
                    for p, s, q in zip(probes, scores, probabilities)])
        del model
    with (output / 'review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['probe_id', 'context_id', 'scope', 'category', 'app', 'body', 'notification_title',
                         'current_window', 'recent_windows', 'reference_score', 'label_status', 'user_score', 'note'])
        for p in probes:
            c, n = p['state']['context'], p['state']['notification']
            writer.writerow([p['id'], p['context_id'], p['scope'], p['category'], p['app'], p['body'], n['title'],
                c['window_title'], json.dumps(c['recent_windows'], ensure_ascii=False),
                p['reference_score'], p['label_status'], '', ''])
    if extra_contexts and digest(extra_contexts) != source_hash:
        raise ValueError('source contexts changed')
    write_first_review(output, probes)
    write(output / 'manifest.json', {'purpose': 'Raw short-expression diagnostic, not independent evaluation or training',
        'created_date': created_date(), 'normalization_applied': False, 'rows': len(probes), 'contexts': len(contexts),
        'context_source_sha256': source_hash, 'models': {k: {'run': r['run'], 'checkpoint_sha256': r['checkpoint_sha256']} for k, r in reports.items()},
        'pending_labels_not_scored': True, 'user_policy': 'Project-code review A-F score 1 on the reviewed app; other cases remain individually scoped; approval-title yes/no without dialogue score 2.',
        'files': {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        'limitations': ['Includes training-side captures and authored development contexts.',
                        'No prior conversation is available.', 'Only explicit reference examples have labels; other expressions await review.']})
    print(json.dumps({name: {'rows': r['rows'], 'unjudged_rows': r['unjudged_rows'],
        'minimal_pair_disagreements': r['minimal_pair_disagreements'],
        'app_variant_disagreements': len(r['app_variant_disagreements']), 'reference_agreement': r['reference_agreement']}
        for name, r in reports.items()}, ensure_ascii=False, indent=2))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--contexts', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    audit(args.run, args.output_dir, extra_contexts=args.contexts, device=args.device)
