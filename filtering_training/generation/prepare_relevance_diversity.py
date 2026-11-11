"""용도: 앱·창 개수 대신 현재 작업과 알림의 관계를 구분하는 관련도 전용 검수 초안을 만든다.
생성일: 2026-11-08
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import INPUT_SCHEMA, RECENT_WINDOWS_POLICY, input_text
from filtering_training.preparation.curate_relevance import OUTPUT, body_group, key, profile, write_lines

# Hand-authored scenarios, not a Cartesian multiplication of interchangeable nouns.
SCENARIOS = (
    {'family': 'login_session', 'process': 'Code.exe',
     'window': 'auth_session_test.py — login-service — Visual Studio Code',
     'reference': '세션 만료 오류 #184 - login-service - Chrome',
     'title': '로그인 수정', 'sender': '민수',
     'messages': {3: '로그인 수정본 배포했어. 세션 만료 재현도 통과했음.',
                  2: '로그인 버튼은 왜 다 오른쪽 위에 있을까ㅋㅋ',
                  1: '주말에 볼 영화 추천 좀'}},
    {'family': 'poster_type', 'process': 'Illustrator.exe',
     'window': '여름행사 포스터 — 제목 글꼴 교체 - Illustrator',
     'reference': '여름행사 포스터 글꼴 사용권 확인 - Chrome',
     'title': '포스터 작업', 'sender': '수빈',
     'messages': {3: '포스터 제목 글꼴 바꾼 파일 보냈어. 줄바꿈도 다시 맞췄음.',
                  2: '요즘 포스터는 글씨를 크게 넣는 게 유행인가 봐',
                  1: '우리 강아지 산책하다가 또 진흙 밟음ㅋㅋ'}},
    {'family': 'shoe_price', 'process': 'chrome.exe',
     'window': '러닝화 무게와 최종 결제금액 비교 - Chrome',
     'reference': '후보 러닝화 무게 측정 후기 - Chrome',
     'title': '운동방', 'sender': '지훈',
     'messages': {3: '비교하던 러닝화 A, 배송비 합치면 89,000원이래.',
                  2: '러닝화 신상 색깔들 진짜 화려해졌더라',
                  1: '회의실 빔프로젝터 리모컨 어디 있어?'}},
    {'family': 'statistics_report', 'process': 'WINWORD.EXE',
     'window': '통계 과제 — 회귀분석 결과 정리 - Word',
     'reference': '회귀표 표준오차 해석 — 강의자료 - Chrome',
     'title': '통계 과제', 'sender': '유진',
     'messages': {3: '과제 회귀표의 표준오차 열 수정했어. 수정본 확인해 줘.',
                  2: '통계 수업에서 들은 평균의 함정 얘기 재밌더라',
                  1: '엘리베이터 정기 점검이 금요일이래'}},
    {'family': 'hotel_cancellation', 'process': 'chrome.exe',
     'window': '오사카 숙소 취소 규정 비교 - Chrome',
     'reference': '숙소 B 무료취소 조건 - Chrome',
     'title': '여행 준비', 'sender': '지수',
     'messages': {3: '비교 중인 오사카 숙소 B는 전날까지 무료취소래.',
                  2: '오사카 여행 브이로그 보니까 또 가고 싶다',
                  1: '프린터 토너 주문해뒀음'}},
    {'family': 'leak_estimate', 'process': 'EXCEL.EXE',
     'window': '욕실 누수 수리 견적 비교.xlsx - Excel',
     'reference': '누수 수리 업체별 자재비 포함 여부 - Notion',
     'title': '집 수리', 'sender': '현우',
     'messages': {3: '누수 수리 견적 18만 원으로 왔어. 자재비 포함이래.',
                  2: '누수 얘기하니까 작년에 천장 새던 날 생각난다',
                  1: '택배 문 앞에 뒀대'}},
    {'family': 'budget_duplicates', 'process': 'EXCEL.EXE',
     'window': '11월 가계부 — 식비 중복 결제 확인 - Excel',
     'reference': '11월 카드 명세서 - Chrome',
     'title': '가계부', 'sender': '하린',
     'messages': {3: '가계부 식비에 같은 결제가 두 번 들어갔어.',
                  2: '요즘 다들 식비 얼마나 쓰나 궁금하네',
                  1: '오늘 밤 유성우 보인대'}},
    {'family': 'interview_noise', 'process': 'Adobe Premiere Pro.exe',
     'window': '인터뷰 영상 — 2번 클립 잡음 제거 - Premiere Pro',
     'reference': '인터뷰 2번 녹음 잡음 구간 메모 - Notion',
     'title': '인터뷰 편집', 'sender': '다은',
     'messages': {3: '인터뷰 2번 파일 잡음 제거본 올림',
                  2: '영상 편집하는 사람들 밤샘 진짜 많이 하더라',
                  1: '동네 도서관 휴관일 바뀌었대'}},
)

APPS = ('Slack', 'Teams', 'KakaoTalk.exe')
DISTRACTORS = (
    {'app_name': 'chrome.exe', 'window_title': '주말 날씨 - Chrome'},
    {'app_name': 'explorer.exe', 'window_title': '다운로드 - 파일 탐색기'},
    {'app_name': 'Notion.exe', 'window_title': '장보기 목록 - Notion'},
)


def make_drafts():
    rows, counters = [], Counter()

    def add(scenario, score, body, kind, *, current=None, history=None, title=None, group=None):
        ordinal = counters[score]
        counters[score] += 1
        process, window = current or (scenario['process'], scenario['window'])
        # Mix unrelated history length within every score, independent of notification app.
        history = deepcopy(list(DISTRACTORS[:ordinal % 4])) if history is None else deepcopy(history)
        index = len(rows)
        timestamp = f'2026-11-{2 + (index * 3) % 6:02d}T{(index * 7) % 24:02d}:{(index * 13) % 60:02d}:00Z'
        state = {'notification': {'app_name': APPS[(ordinal // 4 + ordinal) % 3],
                                 'sender': scenario['sender'], 'title': title or scenario['title'],
                                 'body': body, 'timestamp': timestamp},
                 'context': {'active_process': process, 'window_title': window,
                             'recent_windows': history}}
        rationale = {1: '현재 작업에 도움이 되는 관계를 입력에서 확인할 수 없음',
                     2: '현재 작업과 같은 주제이지만 직접적인 도움은 불분명',
                     3: '현재 작업 대상에 직접 유용한 결과·수정·비교 정보'}[score]
        if kind == 'current_history_conflict':
            rationale = '현재 창은 다른 작업을 명확히 가리키므로 과거 작업의 결과가 현재 작업을 덮어쓰지 않음'
        if kind == 'uninformative_character':
            rationale = '본문·발신자·제목에 현재 작업과 연결할 근거가 없음; 대화 이력은 입력에 없음'
        if kind == 'generic_current_without_history' and score == 2:
            rationale = 'main.py는 개발 분야의 단서이므로 로그인 개발 알림과 일부 관련성은 있으나 정확한 현재 작업까지 확인되지는 않음; 사용자 확인 2점'
        rows.append({'id': f'diverse_{index + 1:03d}', 'group_id': group or 'scenario_' + scenario['family'],
                     'split': 'unassigned', 'state': state, 'score': score,
                     'review_status': 'ai_proposed_unreviewed', 'reason': rationale,
                     'scenario': scenario['family'], 'case_kind': kind,
                     'provenance': {'origin': 'authored_synthetic', 'real_user_data': False}})

    for index, scenario in enumerate(SCENARIOS):
        for score in (3, 2, 1):
            add(scenario, score, scenario['messages'][score], 'same_current_different_notification')
        other = SCENARIOS[(index + 3) % len(SCENARIOS)]
        add(scenario, 1, scenario['messages'][3], 'current_history_conflict',
            current=(other['process'], other['window']),
            history=[{'app_name': scenario['process'], 'window_title': scenario['window']}])
        rows[-1]['state']['notification'] = deepcopy(rows[-4]['state']['notification'])
        rows[-1]['paired_with'] = rows[-4]['id']
    # These pairs deliberately keep every input field except the meaningless character identical.
    for scenario in (SCENARIOS[0], SCENARIOS[4]):
        for character in ('ㅇ', 'ㅁ'):
            add(scenario, 1, character, 'uninformative_character', history=[], title='새 메시지',
                group='uninformative_characters')
        first, second = rows[-2:]
        second['state'] = deepcopy(first['state'])
        second['state']['notification']['body'] = 'ㅁ'
    for scenario, generic in ((SCENARIOS[0], 'main.py — Visual Studio Code'),
                              (SCENARIOS[3], '문서1 - Word')):
        history = [{'app_name': scenario['process'], 'window_title': scenario['window']},
                   {'app_name': 'chrome.exe', 'window_title': scenario['reference']}]
        add(scenario, 3, scenario['messages'][3], 'history_resolves_ambiguous_current',
            current=(scenario['process'], generic), history=history)
        without_history_score = 2 if scenario['family'] == 'login_session' else 1
        add(scenario, without_history_score, scenario['messages'][3], 'generic_current_without_history',
            current=(scenario['process'], generic), history=[])
        rows[-1]['state']['notification'] = deepcopy(rows[-2]['state']['notification'])
        rows[-1]['paired_with'] = rows[-2]['id']
    # Twelve explicit controlled pairs, not a full app/history Cartesian expansion.
    for score in (1, 2, 3):
        eligible = [r for r in rows if r['score'] == score and r['case_kind'] == 'same_current_different_notification']
        for i, original in enumerate(eligible[:4]):
            variant = deepcopy(original)
            variant['id'] = original['id'] + ('_app' if i % 2 == 0 else '_history')
            variant['paired_with'] = original['id']
            variant['case_kind'] = 'app_invariance' if i % 2 == 0 else 'irrelevant_history_invariance'
            if i % 2 == 0:
                app = original['state']['notification']['app_name']
                variant['state']['notification']['app_name'] = APPS[(APPS.index(app) + 1) % len(APPS)]
            else:
                count = len(original['state']['context']['recent_windows'])
                variant['state']['context']['recent_windows'] = deepcopy(list(DISTRACTORS[:(count + 2) % 4]))
            rows.append(variant)
    return rows


def validate_drafts(rows):
    ids, seen = set(), {}
    for row in rows:
        if row['id'] in ids or type(row['score']) is not int or row['score'] not in (1, 2, 3):
            raise ValueError('duplicate ID or invalid relevance score')
        ids.add(row['id'])
        payload = json.loads(input_text(row['state'], 'relevance'))
        c = row['state']['context']
        if len(c['recent_windows']) > 3 or 'recent_processes' in c:
            raise ValueError('incorrect previous-windows schema')
        if payload['recent_windows'] != c['recent_windows']:
            raise ValueError('draft contained the current window in history')
        signature = key(payload)
        if signature in seen:
            raise ValueError('duplicated full model input')
        seen[signature] = row['score']
        if row['review_status'] != 'ai_proposed_unreviewed' or row['split'] != 'unassigned':
            raise ValueError('new proposals must not be silently approved or split')
    by_body = {}
    for row in rows:
        group = body_group(row['state']['notification']['body'])
        if group in by_body and by_body[group] != row['group_id']:
            raise ValueError('the same body cannot be split across scenario groups')
        by_body[group] = row['group_id']


def review_markdown(rows):
    lines = ['<!-- 용도: 관련도 다양성 초안의 입력·제안 점수·근거를 검수한다.',
             f'생성일: {created_date()} -->', '', '# 관련도 다양성 초안 검수', '',
             '합성 제안이며 실제 사용자 데이터가 아니다. 점수·분할은 아직 확정되지 않았고 학습하지 않는다.',
             '같은 주제의 잡담·일반 소식은 사용자 확인에 따라 2점으로 제안한다.',
             '사용자 확인 기준: 현재 창이 명확하면 우선하며, 모호하면 최근 창의 구체적 단서로 보완할 수 있다.',
             '사용자 확인 기준: main.py와 로그인 개발 알림은 최근 창이 없어도 개발 분야가 겹치므로 2점이다.',
             '완전히 근거가 없는 경우의 1점과 다른 분야의 모호한 창 사례는 별도 검수 대상이다.', '']
    for row in rows:
        n, c = row['state']['notification'], row['state']['context']
        lines.extend([f"## {row['id']} · 제안 {row['score']}점 · {row['case_kind']}", '',
                      f"- 알림: {n['app_name']} / {n['sender']} / {n['title']}",
                      f"- 본문: {n['body']}", f"- 현재 창: {c['active_process']} / {c['window_title']}",
                      '- 최근 창: ' + (' → '.join(w['app_name'] + ' / ' + w['window_title'] for w in c['recent_windows']) or '(없음)'),
                      f"- 근거: {row['reason']}", f"- 묶음: {row['group_id']} (분할 시 함께 이동)", '',
                      '- 검수: [ ] 유지 [ ] 수정 [ ] 제외 / 메모:', ''])
    return '\n'.join(lines)


def prepare(output=OUTPUT, batch_name=None):
    output = Path(output).resolve()
    parent = read(output / 'manifest.json')
    if parent.get('status') != 'review_pending' or parent.get('input_schema') != INPUT_SCHEMA:
        raise ValueError('expected current-schema curation draft')
    if batch_name is not None and not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', batch_name):
        raise ValueError('batch_name must be a simple lowercase name')
    prefix = 'new' if batch_name is None else 'new/' + batch_name
    targets = tuple(prefix + '/' + filename for filename in ('candidates.jsonl', 'review.md', 'manifest.json'))
    if any((output / filename).exists() for filename in targets):
        raise ValueError('existing proposal files must be preserved')
    for filename, info in parent['files'].items():
        if digest(output / filename) != info['sha256']:
            raise ValueError('curated source file changed')
    protected = dict(parent['protected_hashes'])
    source_dir = Path(parent['source_dir'])
    old_training = {body_group(r['state']['notification']['body'])
                    for r in load_lines(source_dir / 'relevance/train.jsonl')}
    heldout = set()
    for path in (source_dir / 'evaluation').glob('*.jsonl'):
        protected[str(path)] = digest(path)
        heldout.update(body_group(r['notification']['body']) for r in load_lines(path))
    rows = make_drafts()
    validate_drafts(rows)
    if any(body_group(r['state']['notification']['body']) in old_training | heldout for r in rows):
        raise ValueError('new body exactly overlaps training or held-out data')
    if any(digest(Path(path)) != expected for path, expected in protected.items()):
        raise ValueError('protected source changed before generation')
    output.joinpath(prefix).mkdir(parents=True, exist_ok=True)
    write_lines(output / targets[0], rows)
    (output / targets[1]).write_text(review_markdown(rows), encoding='utf-8')
    if any(digest(Path(path)) != expected for path, expected in protected.items()):
        raise ValueError('protected source changed')
    manifest = {'purpose': 'Small relevance-only diversity pilot for user review, not a training set',
                'created_date': created_date(), 'status': 'ai_proposed_unreviewed', 'input_schema': INPUT_SCHEMA,
                'recent_windows_policy': RECENT_WINDOWS_POLICY, 'rows': len(rows), 'profile': profile(rows),
                'case_counts': dict(Counter(r['case_kind'] for r in rows)),
                'policy_confirmation': {'same_topic_without_direct_help': {'score': 2, 'status': 'user_confirmed'},
                                        'insufficient_evidence': {'score': 1, 'status': 'proposal_pending_review'},
                                        'clear_current_over_conflicting_history': {'status': 'user_confirmed'},
                                        'ambiguous_current_with_specific_history': {'status': 'user_confirmed', 'allows_score': 3},
                                        'main_py_without_history_login_notification': {'status': 'user_confirmed', 'score': 2,
                                            'scope': 'Development-domain cue, not a blanket score for every ambiguous window'}},
                'real_user_data_received': False, 'training_used': False, 'splits_assigned': False,
                'source_files_unchanged': True, 'protected_hashes': protected,
                'files': {name: {'sha256': digest(output / name)} for name in targets[:2]},
                'limitations': ['Eight authored scenario groups are not a representative final benchmark.',
                                'Group whole scenarios and variants before making any evaluation split.',
                                'Labels and app/history variations require semantic review.',
                                'Do not mix unreviewed proposals into approved training data.']}
    write(output / targets[2], manifest)
    write(output / 'review/current-pilot.json', {'purpose': 'Pointer to the latest locally generated pilot',
          'created_date': created_date(), 'candidates': targets[0], 'review': targets[1], 'manifest': targets[2],
          'candidates_sha256': digest(output / targets[0]), 'status': 'ai_proposed_unreviewed'})
    print(json.dumps({'status': manifest['status'], 'rows': len(rows), 'case_counts': manifest['case_counts']}, ensure_ascii=False))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--batch-name')
    args = parser.parse_args()
    prepare(args.output_dir, args.batch_name)
