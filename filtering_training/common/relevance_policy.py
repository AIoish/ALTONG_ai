"""용도: 대상 없는 반응의 확정 관련도 정책을 원문 변경 없이 데이터·진단에 공통 적용한다.
생성일: 2026-11-15
"""

from copy import deepcopy
import re
import unicodedata

POLICY = 'targetless-reactions-v1'
REACTIONS = frozenset(('네', '넵', '넹', '예', '응', '아니', '아뇨', '알겠어', '오케이', 'ok',
                       '하하', '헐', '와', '아', '음', '좋은데요',
                       '👍', '👎', '😂', '😭', '이모티콘을 보냈습니다.', '(이모티콘 또는 미디어)'))


def targetless(body):
    # Matching is for label policy only. Model inputs always retain the raw text.
    value = unicodedata.normalize('NFC', body).strip().casefold()
    return value in REACTIONS or bool(re.fullmatch(r'[ㅇㅁㄱㄴㅋㅎㅠㅜㅅㅡ]+', value))


def apply_policy(rows, policy=POLICY):
    if policy not in (None, 'legacy', POLICY):
        raise ValueError('unknown relevance label policy')
    result = deepcopy(rows)
    if policy in (None, 'legacy'):
        return result
    for row in result:
        if targetless(row['state']['notification']['body']):
            if row.get('label_policy', {}).get('version') == POLICY:
                if row['score'] != 1:
                    raise ValueError('policy-marked reaction has inconsistent score')
                continue
            old = row['score']
            row['score'] = 1
            row['label_policy'] = {'version': POLICY, 'previous_score': old,
                'basis': 'user_confirmed_rule_applied', 'individually_human_reviewed': False,
                'reason': 'Targetless reaction scores 1 regardless of room or approval title; no prior dialogue supplied.'}
    return result
