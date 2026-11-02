"""용도: 재학습과 분리된 일상 알림·긴급 누락 점검용 합성 평가 100건을 고정한다.
생성일: 2026-10-10
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Labels are authored proposals based on the existing guideline, not human approvals.
# Each entry: group, urgency, relevance, context profile, body.
CASES = [
    ('safety',5,1,'dev','집에 불났어'),
    ('safety',5,1,'empty','부엌에 불 붙었어 빨리 나와'),
    ('safety',5,1,'dev','우리집 거실에서 불길이 보여'),
    ('safety',5,1,'empty','아랫집 불났대 우리도 대피하래'),
    ('safety',5,1,'dev','현관 쪽에 불이 번져서 못 나가겠어'),
    ('safety',5,1,'empty','집에 불낫어 지금 밖으로 나와'),
    ('safety',5,1,'dev','방 안에 연기가 꽉 차서 숨쉬기 힘들어'),
    ('safety',5,1,'empty','보조배터리에서 연기 나고 뜨거워'),
    ('safety',5,1,'dev','복도에서 연기 올라와 애가 안에 있어'),
    ('safety',5,1,'empty','가스 냄새 심한데 밸브가 안 잠겨'),
    ('safety',5,1,'dev','가스관 터졌어 안에서 새는 소리 나'),
    ('safety',5,1,'empty','집 가스 경보기 울려 대피하래'),
    ('safety',5,1,'dev','엄마 쓰러졌어 전화 받아'),
    ('safety',5,1,'empty','아빠가 의식이 없어 119 불렀어'),
    ('safety',5,1,'dev','아이가 뭘 삼켰는데 숨을 못 쉬어'),
    ('safety',5,1,'empty','피가 계속 나고 눌러도 안 멈춰'),
    ('safety',5,1,'dev','동생 응급실로 옮기는 중이야 보호자 연락 필요해'),
    ('safety',5,1,'empty','누가 현관문 부수고 들어오려 해 경찰 불렀어'),
    ('safety',5,1,'dev','모르는 사람이 집 안에 들어왔어 방에 숨어 있어'),
    ('safety',5,1,'empty','콘센트에서 불꽃 튀고 타는 냄새 나'),
    ('safety',5,1,'dev','전선에 사람이 감전돼서 쓰러졌어'),
    ('safety',5,1,'empty','전기장판에서 연기 나고 이불이 타고 있어'),
    ('safety',5,1,'dev','충전하던 배터리 터져서 책상에 불 붙었어'),
    ('safety',5,1,'empty','물이 전등 안으로 들어가고 불꽃이 튀어'),
    ('safety',5,1,'dev','천장 물 샌 게 멀티탭까지 흘러 전기가 튀어'),
    ('safety',5,1,'empty','지하방에 물 차는데 문이 안 열려'),
    ('safety',5,1,'dev','엘리베이터에 갇혔고 사람이 쓰러졌어 구조 요청했어'),
    ('safety',5,1,'empty','화재 대피 중인데 할머니가 안에 남아 있어'),
    ('safety',5,1,'dev','베란다 난간이 무너졌어 아이가 매달려 있어'),
    ('safety',5,1,'empty','아이가 차도로 뛰어나갔어 지금 바로 와줘'),
    ('boundary',4,1,'dev','고객에게 보낼 청구서 계좌번호가 틀렸어. 발송 전에 바꿔줘'),
    ('boundary',4,1,'empty','택배 기사님 문 앞이야. 반품할 박스 어디 있는지 알려줘'),
    ('boundary',4,1,'dev','네 승인이 없어 회의 시작을 못 하고 있어. 참석 여부 확인해줘'),
    ('boundary',4,1,'empty','인쇄 맡긴 책자 수량이 잘못됐어. 인쇄 시작 전에 연락 부탁해'),
    ('boundary',4,1,'dev','발표 순서가 당겨졌어. 다음 발표가 너라 자료 열어둬'),
    ('boundary',4,1,'empty','송금 확인 화면의 수취인이 달라. 보내기 전에 확인해줘'),
    ('boundary',4,1,'dev','배송 출발까지 10분인데 주소에 동호수가 빠졌어. 알려줘'),
    ('boundary',4,1,'empty','접수 창구에서 네 서류 한 장을 기다려. 15분 뒤 접수 끝이야'),
    ('boundary',4,1,'dev','회의실에서 네 답변을 기다리느라 검토가 멈췄어. 선택해줘'),
    ('boundary',4,1,'empty','고객사 배포가 인증서 문제로 막혔어. 담당자가 교체 확인을 기다려'),
    ('boundary',3,1,'dev','다음 주 청구서 계좌번호를 검토해줘. 오늘 저녁까지면 돼'),
    ('boundary',3,1,'empty','내일 반품할 박스 위치를 오늘 중 알려줘. 기사 방문은 내일이야'),
    ('boundary',3,1,'dev','다음 주 회의 참석 여부를 오늘 안에 알려줘. 아직 준비 중이야'),
    ('boundary',3,1,'empty','책자 인쇄 수량을 오늘 중 검토해줘. 인쇄는 다음 주에 시작해'),
    ('boundary',3,1,'dev','내일 발표 자료 한번 봐줘. 오늘 저녁까지 의견 주면 돼'),
    ('boundary',3,1,'empty','다음 주 송금 명단의 수취인 이름을 오늘 중 대조해줘. 송금은 아직 안 해'),
    ('boundary',3,1,'dev','다음 주 배송 주소에 동호수가 빠졌어. 오늘 중 보완 부탁해'),
    ('boundary',3,1,'empty','다음 주 접수할 서류 한 장이 빠졌어. 오늘 저녁까지 보내줘'),
    ('boundary',3,1,'dev','내일 검토할 선택 항목에 오늘 중 의견 부탁해. 지금 진행 중인 검토는 없어'),
    ('boundary',2,1,'empty','인증서 갱신은 다음 주 점검에 잡았어. 서비스와 배포는 정상이고 답변은 필요 없어'),
    ('routine',1,1,'dev','오늘 저녁 뭐 먹지 ㅋㅋ'),
    ('routine',1,1,'empty','고양이 사진 보냈어'),
    ('routine',1,1,'dev','주말 여행 사진 올라왔어 시간 날 때 봐'),
    ('routine',1,1,'empty','새 게임 스킨 나왔대'),
    ('routine',2,1,'dev','편의점 쿠폰 5분 남음. 안 써도 아무 불이익 없어'),
    ('routine',1,1,'empty','운동화 할인 시작! 구매는 선택이에요'),
    ('routine',1,1,'dev','친구가 새 게시물을 올렸습니다'),
    ('routine',1,1,'empty','구독 채널 새 영상 올라왔어'),
    ('routine',1,1,'dev','집중할 때 듣는 음악 추천이야'),
    ('routine',2,1,'empty','택배 문 앞에 놓고 갑니다. 냉장품 아닙니다'),
    ('routine',2,1,'dev','이번 달 통신요금 자동납부 완료. 추가 조치 없음'),
    ('routine',2,1,'empty','사진 백업 끝났어. 실패한 파일 없어'),
    ('routine',2,1,'dev','다음 달 관리비 납부일 안내입니다. 오늘 처리할 일은 없습니다'),
    ('routine',2,1,'empty','앱 업데이트 가능. 나중에 설치해도 돼'),
    ('routine',2,1,'dev','다음 주 주차장 정기 청소 안내. 회신 필요 없음'),
    ('routine',2,1,'empty','내일 비 예보래. 그냥 참고해'),
    ('routine',3,1,'dev','내일 저녁 식당 예약 인원을 오늘 중 알려줘'),
    ('routine',3,1,'empty','다음 주 여행 숙소 후보를 오늘 저녁까지 골라줘. 아직 예약 안 해'),
    ('routine',3,1,'dev','내일 제출할 서류 목록 확인 부탁해. 오늘 중 답 주면 돼'),
    ('routine',3,1,'empty','내일 면접 장소 확인해줘. 오늘 저녁까지 읽으면 돼'),
    ('routine',3,1,'dev','세탁기 수리 방문 날짜 오늘 중 알려줘. 고장으로 인한 위험은 없어'),
    ('routine',3,1,'empty','이번 주 설문을 오늘 중 제출 부탁드립니다. 지금 답하지 않아도 됩니다'),
    ('routine',3,1,'dev','금요일 모임 장소가 바뀌었어. 오늘 안에 확인해줘'),
    ('routine',2,1,'empty','지난번 신청한 자료 도착했어. 급한 확인은 아니야'),
    ('routine',1,1,'dev','오랜만이네 잘 지내? 답은 편할 때 줘'),
    ('relevance',2,5,'dev','auth_session_test.py의 만료 토큰 테스트가 실패한 로그 보냈어. 급한 운영 장애는 아니야'),
    ('relevance',2,4,'dev','로그인 세션 테스트할 때 쓸 만료 시간 경계값 체크리스트야. 필요할 때 봐'),
    ('relevance',1,3,'dev','우리 로그인 테스트 얘기하다 보니 점심 로그인도 필요하겠네 ㅋㅋ'),
    ('relevance',1,2,'dev','개발자 키보드 키캡 할인 중. 테스트 기능은 없고 장식용이야'),
    ('relevance',1,1,'dev','동네 꽃 축제 사진이 올라왔어'),
    ('relevance',2,5,'excel','이번 달 식비 시트의 8일 장보기 영수증 보냈어. 급한 요청은 아니야'),
    ('relevance',2,4,'excel','가계부 중복 지출을 찾는 엑셀 수식 예제야. 나중에 참고해'),
    ('relevance',1,2,'excel','엑셀 로고 머그컵 출시. 가계부 기능은 없는 장식 상품이야'),
    ('relevance',2,5,'writing','여행 준비물 문서에서 여권 유효기간 항목 빠진 거 봤어. 출발은 한 달 뒤야'),
    ('relevance',2,4,'writing','여행 준비물 정리할 때 쓸 일반 체크리스트야. 급하게 볼 필요 없어'),
    ('relevance',2,5,'video','산책영상.mp4의 12초 자막 오타 위치 캡처야. 납품 마감은 없어'),
    ('relevance',2,4,'video','영상 자막 대비 조절하는 일반 가이드 보냈어. 시간 날 때 봐'),
    ('relevance',2,5,'calendar','내일 치과 예약 시간이 15시에서 16시로 바뀌었어. 오늘 바로 답할 필요는 없어'),
    ('relevance',2,4,'calendar','진료 예약 일정에 이동 시간을 넣는 방법 안내야. 편할 때 참고해'),
    ('relevance',1,1,'empty','노트북 할인 쿠폰 안내입니다. 구매는 자유입니다'),
    ('state_contrast',2,1,'dev','집에 불났던 건 지난달 일이야. 다 수리했고 지금은 안전해'),
    ('state_contrast',1,1,'empty','영화에서 집에 불나는 장면 진짜 무섭더라'),
    ('state_contrast',2,1,'dev','화재 경보는 오작동으로 확인됐고 해제됐어. 대피할 필요 없대'),
    ('state_contrast',1,1,'empty','배터리에서 연기 날 때 대처하는 교육 영상 공유야. 실제 사고는 없어'),
    ('state_contrast',2,1,'dev','가스 누출 점검 끝났고 이상 없대. 추가 조치는 없어'),
    ('state_contrast',2,1,'empty','천장 누수는 수리 끝났어. 전기 설비도 정상 확인했고 위험 없어'),
    ('state_contrast',3,1,'dev','회사 계정 로그인 승인 요청입니다. 본인이 요청한 것인지 확인하세요'),
    ('state_contrast',5,1,'empty','내가 안 한 출금으로 잔액이 줄었어. 추가 출금도 진행 중이래'),
    ('state_contrast',2,1,'dev','내가 로그인한 기록 맞아. 승인 끝났고 계정도 정상 상태야'),
    ('state_contrast',1,1,'empty','계좌 도용 사례를 설명하는 보안 교육 자료야. 내 계좌 피해는 없어'),
]

PROFILES = {
    'empty': ('', ''),
    'dev': ('Code.exe', 'auth_session_test.py - login-service - Visual Studio Code'),
    'excel': ('EXCEL.EXE', '생활비.xlsx - 이번 달 식비 - Excel'),
    'writing': ('WINWORD.EXE', '여행 준비물.docx - Word'),
    'video': ('DaVinciResolve.exe', '산책영상.mp4 자막 수정 - Resolve'),
    'calendar': ('olk.exe', '내일 치과 예약 시간 정리 - Outlook'),
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output folder; preserve existing evaluation data')
    if len(CASES) != 100:
        raise ValueError(f'Expected 100 authored cases, got {len(CASES)}')
    bodies = [re.sub(r'\s+', '', row[4]).lower() for row in CASES]
    if len(set(bodies)) != 100:
        raise ValueError('Duplicate evaluation body')
    # Exclude exact overlaps with either model training and old validation.
    sources = [ROOT/'filtering_training/outputs/v7_relevance_ladder_01/prepared_laya_v2'/t/'train.jsonl' for t in ('urgency','relevance')]
    sources += [ROOT/'filtering_training/outputs/v5_training_pool_01/prepared_continual'/t/'train.jsonl' for t in ('urgency','relevance')]
    existing = set()
    source_hashes = {}
    for source in sources:
        source_hashes[str(source.relative_to(ROOT))] = digest(source)
        for line in source.read_text(encoding='utf-8').splitlines():
            value = json.loads(line)
            state = value['state'] if 'state' in value else json.loads(value['messages'][1]['content'])
            existing.add(re.sub(r'\s+', '', state['notification']['body']).lower())
    prediction_path = ROOT/'filtering_training/outputs/runs/2026-10-10-klue-base-02/cpu-fp32.predictions.json'
    source_hashes[str(prediction_path.relative_to(ROOT))] = digest(prediction_path)
    existing.update(re.sub(r'\s+', '', r['notification']['body']).lower() for r in json.loads(prediction_path.read_text(encoding='utf-8')))
    overlaps = set(bodies) & existing
    if overlaps:
        raise ValueError(f'Exact normalized overlap: {overlaps}')
    samples, review = [], ['<!-- 용도: 예측을 보지 않고 검수하는 합성 평가 라벨 제안. 생성일: 2026-10-10 -->',
                           '# 실사용 형식 합성 평가 100건',
                           '', '정답은 기존 지침에 따른 AI 제안이며 사용자 검수 완료를 뜻하지 않는다. 예측은 이 파일에 넣지 않는다.',
                           '원문 중복은 없지만 의미 유사성은 허용한다. 무작위 실사용 표본이 아닌 안전·경계 점검용 자료다.', '']
    for index, (group, urgency, relevance, profile, body) in enumerate(CASES, 1):
        process, title = PROFILES[profile]
        notification = {'id': f'personal_eval01_{index:03d}', 'app_name': '카카오톡' if group in ('safety','routine','state_contrast') else 'Slack',
                        'sender': '연락처' if group in ('safety','routine','state_contrast') else '동료',
                        'title': '새 메시지', 'body': body, 'timestamp': '2026-10-10T03:00:00Z'}
        context = {'active_process': process, 'window_title': title, 'last_updated': '2026-10-10T03:00:00Z',
                   'duration_seconds': 60 if process else 0, 'recent_processes': [], 'recent_windows': []}
        sample = {'notification': notification, 'context': context,
                  'label': {'urgency_score': urgency, 'relevance_score': relevance, 'category': '기타',
                            'ai_summary_reason': '기존 점수 지침에 따른 합성 평가용 제안; 인간 검수 미완료'},
                  'evaluation_metadata': {'group': group, 'context_profile': profile, 'label_status': 'ai_proposed_unreviewed',
                                          'source': 'authored_synthetic', 'intended_use': 'evaluation_only'}}
        samples.append(sample)
        review += [f'## {index}. {group}', f'- 본문: {body}', f'- 현재 창: {title or "빈 창"}',
                   f'- 제안: 긴급도 {urgency} / 관련도 {relevance} / {"통과" if urgency>=4 or relevance>=4 else "차단"}', '']
    args.output.mkdir(parents=True)
    data = args.output/'dataset.jsonl'
    data.write_text(''.join(json.dumps(s, ensure_ascii=False)+'\n' for s in samples), encoding='utf-8')
    (args.output/'review.md').write_text('\n'.join(review)+'\n', encoding='utf-8')
    manifest = {'purpose': 'Frozen synthetic personal-use notification challenge set; evaluation only', 'created_date': '2026-10-10',
                'label_status': 'ai_proposed_unreviewed', 'rows': len(samples), 'human_approved_rows': 0,
                'groups': dict(Counter(s['evaluation_metadata']['group'] for s in samples)),
                'urgency_scores': dict(Counter(s['label']['urgency_score'] for s in samples)),
                'relevance_scores': dict(Counter(s['label']['relevance_score'] for s in samples)),
                'training_used': False, 'final_test_used': False, 'dataset_sha256': digest(data),
                'labels_authored_before_model_predictions': True, 'exact_normalized_training_validation_body_overlaps': 0,
                'protected_sources': source_hashes, 'limitations': ['Synthetic, not collected user notifications',
                'Challenge group proportions do not represent real traffic', 'Proposed labels need independent human review',
                'Not an untouched final test; future tuning requires a separate evaluation set']}
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(args.output), 'rows':len(samples),'groups':manifest['groups'],'labels':'AI proposals, frozen before inference'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
