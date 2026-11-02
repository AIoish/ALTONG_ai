"""용도: 짧은 위험 제보·사건 상태·암묵적 긴급도 경계의 별도 보강 자료를 준비한다.
생성일: 2026-10-10
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_DATA = ROOT/'filtering_training/outputs/v7_relevance_ladder_01/prepared_laya_v2'
OLD_EVAL = ROOT/'filtering_training/outputs/v9_personal_use_eval_01/dataset.jsonl'

# Authored short and medium messages; do not copy the prior evaluation cases.
TRAIN = '''
5|fire|살려줘 우리 아파트에 불붙었어
5|fire|안방에서 불이 번지고 있어
5|fire|건물 입구가 불길에 막혔어
5|fire|주방 커튼이 타고 있어 도와줘
5|fire|엄마 집에서 불이 난대 연락해줘
5|fire|화재경보기 울리고 복도에 불꽃이 보여
5|fire|옆집 불이 우리집 쪽으로 번진다
5|fire|불이낫어 대피해야 돼
5|fire|지금 창고가 타고 있어 사람이 안에 있어
5|fire|복도 불 때문에 할아버지가 못 나오고 있어
5|smoke|침대 밑에서 연기가 올라와
5|smoke|아기 방에 연기가 가득해
5|smoke|문틈으로 연기가 들어오고 있어
5|smoke|연기 때문에 숨을 못 쉬겠어
5|smoke|충전기에서 연기나 전원 뽑을 수가 없어
5|smoke|휴대폰이 부풀더니 연기가 나
5|smoke|부엌에서 타는 냄새 나고 연기가 보여
5|gas|가스 새고 있어 밖으로 나가래
5|gas|가스가 새는 것 같아 냄새가 너무 심해
5|gas|가스레인지 연결관에서 냄새 나고 쉬익 소리가 나
5|gas|가스 감지기가 계속 울려 밖으로 대피 중이야
5|gas|보일러실에서 가스 냄새가 퍼져 나와
5|gas|가스누출 확인됐어 집 안에 들어오지 마
5|electric|멀티탭에서 불꽃이 튀고 있어
5|electric|젖은 콘센트가 지지직거리고 연기가 나
5|electric|벽 안 전선이 타는 것 같아 불꽃이 보여
5|electric|분전함에서 불이 붙었어
5|electric|전기난로가 넘어져 이불이 타고 있어
5|electric|물 새는 데 전선이 닿아 사람 감전됐어
5|water|누수된 물이 전선에 닿아서 불꽃이 나
5|water|천장이 무너지고 전등에서 전기가 튀어
5|water|집 안에 물이 빠르게 차는데 아이가 갇혔어
5|water|욕실 천장에서 물 떨어지고 조명이 지지직거려
5|water|침수 때문에 출입문이 막혀서 못 빠져나가
5|medical|할머니가 쓰러져서 반응이 없어
5|medical|남편이 의식 없대 구급대 기다리고 있어
5|medical|애가 숨을 못 쉰대 빨리 연락줘
5|medical|아이가 음식에 목 막혀서 숨이 안 쉬어진대
5|medical|동생이 갑자기 쓰러져서 응급차 타고 가
5|medical|엄마가 호흡을 안 해 구급차 불렀어
5|medical|할아버지 가슴 통증 심하고 숨차대 응급실 가는 중
5|medical|상처에서 피가 쏟아져 멈추지 않아
5|medical|가족이 교통사고 났어 응급실에서 보호자 찾고 있어
5|medical|아빠 사고로 응급 수술 들어간대 전화줘
5|medical|응급실에서 아이 보호자 동의가 필요하대 연락 받아
5|crime|침입자가 문 열고 들어왔어 신고해줘
5|crime|현관 앞에서 누가 흉기로 협박해 경찰 기다리는 중
5|crime|집에 낯선 사람이 들어와서 아이가 숨어 있어
5|trapped|불난 건물에 가족이 갇혔어
5|trapped|엘리베이터에 갇힌 사람이 호흡 곤란이래
5|trapped|방 문이 안 열리는데 안에 연기 차고 있어
5|trapped|차 문이 잠겼는데 어린아이가 더위에 쓰러졌어
5|financial|내가 모르는 이체가 실행됐고 계속 출금되고 있어
5|financial|누가 내 카드로 연속 결제 중이야 차단해야 해
5|financial|도용된 계좌에서 돈이 계속 빠져나가
5|financial|본인 동의 없이 송금된 거래가 확인됐고 추가 송금이 진행돼
5|financial|해킹당한 계정으로 결제가 계속 처리되고 있어
5|deadline|접수 3분 남았는데 필수 서류가 없어 신청 취소된대
5|deadline|필수 치료 접수 곧 닫혀서 못 받게 생겼어 확인 전화 받아
5|deadline|지원금 신청 5분 뒤 끝나는데 확인 안 하면 무효야
2|resolved|불은 완전히 꺼졌고 모두 안전해. 추가 조치 없어
2|resolved|집 화재 수습 끝났어 가족들도 무사하고 지금 위험 없어
2|resolved|연기는 조리 중 발생한 것으로 확인됐고 환기 끝났어 위험 없대
2|resolved|가스 냄새 점검했는데 누출 없고 안전하대
2|resolved|가스 누출 수리 끝나고 안전 확인 받았어
2|resolved|콘센트 불꽃 문제는 수리 완료됐고 전기 정상 확인했어
2|resolved|누수 수리하고 전선도 교체했어 추가 위험 없대
2|resolved|천장 물 새던 건 고쳤고 전등도 정상 확인받았어
2|resolved|응급실 치료 끝나 퇴원했고 지금은 괜찮아. 급한 연락 아니야
2|resolved|엄마 쓰러졌던 건 진료 끝났고 지금 안정적이래 추가 요청 없어
2|resolved|신고했던 침입자는 잡혔고 집도 안전 확인했어
2|resolved|엘리베이터 구조 끝났고 다친 사람 없어. 추가 연락 불필요
2|resolved|모르는 카드 결제는 정지 완료했고 피해 없이 취소됐어
2|resolved|계좌 도용 의심은 본인 거래로 확인됐어 추가 피해 없어
2|past|작년 집 화재 보험 정리 완료 알림이야 지금 사고는 없어
2|past|어제 연기 났던 장비는 폐기됐고 작업장 안전 확인했대
2|past|지난달 가스관 고장 기록 정리 끝났어 현재는 정상
2|past|지난번 응급실 영수증 보관했어 현재 아픈 사람 없어
2|past|예전에 누수 났던 사진 보냈어 지금은 다 고쳤어
1|education|가스 냄새 날 때 대피하는 법 교육자료야 현재 누출은 없어
1|education|아파트 화재 대응 교육 영상이야 실제 불난 건 아니야
1|education|아이 목에 음식 걸렸을 때 대처하는 강의 링크야 사고는 없어
1|education|누수와 감전 사고 사례를 설명하는 기사 공유야
1|education|응급실 이용 절차 설명 영상 올라왔어 실제 응급 상황은 아냐
1|education|카드 도용 방지법을 소개하는 뉴스야 내 카드 피해는 없어
1|fiction|드라마에서 가스 터지는 장면 보고 놀랐어 실제 사건 아니야
1|fiction|꿈에서 우리 집 불나는 걸 봤어 현실은 괜찮아
1|fiction|게임 캐릭터가 쓰러졌어 ㅋㅋ 현실 사람은 멀쩡해
1|fiction|영화 속 응급실 장면 후기야 실제 가족 사고 아냐
2|planned|가스 정기 점검은 다음 달이야 지금 누출은 없고 답 필요 없어
2|planned|소방 훈련이 다음 주에 예정돼 있어. 실제 화재 아니고 회신 불필요
2|planned|내달 전기 설비 점검 안내야 현재 설비는 안전해
3|planned|정기 가스 점검 날짜 오늘 중 골라줘. 지금 누출이나 고장은 없어
3|planned|다음 주 누수 예방 공사 날짜 오늘 저녁까지 알려줘. 현재 누수 없어
3|planned|다음 달 안전교육 참석 여부 오늘 안에 부탁해. 현재 사고는 없어
3|security_check|요청한 로그인 맞는지 오늘 중 확인해줘. 계정 침해 징후는 없어
3|security_check|새 기기 로그인 승인 요청이야. 피해는 확인되지 않았으니 본인 요청인지 확인해
2|security_check|네가 요청한 로그인 승인 완료됐고 계정도 정상 상태야
2|security_check|카드 결제 승인 내역이야 본인 거래로 확인됐고 추가 조치 없어
4|boundary|출발할 배송 라벨에 아파트 호수가 잘못됐어. 보내기 전에 정정 부탁해
4|boundary|고객에게 전달할 견적 총액이 틀렸어. 전송 전에 숫자 확인해줘
4|boundary|결재 기다리느라 납품 차량이 못 나가고 있어. 승인 여부 알려줘
4|boundary|프린터 앞에서 답 기다려. 양면인지 단면인지 알려줘야 출력해
4|boundary|업체가 생산을 시작하려는데 도면 치수가 달라. 시작 전에 수정해줘
4|boundary|네 발표가 다음 순서래 자료 준비해줘
4|boundary|접수 종료 10분 남았는데 사진 파일 빠졌대. 추가해줘
4|boundary|기사님 건물 앞에서 열쇠 위치 기다려. 문을 못 열고 있어
4|boundary|결제 진행하려는데 주문 수량이 틀렸어. 결제 전에 고쳐줘
4|boundary|이체 실행 전에 수취 계좌 확인해줘. 은행 담당자가 대기 중이야
4|boundary|인쇄 시작 직전인데 날짜가 잘못 적혔어. 인쇄 전에 연락 부탁해
4|boundary|서비스 복구가 네 설정 확인을 기다리고 있어. 담당자 통화 받아줘
4|boundary|배포가 승인 누락으로 멈췄어 담당자가 응답 기다려
4|boundary|회의에서 자료를 열었는데 최신 파일이 아니래. 교체해줘야 진행해
3|boundary|다음 주 보낼 견적 총액 오늘 중 검토해줘. 아직 발송 전 준비 단계야
3|boundary|배송은 사흘 뒤야. 주소 검토 오늘 저녁까지 부탁해
3|boundary|도면은 다음 주에 쓸 거야. 오늘 안에 치수 확인해줘
3|boundary|이번 달 주문 수량을 오늘 중 대조해줘. 오늘 결제 일정은 없어
3|boundary|이체는 다음 주야. 명단을 오늘 중 검토해주면 돼
3|boundary|내일 회의 자료 오늘 저녁까지 보내줘. 지금 진행 중인 회의는 없어
2|routine|파일 백업 끝났고 오류도 없어 편할 때 봐
2|routine|선물 택배 놓고 갔대 상하지 않는 물건이야
1|routine|주말 캠핑 사진 공유할게 답 안 줘도 돼
1|routine|오늘 밥 뭐 먹을까 그냥 잡담이야
2|routine|할인 쿠폰 곧 끝나지만 안 써도 손해 없어
1|routine|새 영화 예고편 나왔어 시간 나면 봐
'''

# Separate development originals, authored and frozen before fine-tuning.
DEV = '''
5|safety|커피포트 전선에서 불 나서 식탁이 타고 있어
5|safety|살려줘 침실 안에 불이 번졌어
5|safety|연기가 방문 사이로 들어오는데 아기가 방에 있어
5|safety|보일러 옆에서 가스 냄새가 너무 나서 대피했어
5|safety|할아버지 넘어지고 의식이 돌아오질 않아
5|safety|아이 입술이 파래지고 호흡이 안 돼 구급차 기다려
5|safety|건물 지하에 불꽃 보여 사람들이 안에서 못 나와
5|safety|누수된 물 때문에 천장 조명에서 연기 나
5|safety|젖은 전기선에 사람이 닿아서 쓰러졌어
5|safety|도둑이 집 안까지 들어와 경찰에 신고했어
5|safety|우리 카드로 모르는 결제가 계속돼 정지해야 해
5|safety|발목 상처에서 피가 너무 많이 나고 안 멈춰
5|safety|세탁기 뒤에서 불꽃 튀어 옷에 불 붙었어
5|safety|아이가 뜨거운 차에 갇혀 반응이 없어 구조 요청했어
5|safety|동생 사고 나서 응급 수술 중이야 병원에서 가족 찾는대
5|safety|가스경보기가 울리고 실제 누출이 확인됐대 밖으로 나가
5|safety|실내 배터리가 터졌고 연기가 퍼지는 중이야
5|safety|다락방 화재로 출입구가 막혔고 사람이 갇혔어
5|safety|전기 스파크 때문에 커튼에 불이 붙었어
5|safety|보호자 응답 없어 응급 치료 동의가 지연되고 있어 연락줘
4|boundary|출고할 상품 수량이 주문보다 적어. 출고 전에 바로잡아줘
4|boundary|담당자가 문 앞에서 인수 확인을 기다려. 누구한테 전달할지 알려줘
4|boundary|고객과 통화 중인데 네 답 없어서 상담 못 끝내. 확인 연락줘
4|boundary|문서 인쇄 시작 전에 오타 수정해야 해 담당자가 대기 중이야
4|boundary|배포 담당자가 오류 수정 승인을 기다리며 작업을 멈췄어
4|boundary|다음 발표자가 너래 순서 바뀌었으니 발표 자료 준비해
4|boundary|기사님이 픽업하러 왔는데 주소가 달라. 출발 전에 정정해줘
4|boundary|은행 창구에서 수취인 확인 기다려. 아직 실행 전이야
4|boundary|행사 입장 마감 10분 남았어 필수 확인 응답이 없어서 못 들어가
4|boundary|입찰 제출창 닫히기 15분 전인데 서명이 빠졌어 보완해줘
2|resolved|어제 분전반 불은 수리하고 안전 확인 끝났어 추가 처리 없어
2|resolved|누출된 가스는 차단했고 누출 검사도 정상 나왔어 위험 없대
2|resolved|응급실 진료 마치고 퇴원했어 연락은 편할 때 해
1|education|가정 화재 대피 순서 설명하는 교육자료 공유야 실제 사고 없어
1|fiction|게임에서 아파트가 불타는 장면 봤어 실제 우리 집은 안전해
3|planned|이번 주 안전 점검 날짜 오늘 중 정해줘. 설비는 정상 상태야
3|boundary|다음 달 납품 수량 오늘 저녁까지 검토해줘. 오늘 출고 없어
2|routine|화재 보험 증권 새 파일 도착했어 사고는 없고 추가 조치 필요 없어
1|routine|소방차 장난감 세일 안내야 구매는 자유야
3|security_check|본인 요청 로그인인지 확인 부탁해 침해나 피해는 확인 안 됐어
'''

def parsed(text):
    return [(int(score),group,body) for score,group,body in (line.split('|',2) for line in text.strip().splitlines())]

def norm(body):
    return re.sub(r'\s+','',body).lower()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new data folder')
    old=[json.loads(line) for line in (BASE_DATA/'urgency/train.jsonl').read_text(encoding='utf-8').splitlines()]
    question=old[0]['questions']
    excluded={norm(r['state']['notification']['body']) for r in old}
    old_validation=json.loads((ROOT/'filtering_training/outputs/runs/2026-10-10-klue-base-02/cpu-fp32.predictions.json').read_text(encoding='utf-8'))
    excluded.update(norm(r['notification']['body']) for r in old_validation)
    excluded.update(norm(json.loads(line)['notification']['body']) for line in OLD_EVAL.read_text(encoding='utf-8').splitlines())
    dev_bodies={norm(body) for _,_,body in parsed(DEV)}
    if len(dev_bodies)!=40 or dev_bodies & excluded:
        raise ValueError('Development case overlap or count error')
    output=[]
    for i,(score,group,body) in enumerate(parsed(TRAIN),1):
        for j,text in enumerate([body,'연락 왔어. '+body,body+'. 내용 확인 부탁해'],1):
            notification={'app_name':['카카오톡','휴대폰과 연결','Slack'][(i+j)%3],
                          'sender':['가족','지인','동료'][i%3],'title':'새 메시지' if j!=2 else '',
                          'body':text,'timestamp':'2026-10-10T05:00:00Z'}
            output.append({'state':{'notification':notification},'questions':question,'expected':{'urgency':str(score)},
                           'metadata':{'id':f'safety10_{i:03d}_{j}','family':group,'original_id':f'safety10_{i:03d}',
                                       'label_status':'ai_proposed_user_authorized_training'}})
    if len({norm(r['state']['notification']['body']) for r in output})!=len(output):
        raise ValueError('Supplement duplicates')
    if any(norm(r['state']['notification']['body']) in excluded|dev_bodies for r in output):
        raise ValueError('Training intersects evaluation or existing urgency body')
    development=[]
    for i,(score,group,body) in enumerate(parsed(DEV),1):
        development.append({'notification':{'id':f'safety_dev10_{i:03d}','app_name':'카카오톡','sender':'지인',
                                           'title':'새 메시지','body':body,'timestamp':'2026-10-10T05:00:00Z'},
                            'context':{'active_process':'Code.exe' if i%2 else '',
                                       'window_title':'auth_session_test.py - login-service - Visual Studio Code' if i%2 else '',
                                       'duration_seconds':60 if i%2 else 0,'last_updated':'2026-10-10T05:00:00Z',
                                       'recent_processes':[],'recent_windows':[]},
                            'label':{'urgency_score':score,'relevance_score':1,'category':'기타',
                                     'ai_summary_reason':'합성 개발 평가의 AI 제안, 인간 개별 검수 미완료'},
                            'evaluation_metadata':{'group':group,'label_status':'ai_proposed_unreviewed'}})
    args.output.mkdir(parents=True)
    for name,rows in [('supplement.jsonl',output),('development.jsonl',development)]:
        (args.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest={'purpose':'Separate short safety/state/boundary supplement and frozen development set',
              'created_date':'2026-10-10','label_status':'ai_proposed_user_authorized_training',
              'human_individually_reviewed':False,'originals':len(parsed(TRAIN)),'rows':len(output),
              'score_counts':dict(Counter(r['expected']['urgency'] for r in output)),
              'family_counts':dict(Counter(r['metadata']['family'] for r in output)),
              'development_rows':len(development),'original_variants':3,'prior_100_eval_used_for_training':False,
              'supplement_sha256':sha(args.output/'supplement.jsonl'),'development_sha256':sha(args.output/'development.jsonl'),
              'overlap_checks':'Normalized exact bodies excluded against prior urgency, original 74 and prior challenge 100; development disjoint',
              'sources':{str(BASE_DATA.relative_to(ROOT))+'/urgency/train.jsonl':sha(BASE_DATA/'urgency/train.jsonl'),
                         str(BASE_DATA.relative_to(ROOT))+'/relevance/train.jsonl':sha(BASE_DATA/'relevance/train.jsonl'),
                         str(OLD_EVAL.relative_to(ROOT)):sha(OLD_EVAL)},
              'limitations':['Synthetic proposals, not individual human approvals','Character exact checks do not guarantee semantic independence',
                             'Prior challenge errors informed domains; prior challenge is regression diagnostic, not fresh untouched final test']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    review=['<!-- 용도: 별도 안전 보강 원문과 점수 제안을 검토한다. 생성일: 2026-10-10 -->',
            '# 안전·상태·경계 보강 원문','', '사용자 보강 요청으로 학습할 AI 제안이다. 인간 개별 검수 완료를 뜻하지 않는다.',
            '이전 평가 100건의 원문은 사용하지 않았다. 점수는 기존 지침 기준이다.','',
            '| 번호 | 유형 | 긴급도 제안 | 원문 |','| --- | --- | ---: | --- |']
    review += [f'| {i} | {group} | {score} | {body} |' for i,(score,group,body) in enumerate(parsed(TRAIN),1)]
    (args.output/'review.md').write_text('\n'.join(review)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(output),'originals':manifest['originals'],'scores':manifest['score_counts'],'development':40},ensure_ascii=False))

if __name__=='__main__':
    main()
