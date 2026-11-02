"""용도: 긴급도 3/4 경계·관련도 보강과 별도 평가를 생성한다.
생성일: 2026-10-10
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from filtering_training.common.score_tasks import HistorySample

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'filtering_training/outputs/v7_relevance_ladder_01/prepared_laya_v2'
SAFETY=ROOT/'filtering_training/outputs/v10_safety_supplement_01'
URG='''
4|서명 대기 중인 구매 계약에 공급가가 잘못 들어갔어. 서명 전에 확인해줘
3|다음 달 구매 계약 초안의 공급가 오늘 안에 검토해줘. 서명 일정은 아직 없어
4|거래처에 주문서를 보내려는데 합계가 안 맞아. 전송을 멈추고 답 기다려
3|다음 주 주문서의 합계 오늘 저녁까지 확인해줘. 지금 전송하지 않아
4|담당자가 계약을 체결하려는데 세금 포함 금액이 달라. 체결 전에 연락줘
3|계약 조건 비교표의 세금 포함 금액을 오늘 중 살펴봐줘. 체결은 다음 달이야
4|서명 화면에 월 이용료가 협의한 것과 달라. 승인 전에 수정 요청해줘
3|월 이용료 협의안 오늘 중 의견 줘. 아직 승인할 문서는 없어
4|업체가 작업을 시작하려는데 발주 금액이 잘못됐대. 시작을 보류하고 기다려
3|업체가 다음 주 쓸 발주 금액 오늘 안에 검토해달래. 지금 작업 시작은 없어
4|이체 직전인데 거래 명세서 합계가 다르게 나와. 실행하기 전에 확인해줘
3|거래 명세서 합계 오늘 중 확인해줘. 송금 예정은 사흘 뒤야
4|연장 계약을 지금 승인하려는데 기간이 틀려. 확인할 때까지 승인 멈췄어
3|연장 계약 기간을 오늘 저녁까지 정리해줘. 승인 일정은 다음 주야
4|고객이 결제하려는데 할인 적용 금액이 달라. 결제를 보류하고 답 기다려
3|다음 달 할인 적용 금액을 오늘 중 계산해줘. 지금 결제하는 고객은 없어
4|회의에서 비용안을 확정하려는데 수량이 달라. 확정을 멈췄으니 확인해줘
3|내일 비용안 회의 준비로 수량 오늘 중 확인해줘. 지금 확정하는 건 없어
4|서류 접수가 네 누락 항목 확인을 기다리고 있어. 창구 닫기 전에 회신해줘
3|다음 주 접수할 서류의 누락 항목 오늘 중 찾아줘. 현재 접수 중은 아니야
2|비상식량 배송 완료됐어. 문 앞에 있고 보관기한도 넉넉해
2|긴급 배송 상품 도착 안내야. 이미 받았고 추가로 처리할 건 없어
2|보냉 필요 없는 책을 택배함에 보관했대. 나중에 찾아가면 돼
2|네가 신청한 기기 로그인 처리 완료됐어. 계정 정상이고 확인 요청 없어
3|네가 요청한 로그인인지 오늘 중 확인해줘. 침해 증거는 없고 계정 정상이어
3|새 브라우저 승인 요청이야. 거래 피해 없으니 본인 요청인지 살펴봐줘
5|우리 계좌에서 내가 안 보낸 돈이 계속 나가 지금 차단해야 해
5|카드 도용 결제가 반복되는 중이래 지금 고객센터 연락해
5|아이 방 문이 잠겼는데 안에서 타는 냄새와 기침 소리가 나
5|계단에서 엄마가 넘어져 움직이지 않고 대답도 없어
2|방에서 났던 타는 냄새는 점검 완료했고 가족 모두 안전해
1|계단 사고 예방 동영상 공유야. 지금 다친 사람이 있는 건 아니야
'''
REL='''
Excel.exe|생활비 결산 - Excel|이번 달 카드값 중복 입력을 찾는 함수 설명 자료야
Excel.exe|가족 예산 관리.xlsx - Excel|가족 예산표에서 합계가 두 번 계산되는 걸 막는 수식 예시야
Excel.exe|가계부 월별 지출 - Excel|월별 지출표의 같은 거래를 표시하는 조건부 서식 방법이야
Excel.exe|카드 사용 내역 정리 - Excel|카드 사용 내역에서 중복 거래를 골라내는 필터 예제야
Excel.exe|생활비 결산 - Excel|분류별 생활비 합계를 구하는 피벗 테이블 참고 자료야
Excel.exe|가족 예산 관리.xlsx - Excel|가족 예산표 누락 항목을 찾는 엑셀 함수 안내야
Code.exe|budget_parser.py - household-tools - Visual Studio Code|가계부 CSV에서 같은 지출 행을 제거하는 파이썬 예제야
Code.exe|budget_parser.py - household-tools - Visual Studio Code|가계부 CSV 날짜 형식을 통일하는 파서 참고 코드야
chrome.exe|월간 가계부 - Google 스프레드시트|지출표의 거래번호 중복을 찾는 구글 시트 수식 예제야
chrome.exe|월간 가계부 - Google 스프레드시트|구글 시트에서 생활비 항목별 합계를 내는 함수 안내야
'''
DEV='''
4|전자 서명 직전인데 계약 총액에 한 자리 더 들어갔어. 서명 중단하고 확인해줘
3|이번 주 계약 초안 총액을 오늘 중 검토해줘. 전자 서명은 다음 주야
4|납품 계약을 승인하려는데 단가가 달라. 담당자가 승인을 멈추고 답 기다려
3|다음 달 납품 계약의 단가 검토 오늘 안에 부탁해. 오늘 승인 계획 없어
4|최종 견적을 지금 전달하려는데 부가세가 중복됐어. 전달 보류했으니 확인해줘
3|다음 주 제출할 견적의 부가세를 오늘 중 점검해줘. 지금 제출하지 않아
2|주문한 수건 배송 끝났어. 수령했고 별도 연락 필요 없어
3|등록하려던 태블릿 접속 승인 요청이야. 계정 피해는 없으니 본인 요청인지 확인해
5|아이 있는 방에서 연기가 나오는데 문고리가 뜨거워 열 수 없어
5|아버지가 욕실 바닥에 쓰러져 있고 불러도 대답이 없어
2|연기가 나던 전열기는 제거했고 소방 점검도 끝나 안전해
1|가스 유출 사고를 다룬 영화 리뷰야. 현실에서 새는 가스는 없어
'''
TEST='''
4|고객이 서명하려는데 계약서의 결제 금액이 협의액보다 커. 서명을 잠깐 멈추고 확인해줘
3|내달 고객 계약의 결제 금액을 오늘 저녁까지 검토해줘. 지금 서명할 일은 없어
4|매장에서 결제 화면에 주문한 수보다 많이 찍혔어. 결제 전에 답 줘
3|다음 주 주문할 수량을 오늘 안에 정리해줘. 지금 결제하지 않아
2|책 배송 사진 도착했어. 현관에 두었고 급하게 받을 물건 아니야
3|본인이 신청한 PC 인증인지 확인 요청이야. 계정 이상이나 피해는 발견되지 않았어
5|침대 옆 충전 케이블이 타고 불이 이불로 옮겨붙었어
5|할머니가 전화 중 갑자기 말 못하고 호흡 소리만 이상해. 옆 사람이 의식 없대
2|이불에 붙었던 불은 진압했고 다친 사람 없어. 대피도 끝났어
1|응급실 장면을 촬영하는 드라마 현장 사진이야. 실제 환자는 없어
'''

def readlines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return re.sub(r'\s+','',s).lower()
def sample(i,body,u=2,r=1,app='',title=''):
    return {'notification':{'id':i,'app_name':'카카오톡','sender':'지인','title':'새 메시지','body':body,'timestamp':'2026-10-10T05:00:00Z'},'context':{'active_process':app,'window_title':title,'last_updated':'2026-10-10T05:00:00Z','duration_seconds':60 if app else 0,'recent_processes':[],'recent_windows':[]},'label':{'urgency_score':u,'relevance_score':r,'category':'기타','ai_summary_reason':'AI 제안 라벨; 인간 개별 검수 미완료'}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('New output directory required')
    protected={BASE/t/'train.jsonl':sha(BASE/t/'train.jsonl') for t in ('urgency','relevance')}
    original=json.loads((ROOT/'filtering_training/outputs/runs/2026-10-10-klue-base-02/cpu-fp32.predictions.json').read_text(encoding='utf-8'))
    excluded={norm(x['state']['notification']['body']) for t in ('urgency','relevance') for x in readlines(BASE/t/'train.jsonl')}
    excluded.update(norm(x['notification']['body']) for x in original)
    for path in (ROOT/'filtering_training/outputs/v9_personal_use_eval_01/dataset.jsonl',SAFETY/'development.jsonl'):
        excluded.update(norm(x['notification']['body']) for x in readlines(path))
    excluded.update(norm(x['state']['notification']['body']) for x in readlines(SAFETY/'supplement.jsonl'))
    train=[]
    for i,line in enumerate(URG.strip().splitlines()):
        u,body=line.split('|',1)
        for j,text in enumerate((body,'메시지 왔어. '+body)):
            train.append({'state':sample(f'b11_u{i}_{j}',text,int(u)),'expected':{'urgency':u},'metadata':{'id':f'b11_u{i}_{j}','family':'boundary_state'}})
    for i,line in enumerate(REL.strip().splitlines()):
        app,title,body=line.split('|')
        for j,(proc,window,r) in enumerate(((app,title,4),('game.exe','게임 로비',1))):
            train.append({'state':sample(f'b11_r{i}_{j}',body,2,r,proc,window),'expected':{'relevance':str(r)},'metadata':{'id':f'b11_r{i}_{j}','family':'context_contrast'}})
    dev=[sample(f'b11_dev{i}',s.split('|',1)[1],int(s.split('|',1)[0])) for i,s in enumerate(DEV.strip().splitlines())]
    test=[sample(f'b11_test{i}',s.split('|',1)[1],int(s.split('|',1)[0])) for i,s in enumerate(TEST.strip().splitlines())]
    for group,pairs in ((dev,[('집계표.xlsx - 지출 정리 - Excel','월간 지출표의 동일 거래를 찾아 색칠하는 엑셀 수식 참고야'),('예산 계획 - Excel','지출표 누락 금액을 확인하는 엑셀 검증 수식 설명이야')]),(test,[('가정 지출 기록 - Excel','가정 지출 기록에서 중복 거래일을 찾는 엑셀 함수 활용 예시야'),('생활비 통계 - Excel','생활비 통계의 분류별 합계가 맞는지 대조하는 엑셀 계산식 자료야')])):
        for title,body in pairs:
            for app,window,r in (('Excel.exe',title,4),('game.exe','게임 로비',1)):
                group.append(sample(f'b11_{"dev" if group is dev else "test"}{len(group)}',body,2,r,app,window))
    tb={norm(x['state']['notification']['body']) for x in train};db={norm(x['notification']['body']) for x in dev};fb={norm(x['notification']['body']) for x in test}
    if (tb|db|fb)&excluded or tb&db or tb&fb or db&fb:raise ValueError('Normalized overlap')
    for x in dev+test:HistorySample.model_validate(x)
    for x in train:HistorySample.model_validate(x['state'])
    replay=readlines(SAFETY/'supplement.jsonl');a.output.mkdir(parents=True)
    for name,data in [('new_supplement.jsonl',train),('supplement.jsonl',replay+train),('development.jsonl',readlines(SAFETY/'development.jsonl')+dev),('dataset.jsonl',test)]:
        (a.output/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8')
    m={'purpose':'Boundary/context supplement with safety replay; separate final synthetic diagnostic','created_date':'2026-10-10','new_rows':len(train),'safety_replay_rows':len(replay),'rows':len(test),'development_rows':len(dev)+40,'label_status':'AI proposed; user authorized training; no individual human review','supplement_sha256':sha(a.output/'supplement.jsonl'),'development_sha256':sha(a.output/'development.jsonl'),'dataset_sha256':sha(a.output/'dataset.jsonl'),'protected_sources':{str(k.relative_to(ROOT)):v for k,v in protected.items()},'limitations':['Exact exclusion is not semantic independence','Small synthetic diagnostic, not real traffic','Prior evaluation errors informed topics']}
    (a.output/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    review=['<!-- 용도: 새 경계·관련도 학습 사례를 검토한다. 생성일: 2026-10-10 -->','# 경계·관련도 보강','AI 제안 라벨이며 인간 개별 검수 완료가 아니다. 같은 본문에 다른 창을 넣은 관련도 대조쌍은 의도적이다.','| 과제 | 점수 | 현재 창 | 본문 |','| --- | --- | --- | --- |']
    for x in train:
        t=next(iter(x['expected']));review.append(f"| {t} | {x['expected'][t]} | {x['state']['context']['window_title']} | {x['state']['notification']['body']} |")
    (a.output/'review.md').write_text('\n'.join(review)+'\n',encoding='utf-8')
    print(json.dumps(m,ensure_ascii=False))

if __name__=='__main__':main()
