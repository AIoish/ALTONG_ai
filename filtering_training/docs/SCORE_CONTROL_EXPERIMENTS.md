<!-- 용도: 숫자 토큰 비중과 정답 분포의 원인을 분리하는 제한 대조 실행 안내.
생성일: 2026-10-04 (Asia/Seoul)
상태: 보강 묶음 승인 및 기존 데이터 세 대조 학습·검증 완료. -->

# 점수 학습 대조

## 고정 조건

기존 0.6B 분리 모델과 같은 데이터·분할·프롬프트·seed 42를 사용한다.
긴급도 50스텝, 관련도 150스텝으로 고정하고 매번 기반 모델에서 시작한다.
NF4 이중 양자화·BF16, LoRA r=8/alpha=16/dropout=0.05, q_proj/v_proj,
학습률 0.0002, batch 1, 최대 길이 768도 유지한다.
새 보강 자료를 동시에 넣지 않아 데이터 추가 효과를 섞지 않는다.
검증 원문은 8개·문맥 24건이며 최종 테스트는 추론하지 않는다.

## 대조

| 이름 | 숫자 loss 가중치 | 샘플링 | 계산 경로 | 비교 목적 |
| --- | --- | --- | --- | --- |
| original | 1 | 기존 원문 순서에서 학습기 shuffle | 기본 chunked_nll | 기존 결과 |
| standard_nll | 1 | original과 동일 | nll + 명시적 completion 평균 | 계산 경로 영향 확인 |
| numeric_weight | 7 | original과 동일 | standard_nll과 같은 경로 | 숫자 가중치만 변경 |
| balanced | 1 | 점수별 균등, replacement | original과 같은 chunked_nll | 정답 분포만 변경 |

숫자 가중치 실험에서는 8개 completion 토큰 중 숫자 1개에 가중치 7,
나머지 7개에는 1을 주고 전체 가중치 합으로 정규화한다. 숫자와 형식의
비중이 각각 절반이다. shift는 한 번만 적용하며 prompt와 padding의 -100
label은 loss와 gradient에 기여하지 않는다. 실제 학습 준비 결과 모든 행에
supervised 숫자 토큰이 정확히 하나인지 확인한다.

설치된 TRL 1.13.0의 기본 chunked_nll은 사용자 정의 loss callback을 쓸 때
로깅 경로와 충돌했다. 실패한 2스텝 폴더는 삭제하지 않았다. callback 대조를
nll 경로로 명시하고 가중치 1의 기준선을 추가했다. 두 계산 경로의 수학적
목표는 같지만 부동소수점·실행 차이가 있을 수 있어 재현 여부를 따로 확인한다.
수정한 2스텝 점검은 성공했다.

균등 샘플링은 준비된 train 행에서만, seed 42로 replacement 샘플링한다.
긴급도는 점수마다 10건, 관련도는 점수마다 30건으로 실제 스텝 수를 유지한다.
뽑힌 원본 인덱스, 점수별 건수, 고유 행 수, 반복 횟수를 기록한다.
기존 관련도 2점이 3행뿐이라 반복될 수 있으며 새 원문 확보와 동일시하지 않는다.
문맥별 행을 다시 뽑으므로 실제 처리 토큰 수까지 같은 대조라고 주장하지 않는다.

## 승인 자료

사용자 수정 `6번 관련도 2. 7번 관련도 4로 변환.`을 보존하고,
뒤이은 `ㄱㄱ`를 수정 결과 제시 후 진행 승인으로 기록했다.
outputs/v3_score_review_02_approved_01/에 원문 8개·문맥 24건과 승인 이력이 있다.
6번 머그컵 광고는 관련도 2/BLOCK, 7번 스페인어 농담은 관련도 4/PASS다.
7번은 일반 농담 기준의 사용자 지정 예외로 별도 식별한다.
공통 프롬프트나 기존 라벨링 가이드를 바꾸지 않았다. 변경 이유 문장은
사용자 점수 지정의 근거를 꾸며내지 않고 검수 출처를 명시하도록 정리했다.
이 자료는 승인 확정됐지만 이번 기존 데이터 대조 학습에는 넣지 않았다.
기존 24개 원문의 인간 검수 범위와 첫 보강 묶음의 격리 상태도 유지한다.

## 실행과 검증

```powershell
python -m filtering_training.preparation.freeze_score_review --approval-text "ㄱㄱ"
python -m unittest filtering_training.checks.test_score_controls filtering_training.checks.test_freeze_score_review -v
python -m filtering_training.modeling.run_score_pilot --experiment controls --prepared-root filtering_training/outputs/v3_score_separation_01/prepared --run-dir filtering_training/outputs/runs/<새-실행-폴더>
python -m filtering_training.modeling.compare_score_controls --prepared-root filtering_training/outputs/v3_score_separation_01/prepared --baseline filtering_training/outputs/runs/2026-10-03-qwen06-score-pilot-01 --controls filtering_training/outputs/runs/2026-10-04-qwen06-score-controls-01 --output filtering_training/outputs/runs/2026-10-04-qwen06-score-controls-01/comparison.json
```

준비·승인·실행 폴더가 이미 존재하면 덮어쓰지 않고 중단한다.
비교기는 분할·ID·입력·정답·원시 응답과 지표를 재확인하며,
학습 설정·샘플링 인덱스·실제 점수 분포와 변경 범위를 함께 검증한다.
생성 JSON 형식 준수율뿐 아니라 숫자 정답, 긴급 실패, 오통과를 확인한다.
점수별 학습 표본 10건은 편향 진단용이며 전체 학습 정확도로 보고하지 않는다.

## 완료 결과

일반 nll·숫자 가중치 대조는 여전히 모든 검증 점수를 1로 냈다.
균등 샘플링은 검증을 전부 PASS시켜 긴급 오차단 0/6 대신 불필요 오통과가
12/12가 됐다. 정책 정답은 세 대조 모두 12/24다. 어느 대조도 앱에 적용하지 않는다.
로컬 상세 결과는 outputs/runs/2026-10-04-qwen06-score-controls-01/summary.md에 있다.
다음은 JSON 형식 생성을 제외하고 숫자 자체를 학습·평가하는 비교다.
