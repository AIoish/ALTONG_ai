# Filtering AI 모델 학습 구현 계획

## 목표와 현재 상태

Qwen3-1.7B가 `RawNotification`과 `CurrentContext`를 입력받아
`urgency_score`, `relevance_score`, `category`, `ai_summary_reason` 네 필드의
JSON을 출력하도록 LoRA SFT 파이프라인을 만든다. 최종 `is_passed`는
`src/filtering/policy.py`가 결정한다. 출력 데이터 계약은 유지한다. 확장된 `CurrentContext` 입력은 아래 선행 단계에서 반영한다.

현재 `feature/filtering-training` 브랜치에는 프롬프트 빌더, 출력 검증기,
샘플 한 건 추론 명령이 구현돼 있다. 기존 30개와 새 맥락을 포함한 합성 샘플 4개는 스키마 검증을
통과한다. Qwen3-0.6B 기본 모델의 실제 추론에서 JSON 출력은 성공했지만,
샘플 한 건의 관련도는 정답 5에 대해 3이었다. 이는 성능 평가 결과가 아니다.

기존 30개 기준 데이터는 8개 카테고리에 걸쳐 있고 `기타`는 1건뿐이다.
추가 샘플을 포함한 34건으로 만든 평가셋은 점수 변동이 크므로 학습 코드의 작동만
확인한다. 성능 개선 주장은 확장된 별도 평가셋에서 판단한다.

## 선행 단계: 확장된 CurrentContext 반영

클라이언트의 `DurationSeconds`는 현재 창에 머문 시간이며, 0은 초기 기본값이고 유효한 맥락은 5초 룰을 통과한다. `RecentProcesses`는 최근 1~2분 동안 교차 사용한 상위 프로세스의 작업 세트다. 이는 단순히 마지막으로 연 앱의 목록이나 작업 제목이 아니다. C# 속성 이름만으로 실제 전송 JSON 키가 snake_case, camelCase, PascalCase 중 무엇인지 정할 수 없다. 작업 세트의 정렬 기준과 현재 앱 포함 여부는 실제 클라이언트 JSON과 함께 확인한다. `RawNotification`의 기존 여섯 필드는 그대로 사용한다.

- `src/filtering/schema.py`에 경과 시간(0 이상)과 최근 프로세스 이름(최대 3개)을 선택 필드로 추가한다. 기존 30개 샘플은 기본값으로 읽되, 필드 누락과 `null`을 학습 입력에서 같은 의미로 다룰지 정의한다.
- `src/filtering/prompt.py`의 입력 JSON에도 두 필드를 포함한다. 학습과 추론은 같은 필드 이름, 기본값, 목록 순서를 사용한다. 빈 프로세스와 빈 창 제목은 작업 맥락이 없다는 뜻이며, `Empty.LastUpdated`는 실제 작업의 근거가 아니다.
- 라벨링 가이드와 샘플을 갱신한다. 최근 앱 이름은 작업 주제를 직접 알려주지 않으므로 `relevance_score`를 과도하게 높이지 않는다. 알림 자체의 긴급도는 최근 앱 목록에 따라 바뀌지 않게 검수한다. 기존 30개에 빈 목록만 일괄 추가해서 학습 데이터를 완성한 것으로 보지 않는다.
- 새 맥락이 있는 사례와 없는 사례, 현재 앱과 최근 앱이 다른 사례, 앱에 머문 시간이 다른 사례를 만든다. 같은 알림에서 맥락만 바꾼 쌍은 학습·평가 분할에서 같은 그룹에 둔다.
- 구형 샘플과 새 클라이언트 JSON 예시, 빈 맥락, 목록 0~3개, `null`, 음수 시간, 4개 이상 목록을 계약 테스트로 확인한다. 평가 결과는 최근 앱 정보가 있는 경우와 없는 경우를 나눠 본다.

이 단계가 완료된 뒤 1단계 데이터 변환과 2단계 기본 모델 평가를 진행한다. 프롬프트 입력이 바뀌므로 앞서 실행한 한 건 추론 결과는 새 입력 형식의 기준 점수로 재사용하지 않는다.

## 구현 순서

### 1. 학습 데이터 준비기

- `filtering_training/prepare_dataset.py`를 추가한다.
- JSONL을 `FilteringSample`로 검증하고 중복 ID, 누락·빈 라벨,
  허용되지 않은 카테고리를 확인한다.
- `src/filtering/prompt.py`의 `build_messages`를 그대로 사용해
  system/user 메시지를 만들고, label 네 필드만 assistant 응답 JSON으로
  직렬화한다. 모델이 직접 `is_passed`를 학습하지 않게 한다.
- Qwen chat template에서 `enable_thinking=False`를 사용한다. 추론과
  학습의 메시지 형식, JSON 키 순서, 종료 토큰을 동일하게 유지한다.
- 고정 시드와 분할 목록을 기록한다. 같은 문구의 변형이나 동일한 상황을
  공유하는 샘플은 한 분할에만 넣는다. 현재 34건의 train/validation/test 분할은
  파이프라인 점검용으로만 사용한다.

완료 기준: 현재 34건 모두 변환되고, 변환된 assistant 응답을 기존 출력 검증기로
다시 읽을 수 있으며, 서로 다른 분할에 중복·유사 사례가 없다.

### 2. 학습 전 평가기 (스모크 구현·실행 완료)

- `filtering_training/evaluate.py`를 추가해 기본 Qwen3-0.6B를 고정된
  평가셋에 추론한다. 평가셋의 정답은 프롬프트에 넣지 않는다.
- JSON 유효율, 두 점수의 정확도·±1 정확도·MAE, 카테고리 macro F1,
  정책 적용 후 PASS/BLOCK 정확도, 긴급 알림 오차단 비율과 불필요 알림
  오통과 비율을 기록한다. 파싱 실패도 실패 사례로 집계한다.
- 프롬프트 버전, 모델 ID 및 리비전, 데이터셋 버전, 디코딩 설정과 측정
  환경을 함께 기록한다. 개별 알림 원문이나 개인정보는 로그에 남기지 않는다.

완료 기준: 동일한 평가셋과 설정으로 기본 모델 결과를 재현할 수 있다.
초기 34건 결과는 참고용으로만 표시한다.

### 3. LoRA 학습 코드와 짧은 실행 (2단계 스모크 완료)

- `filtering_training/train.py`와 재현 가능한 설정 파일을 추가한다.
  로컬에 설치된 `transformers`, `trl`, `peft`, `accelerate` 버전을 먼저
  확인하고 해당 버전의 API에 맞춘다. 학습 설정과 시드를 저장한다.
- Qwen3-0.6B에 LoRA를 적용한다. 학습 시 프롬프트 부분은 loss에서
  제외하고 assistant JSON 응답만 학습한다. 저장물은 어댑터와 설정으로
  제한한다.
- RTX 3060 Laptop GPU 6GB에서 배치 크기, 시퀀스 길이,
  gradient accumulation을 작게 시작하고 실제 메모리 사용량을 확인한다.
  메모리 부족 시 설정을 줄이거나 다른 학습 환경을 검토한다.
- 현재 34건으로 짧게 실행해 데이터 로딩 → 학습 → 어댑터 저장 → 재로딩 →
  샘플 추론을 확인한다. 이 실행의 점수로 모델 성능을 주장하지 않는다.

완료 기준: 새 프로세스에서 저장된 어댑터를 불러와 네 필드 JSON을
출력하고, 기존 정책 코드로 최종 결정을 계산할 수 있다.

### 4. 데이터 확장과 품질 관리

- 라벨링 가이드에 따라 공개 가능한 합성·익명화 데이터로 확장한다.
  기존 문서의 목표 규모는 약 2,500~3,000건이다.
- 각 점수와 카테고리, PASS/BLOCK, 앱·발신자·작업 맥락 조합을 고르게
  늘리고 경계 사례와 긴급 알림을 충분히 포함한다. 라벨 불일치를
  사람이 검토하고 출처·사용 범위를 기록한다.
- 평가셋은 학습 전에 고정한다. 유사 문구, 동일 템플릿, 동일 상황이
  학습과 평가에 섞이지 않도록 그룹 단위로 분리한다. 평가셋의 오류를
  보고 만든 예시는 다음 실험용 데이터로 분리한다.

완료 기준: 분할별 건수와 라벨 분포, 중복 검사 결과, 검수 이력이 기록된다.
실제 개인 알림과 대용량 데이터는 Git에 올리지 않는다.

### 5. 본학습, 비교 평가, 채택 판단

- 고정된 train/validation/test 분할로 Qwen3-1.7B LoRA를 학습한다.
  validation으로 설정을 선택하고 test는 최종 비교에 사용한다.
- 2단계의 기본 모델과 같은 평가셋·프롬프트·디코딩 설정으로 비교한다.
  전체 평균과 함께 카테고리별, 긴급도별, 경계 사례별 결과를 확인한다.
- JSON 유효율과 긴급 알림 오차단을 우선 살핀다. 성능 기준은 평가셋
  규모와 기본 모델 결과를 확인한 뒤 명시한다. 추론 지연과 GPU 메모리도
  기록해 Windows 로컬 실행 가능성을 판단한다.
- 결과가 부족하면 오류 유형과 데이터 다양성을 먼저 수정하고 재학습한다.
  필요하면 QLoRA, 추론 최적화 또는 다른 학습 환경을 비교한다.

완료 기준: 기본 모델과 학습 모델의 재현 가능한 비교 보고서가 있고,
선택한 어댑터가 기존 `FilterResult` 계약으로 추론된다.

## 현재 스모크 실행 결과

Qwen3-0.6B 기본 모델과 BF16 LoRA 2단계 어댑터를 같은 test 3건에서
평가했다. 두 모델 모두 JSON 유효율 3/3, 정책 정확도 2/3이었다.
긴급 알림 1건은 통과했고, 차단해야 할 1건은 잘못 통과했다.
표본이 매우 작고 학습은 2단계뿐이므로 성능 변화로 해석하지 않는다.
어댑터 저장과 새 프로세스에서의 재로딩, 평가까지 동작함을 확인했다.
실행 설정과 집계 보고서는 `filtering_training/outputs`에 저장된다.

## 현재 선택과 다음 작업 (2026-09-28)

주 학습·평가 모델을 `Qwen/Qwen3-1.7B`로 변경했다. 0.6B의 과거 스모크
기록은 위에 보존한다. 로컬 한국어 후보는 3,000건이며 상황 유형을 기준으로
학습 2,398건, 검증 302건, 평가 300건으로 분리했다. 데이터셋과 모델은
로컬에만 보관하고 Git에서 제외한다.

별도 합성 24건에서 보강 데이터로 500단계 학습한 0.6B는 정책 정확도
16/24, 긴급 오차단 7/9였다. 1.7B는 정책 정확도 19/24, 긴급 오차단
5/9였다. 1.7B가 더 나아 이후 주 모델로 선택했으나 아직 실사용 모델은
아니다. RTX 3060 Laptop GPU에서 1.7B BF16 LoRA 학습이 동작했고,
500단계에 약 534초가 걸렸다. 알림당 추론 지연과 메모리는 별도 측정한다.

2026-09-29에 학습 2,398건 전체 1회 학습과 기존 개발용 24건 평가를 완료했다.
학습 시간은 약 16분 47초였다. 정책 정확도는 17/24, 긴급 오차단은 6/9로,
500스텝 결과보다 악화됐다. 전체 1회 어댑터를 새 기준 모델로 채택하지 않고
500스텝 결과를 이후 비교 기준으로 유지한다. 두 모델 모두 실사용 기준에는 부족하다.

다음 작업은 추가 2,000건을 8개 카테고리별 250건으로 만드는 것이다.
업무 알림은 IT 중심으로 유지하되 개인·광고·기타도 함께 늘린다. 기존 상황의
이름만 바꾸지 않고 짧은 메시지, 임박한 마감, 장애의 실제 영향, 작업과의
직접·간접·무관한 관계를 보강하고 라벨을 검토한다. 개발 평가 문장을 그대로
학습 데이터로 복제하지 않는다. 상황 계보로 분할 중복을 검사한 후 재학습한다.
반복적으로 사용한 24건은 개발 비교용으로 남기고, 최종 성능 보고용으로
미사용 평가셋을 별도 준비한다. 모든 실험은 입력·출력 예시 3~5건과 설정을
`docs/filtering-experiment-log.md`에 기록한다. 데이터셋과 모델은 계속 Git에서 제외한다.

## 데이터 보강 완료 (2026-09-29)

추가 2,000행을 생성해 통합 5,000행을 로컬에 저장했다. 카테고리별 625행이며,
새 알림 내용 1,000개에 서로 다른 맥락 두 개를 붙였다. 새 상황 유형 40종을
보강해 전체 상황 유형은 120종이다. 성능 개선은 아직 확인하지 않았다.

기존 분할을 보존해 학습 3,598행, 검증 702행, 테스트 700행으로 준비했다.
다음은 대표 라벨의 독립 검토, 확장 데이터의 제한된 LoRA 실험, 개발 비교,
검증셋에서 학습량 선택, 미사용 테스트/최종 평가 순서로 진행한다. 전체 1회
학습이 악화됐으므로 단순히 스텝 수를 늘리는 방식은 채택하지 않는다.
확장본 경로와 실제 입력·정답 예시 4개는 실험 기록에 적었다. 모델 출력은
재학습 평가를 실행한 뒤 별도로 기록한다. 모든 데이터와 어댑터는 로컬 보관한다.

## 4B 학습 전 비교 완료 (2026-09-30)

Qwen3-4B-Instruct-2507 사전 NF4 배포본을 로컬 GPU에서 기존 개발 24건에
평가했다. 정책 정답은 17/24로 1.7B 기본 모델과 같았고 긴급 오차단은 1/9,
불필요한 통과는 6/14였다. 카테고리 F1과 점수 오차는 기본 1.7B보다 좋아졌지만
바로 교체할 근거로는 부족하다. 기본 모델 설정은 유지했다.

다음 비교 후보는 같은 4B의 제한된 QLoRA 학습이다. 먼저 소규모 실행으로
6GB에서 학습 가능한지 확인해야 한다. 확장 데이터 학습 결과를 1.7B 기준과
비교하고 검증셋으로 학습량을 선택한다. 24건 반복 비교만으로 최종 채택하지
않으며 미사용 평가셋 검증을 남겨 둔다. 실험 기록에는 입력·출력 3개를 남겼다.

## 4B QLoRA 500스텝 결과와 다음 작업 (2026-09-30)

로컬 6GB GPU에서 4B QLoRA 2스텝 저장·재로딩 시험과 500스텝 학습을 완료했다.
확장 데이터의 학습 분할 3,598행 중 500스텝을 실행했고, 검증 702행의 loss는
250스텝 0.2815에서 500스텝 0.2525로 낮아졌다. 학습과 두 검증에 약 21분 10초가
걸렸다. 최대 CUDA 예약 메모리는 4.95GiB였다.

기존 개발용 24건의 통과·차단은 24/24, 긴급 오차단 0/9, 불필요한 통과 0/14였다.
카테고리는 18/24, macro F1 0.7440으로 오류가 남았다. 추론 생성은 평균 7.89초다.
이 24건은 데이터 확장 방향에 참고한 개발 평가이므로 최종 성능으로 해석하지 않는다.

현재 4B 500스텝 어댑터를 가장 나은 개발 후보로 보존한다. 다음 우선순위는
미사용 알림의 문장·라벨 검수와 독립 평가, 카테고리·설명 오류 분석, 추론 지연
측정 및 최적화다. 필요할 때 검증 데이터를 기준으로 추가 학습량을 비교한다.
500스텝 초과 학습과 전체 1회 학습은 아직 하지 않았고 기본 모델 설정도 유지했다.
입력·출력 예시 3개와 재로딩 시험 3개를 실험 기록에 남겼다. 커밋·푸시는 하지 않았다.

## Frozen 100-case candidate evaluation and follow-up (2026-09-30)

A new local, Git-ignored 100-case Korean synthetic evaluation set was written
before inspecting predictions. Its alert IDs and text do not duplicate the
5,000-row training candidate set, 24-case development set, or tracked samples.
It contains 12-13 cases per category, 27 urgent, 37 PASS and 63 BLOCK labels.
Gold labels are author-provisional and require another person's review.

The 1.7B LoRA candidate achieved 86/100 policy decisions, 73/100 exact
categories, and 3.21 s mean generation. The 4B NF4 QLoRA candidate achieved
94/100 policy decisions, 85/100 exact categories, and 7.70 s mean generation.
The 4B candidate still blocked 2/27 urgent cases, both imminent meetings.
Different datasets, model versions, LoRA targets and precision prevent a
model-size-only conclusion. Model defaults have not changed.

Next, review disputed gold labels and add fresh unseen examples for the six
remaining policy-error types. Generate analogous training examples without
copying the evaluation cases, choose any further QLoRA duration using the
training validation split, and reserve a second unseen set for the final check.
The inspected 100 cases are now development evidence, never training rows.
For the demo, the CLI now accepts a notification/context JSON file without a
gold label and a saved adapter. Three local probes worked, but end-to-end client
integration and response-time optimization remain. Benchmark model loading and
per-alert latency separately before changing the deployed candidate.

## Continued training decision (2026-09-30)

A separate 250-step QLoRA continuation from the original 4B adapter used the
same dataset, lower learning rate and fresh optimizer. Validation loss fell to
0.2465, but the inspected 100-case policy result regressed from 94/100 to
92/100. Preserve the original 500-step adapter as the development candidate.
Do not treat a lower validation loss as proof of better filtering. Further
training should follow label review, targeted new synthetic examples, and a
fresh unseen evaluation set. The inspected 100 cases remain excluded from
training, and the model default is unchanged.

## Additional synthetic stress check (2026-09-30)

A fresh 80-case, 10-per-category synthetic stress set was authored after the
first 100-case errors were known. It has no exact alert-text overlap with the
training candidates or earlier evaluation sets. On it, the original 4B 500-step
adapter made 76/80 policy decisions correctly and missed no urgent case,
while the 1.7B adapter made 65/80 policy decisions correctly. The 4B still
made false passes. This targeted result strengthens the 4B candidate comparison
but cannot replace independently reviewed real-world evaluation. Both sets
remain excluded from training. The original 4B adapter and default setting
remain unchanged; use the blind review sheets before another data revision.
