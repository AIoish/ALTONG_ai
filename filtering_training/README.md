# 필터링 학습 도구

필터링 데이터 생성, 검수, 학습, 평가에 사용하는 오프라인 도구입니다.
파일별 역할과 현재 사용 여부는 [FILE_GUIDE.md](FILE_GUIDE.md)를 참고하세요.

## 한국어 Encoder CPU 비교

모델 크기 비교에는 `--base outputs의_로컬_모델_경로 --fp32-only`를 사용합니다. 기존 실행 폴더는 덮어쓸 수 없습니다.
2026-10-10 비교 모델은 `klue-roberta-small`, `klue-roberta-large`, `jhu-clsp-mmBERT-small`이며 원본 revision은 각 모델 폴더의 `manifest.json`에 기록했습니다.
large의 6GB GPU 실행은 `--micro-batch 2 --attention sdpa --fused-adam --release-cuda-cache`와 `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.7,max_split_size_mb:64`를 사용합니다. 실효 배치는 16으로 유지합니다.


`modeling/encoder_score_experiment.py`는 최신 정제 학습 자료의 원본·라벨·분할을 유지하고
KLUE RoBERTa 또는 mmBERT와 긴급도·관련도 5클래스 헤드를 전체 학습합니다. 모델 가중치는 하나이며,
긴급도에는 알림만, 관련도에는 알림과 실제 현재·최근 창 정보를 넣어 각각 처리합니다.
작업 설명을 추측해 입력에 추가하지 않으며, 512토큰을 넘으면 잘라서 학습하지 않고 중단합니다.

```powershell
python -m filtering_training.modeling.encoder_score_experiment --action all --run filtering_training/outputs/runs/2026-10-10-klue-base-02
```

기존 실행 폴더를 덮어쓰지 않습니다. 최대 3에폭, 승인 검증 74행으로 체크포인트를 선택하고
최종 테스트는 사용하지 않습니다. 같은 체크포인트의 FP32 ONNX와 동적 INT8을 CPU 4스레드로
각 100회 측정합니다. 시간은 미리 토큰화한 입력의 두 점수 추론이며 앱 입력 수집 시간은 제외합니다.
메모리는 Torch를 불러오지 않는 독립 Python CPU 프로세스 전체 RSS로, C# 앱 전체 RAM과 다릅니다.
학습 조건이 다른 기존 Qwen/Laya 결과는 앱 교체 판단의 기준이며 구조만의 우열을 증명하지 않습니다.

모델·토크나이저·설정·에폭별 검증·변환 출처·CPU 예측은 해당 실행 폴더에 모읍니다.
실제 결과와 채택 여부는 [기존 실험 로그](../docs/filtering-experiment-log.md) 끝에 기록합니다.

2026-10-10 비교에서는 3에폭 체크포인트의 FP32 ONNX가 두 점수 평균 145ms,
통과·차단 67/74(90.5%)였다. 긴급도 점수 정확도 저하와 오통과 증가가 있어 앱 교체는 확정하지 않았다.
INT8 62ms와 full-range UINT8 96ms는 모두 통과·차단 64/74(86.5%)로 내려가 선택하지 않았다.
앱 연결을 검토할 후보는 해당 실행의 `model-fp32.onnx`와 `tokenizer/`이며,
입력 직렬화는 실행 코드의 `input_text`와 일치해야 한다. C# 모듈과 전달 안내는
[integration/csharp](../integration/csharp/README.md)에 있다. 실제 WPF 앱의 연결은 아직 수행하지 않았다.
기록된 실행 폴더는 이미 존재하므로 재실행할 때 새 폴더를 지정한다.

같은 자료로 small·large·mmBERT-small도 비교했다. 통과·차단은 small 65/74·large 61/74·mmBERT 65/74이며 CPU 평균은 각각 74·506·82ms다. 현재 base 67/74·145ms를 기준 후보로 유지한다.
보강 후보 10건은 `outputs/v8_encoder_boundary_review_01/review.md`에서 검수하며 현재 학습 자료에는 합치지 않았다.


## 폴더 구조

| 폴더 | 역할 |
| --- | --- |
| `common/` | 공통 데이터 읽기·학습 형식 변환·경로 처리 |
| `generation/` | 현재 v3 데이터 생성·확장 |
| `preparation/` | 데이터 분할·학습 형식 준비 |
| `quality/` | 데이터 형식·모순·문맥 편향 검사 |
| `modeling/` | 모델 학습·추론·평가 |
| `review/` | 원격 #14에서 추가된 정답 비공개 검수표·오류 검수 도구 |
| `legacy/` | 현재 v3에서 사용하지 않는 이전 실험 코드와 과거 안내 |
| `data/` | 기존 고정 입력 자료. 이번 정리에서 내용과 위치를 변경하지 않음 |
| `outputs/` | 데이터·가중치·검수표·실험 결과. [결과 안내](docs/ARTIFACTS.md) |

## 초기 v3 작업

```text
generation/generate_v3_seed.py
  → generation/expand_v3_dataset.py
  → preparation/prepare_reviewed_v3.py
  → modeling/train.py 또는 modeling/train_linear_baseline.py
  → modeling/evaluate.py
```

현재 데이터는 `outputs/v3_reviewed_01/`에 있습니다. 원문 66개·문맥 변형 198건이며,
학습 150건, 검증 24건, 테스트 24건입니다. 원문 24개를 사람이 개별 검수했고
나머지 42개는 합성 후보입니다. 테스트는 모델 선택에 사용하지 않았습니다.

- [최근 실험 결과와 날짜순 기록](../docs/filtering-experiment-log.md)
- 로컬 상세 결과: `outputs/v3_reviewed_01/pilot_summary.md`
- 초기 학습 어댑터는 2026-10-09 정리에서 삭제했고 결과와 설정은 보존했습니다.

`outputs/`와 로컬 검수·이동 기록은 Git에서 제외되므로 GitHub에는 위 로컬 파일이 없습니다.
공유할 성능 요약은 추적되는 실험 기록에 남깁니다.

정책은 긴급도 4 이상 **또는** 관련도 4 이상이면 통과입니다.
집중 모드 ON을 전제로 하며, 빈 창도 쉬는 상태로 추정하지 않습니다.
실제 앱의 공통 규칙과 출력 계약은 `src/filtering/`에 있습니다.

## 현재 점수 분리·누적 학습

Qwen3-0.6B 기반 모델 하나에 긴급도·관련도 LoRA 어댑터를 각각 사용합니다.
긴급도 입력은 알림만, 관련도 입력은 알림과 현재 창·최근 창 최대 3개입니다.
추론은 다섯 숫자 후보를 직접 비교하며 카테고리·이유 문장을 생성하지 않습니다.

- 학습 자료: `outputs/v5_training_pool_01/prepared_continual/`의 긴급도 4,936건·관련도 11,771건
- 최신 어댑터: `outputs/runs/2026-10-05-qwen06-accumulated-01/{urgency,relevance}/adapter/`
- 직전 어댑터: `outputs/runs/2026-10-05-qwen06-continual-01/{urgency,relevance}/adapter/`
- V7 관련도 보강 3천 건과 재검수 제안은 아직 기존 학습 자료에 병합하지 않았습니다.
- 학습·평가 실행: `modeling/digit_score_experiment.py`, 실험 실행 계획: `modeling/run_score_pilot.py`
- 데이터 준비: `preparation/prepare_digit_supplement.py`, 검사: `quality/audit_digit_training.py`

최신 설정은 유효 배치 8, 1 epoch, 학습률 5e-5, warmup 5%, LoRA rank 8입니다.
관련 실행은 새로운 출력 폴더를 지정하고 `--dry-run` 사전 점검부터 진행합니다.
이 프로젝트는 학습 도구 저장소이며 앱의 모델 연결을 자동으로 바꾸지 않습니다.

## Qwen CPU 앱용 형식과 실측

`outputs/models/qwen3-06b-cpu/`에 동일한 기반 모델 revision을 변환한 `base-Q8_0.gguf`,
학습된 `urgency-F16.gguf`·`relevance-F16.gguf`, Windows x64 CPU `runtime/`을 준비했습니다.
기반 모델 하나를 로딩하고 요청마다 어댑터 하나만 선택합니다. 원본 가중치·데이터는 보존합니다.
공식 llama.cpp `b11532` 변환 코드·배포물 출처와 해시는 같은 폴더 `manifest.json`에 있습니다.

```powershell
.venv\Scripts\python.exe -u -X utf8 -m filtering_training.modeling.benchmark_digit_cpu --output filtering_training/outputs/runs/qwen_cpu_next/report.json --threads 4 --count 100 --idle-seconds 30 --dry-run
```

`--dry-run`을 제거하면 CPU 전용 서버를 숨김 실행해 측정하고 종료합니다. 새 실행 폴더를 사용합니다.
100회는 고정 검증 74건과 반복 26회입니다. 정확도는 74건, 속도는 두 점수 요청을 포함한 100회로 계산합니다.
최초 실측은 `outputs/runs/2026-10-10-qwen06-cpu-01/report.json`·예측·자원 표본에 저장했습니다.
i7-11800H(8코어·16스레드), RAM 24GB, CPU 4스레드: 두 점수 평균 2.07초·p95 2.85초,
처리 CPU 평균 24.9%, 로딩된 대기 평균 0.12%(첫 1초 제외 시 0.03%), 모델 서버 RSS 평균 약 854~901MiB입니다.
통과·차단 66/74(89.2%), 긴급 누락 2/17이며 GPU NF4와 점수쌍은 63/74행 일치합니다.
CPU 백분율은 전체 논리 프로세서 기준입니다. RAM은 앱 UI·Python 측정 도구를 제외한 서버 working set입니다.
속도는 입력 템플릿을 미리 구성한 상주 모델의 순차 추론이며 실제 앱의 큐 대기·UI·입력 수집은 포함하지 않습니다.

앱은 같은 서버를 `-ngl 0 -t 4 -tb 4 -c 1024 -np 1 --poll 0 --poll-batch 0`과
두 `--lora`·`--lora-init-without-apply`로 실행하고 `127.0.0.1`로만 연결합니다.
기존 [점수 프롬프트](common/digit_scores.py)를 유지하고 긴급도에는 알림만,
관련도에는 알림과 현재 창·최근 창 최대 3개를 전달합니다. 알림의 변형 ID는 입력에서 제거합니다.
`POST /apply-template`의 `messages`와 `chat_template_kwargs: {"enable_thinking": false}`로 템플릿을 만든 뒤
`POST /completion`에 `prompt`, `n_predict: 1`, `temperature: 0`, `samplers: ["temperature"]`,
`grammar: "root ::= [1-5]"`, `repeat_penalty: 1`, `cache_prompt: false`를 전달합니다.
`lora: [{"id": 0, "scale": 1}]`은 긴급도, ID 1은 관련도입니다. 지정하지 않은 어댑터는 비활성입니다.
148개 템플릿의 토큰 일치를 확인했으며 앱 연결에는 Python 토크나이저가 필요하지 않습니다.
통과 규칙은 긴급도 ≥4 또는 관련도 ≥4입니다. 현재 저장소에서 앱 연결 코드 자체를 변경하지는 않았습니다.
API와 변환 근거는 [llama.cpp 공식 문서](https://github.com/ggml-org/llama.cpp/blob/b11532/tools/server/README.md)를 참고합니다.

## 검증

학습 의존성을 설치한 뒤 저장소 루트에서 실행합니다.

```powershell
python -m unittest discover -s tests/filtering -q
python -m unittest discover -s filtering_training/checks -q
```

로컬 `outputs/`가 없는 저장소에서는 실험 자료·어댑터에 의존하는 통합 테스트 11개를
명시적으로 건너뜁니다. 자료가 있으면 모두 실행하며 불완전하거나 변경된 자료는 실패합니다.
모델·데이터·실행 로그는 Git에서 제외됩니다. 학습을 이어받으려면 승인 자료·분할·manifest와
기반 모델 및 해당 어댑터를 별도로 전달받아야 합니다.

## Laya 다국어 모델 비교

`laya==0.4.1`과 `outputs/models/laya-multilingual/`의 고정 체크포인트를 준비한 뒤 실행합니다.
모델 버전과 로컬 위치는 [FILE_GUIDE.md](FILE_GUIDE.md)에 기록했습니다.
기본 평가는 추가 학습 전 Laya와 이미 학습된 Qwen의 같은 검증 74행을 비교합니다.
V7 보강·재검수 제안은 포함하지 않고, 기존 데이터·라벨·분할도 변경하지 않습니다.

```powershell
python -m filtering_training.modeling.laya_score_experiment --output filtering_training/outputs/runs/laya_next/report.json --dry-run
python -m filtering_training.modeling.laya_score_experiment --output filtering_training/outputs/runs/laya_next/report.json --device cuda
```

긴급도는 알림만, 관련도는 알림·현재 창·최근 창을 각각 입력합니다.
짧은 생성 문장 대신 5개 선택지의 `choice` 결과를 점수로 사용합니다.
기존 결과를 덮어쓰지 않도록 새 출력 위치를 지정하세요. CPU 측정은 `--device cpu`로 실행할 수 있습니다.

같은 기존 학습 자료로 Laya를 학습하려면 새 실행 폴더를 지정합니다. 변환 자료는 메모리에서
사용하고 원본 데이터 파일은 변경하지 않습니다. 긴급도·관련도 입력은 분리하되 모델 하나에 함께 학습합니다.

```powershell
python -m filtering_training.modeling.laya_score_experiment --action train --output filtering_training/outputs/runs/laya_train_next --dry-run
python -m filtering_training.modeling.laya_score_experiment --action train --output filtering_training/outputs/runs/laya_train_next --device cuda
python -m filtering_training.modeling.laya_score_experiment --model-path filtering_training/outputs/runs/laya_train_next/checkpoint --output filtering_training/outputs/runs/laya_train_next/validation.json --device cuda
```

기본 설정은 1 epoch·유효 배치 8, 인코더 LoRA 학습률 5e-5·분류 head 1e-4입니다.
5개 선택지에 대한 cross-entropy를 사용하고, 선택지 위치 편향을 줄이도록 정답의 의미를 유지하며
학습 중 선택지 순서를 섞습니다. 검증 자료로 학습·신뢰도 보정·중간 체크포인트 선택을 하지 않습니다.

학습 사례 자체를 맞히는지 확인하려면 같은 체크포인트에 `--action diagnose-train`을 사용합니다.
`--per-score` 기본값은 100으로 긴급도·관련도별 각 점수에서 최대 100개 입력을 추출합니다.
추가 학습 없이 적합도를 검사하며, 균형 학습 표본 정확도는 검증 정확도와 구분합니다.

### Laya 전용 질문 v2

정의는 `modeling/laya_score_experiment.py`의 `LAYA_INSTRUCTIONS_V2`·`LAYA_LEVELS_V2`에 모았습니다.
Qwen 공통 프롬프트를 변경하지 않고 `instructions`와 점수별 `criteria`를 분리합니다.
[공식 질문 문서](https://nandhakishorm.github.io/laya/questions-and-answers/)는 선택지 설명을 실제 모델 입력으로 사용하며,
[학습 문서](https://nandhakishorm.github.io/laya/finetune/)는 학습한 질문을 추론에도 사용할 것을 안내합니다.
길이를 늘리는 것 자체가 정확도를 높인다는 근거는 없으므로 핵심 지침이 잘리지 않도록 정의했습니다.

긴급도 질문: 알림의 현재 사건·시점·지연 영향을 근거로 긴급도 하나를 선택하세요.
작업 관련성은 별개입니다. 이미 해결된 사건과 앞으로의 일정을 구분하세요.
발신자·광고 마감만으로 높이지 말고 명시된 사실로 판단하세요.

관련도 질문: 알림이 현재 작업 대상·행동에 주는 도움을 선택하세요. 현재 창이 우선입니다.
최근 창은 현재 목적의 보조 근거이며, 현재 제목이 모호할 때 작업 대상을 확인하는 데 쓰세요.
목적을 알 수 없으면 추측하지 마세요. 단어 일치보다 실제 도움을 판단하세요.

| 점수 | 긴급도 선택지 | 관련도 선택지 |
| --- | --- | --- |
| 1 | 확인 시점이 자유로운 선택적 정보 | 현재 목적과 무관하거나 목적을 확인할 근거 없음 |
| 2 | 나중에 확인·대응 가능, 당장 영향 없음 | 같은 분야의 주변 정보·장식 광고, 작업 도움 거의 없음 |
| 3 | 비교적 빠른 확인 필요, 즉각 중단 근거 없음 | 같은 주제의 잡담·일반 정보, 직접 도움 불분명 |
| 4 | 현재 확인할 가치, 진행 중인 일이 응답을 기다림 | 현재 목적에 직접 유용한 자료·방법·비교 정보 |
| 5 | 즉시 대응 필요, 진행 중인 안전 위험·중대한 피해 | 정확한 작업 대상·행동의 요청·결과·해결 절차 |

새 학습은 기본 `v2`, 기존 체크포인트 평가는 실행 설정의 질문 버전을 자동 사용합니다.
기존 모델에서 질문만 바꾸는 진단은 `--prompt-version v2`로 명시합니다.
질문 정의 변경은 입력 분포 변경이므로 기존 모델의 앱 질문을 자동 교체하지 않습니다.
v2 학습 전에는 현재 창 우선 지침과 기존 학습 라벨의 일치도 확인해야 합니다.
원본 데이터·라벨은 이 변경으로 수정하지 않습니다. SDK 0.4.1에서 질문과 선택지가 잘리면 v2 실행을 중단합니다.

### Laya v2 두 절반 비교

`--action train-halves`는 사전 준비된 `run_config.json`의 제외 목록·그룹 분할·원본 해시를 확인하고,
첫 절반 학습→검증 1회→같은 모델과 옵티마이저로 나머지 학습→검증 1회를 실행합니다.
전체 합계 1 epoch이며 검증 결과로 두 번째 절반의 설정을 변경하지 않습니다.
완료 실행은 덮어쓰지 않습니다. 이 명령은 준비 설정이 있는 실행 폴더에서만 사용할 수 있습니다.

```powershell
.venv\Scripts\python.exe -u -X utf8 -m filtering_training.modeling.laya_score_experiment --action train-halves --output filtering_training/outputs/runs/2026-10-09-laya-v2-halves-01 --device cuda
```

위 실행은 완료됐습니다. `phase50/`·`phase100/` 각각에 `run_config.json`, `checkpoint/`,
`validation.json`, `validation.predictions.jsonl`을 저장했습니다. 최종 모델 경로는
`outputs/runs/2026-10-09-laya-v2-halves-01/phase100/checkpoint/`입니다.
충돌 관련도 4,161행은 학습에서 제외하고 원본 데이터·라벨은 보존했습니다.
두 단계 비교 결과와 실제 입출력은 기존 실험 로그 끝에 기록했습니다.
이 체크포인트의 `--action diagnose-train`은 제외 목록과 두 단계 학습 순서까지 확인합니다.
최신 적합도 결과는 `phase100/train-fit.json`과 예측 파일에 있으며 학습 표본 성능으로 해석합니다.

저관련도 보강 입력은 `outputs/v7_relevance_ladder_01/prepared_laya_v2/`에 있습니다.
긴급도 4,946행·관련도 10,173행이며 관련도 충돌 4,161행은 제외 상태를 유지합니다.
관련도에 기존 저관련도 후보 2,551행과 승인된 대조 12행, 긴급도에 승인된 원문 10개를 추가했습니다.
이 폴더는 native Laya v2 형식으로 원본 SFT 파일과 구분하며 파일·승인 근거·보존 해시는 manifest에 기록합니다.

```powershell
.venv\Scripts\python.exe -u -X utf8 -m filtering_training.modeling.laya_score_experiment --action train --prepared-inputs filtering_training/outputs/v7_relevance_ladder_01/prepared_laya_v2 --output filtering_training/outputs/runs/2026-10-09-laya-v2-low-replenished-01 --device cuda --epochs 1 --dry-run
```

위 명령은 입력·선택지·해시 검사만 수행합니다. 가중치를 로딩하거나 실행 폴더를 만들지 않습니다.
실제 학습은 2026-10-10 완료했습니다. `--dry-run`을 제거하면 고정 기반 모델에서 새로 학습하며,
기존 100% 체크포인트에 이어 학습하는 명령은 아닙니다. 완료된 실행 폴더는 덮어쓰지 않습니다.
평가와 적합도 진단은 저장된 질문·자료 출처를 확인합니다.
결과는 `outputs/runs/2026-10-09-laya-v2-low-replenished-01/`의 `validation.json`·`train-fit.json`과 예측 파일에 있습니다.
검증 통과·차단 43/74 (58.1%), 긴급 누락 16/17로 악화돼 앱에 채택하지 않습니다.

과제 비중 비교는 같은 입력으로 `--urgency-loss-share 0.3934321696158138`을 추가합니다.
이는 직전 4,936/12,546 비율로 총 명목 손실 가중치를 나누는 설정입니다.
긴급도 CE 가중치 1.20265·관련도 0.90147이며 행 반복·삭제 없이 평균 행 가중치 1을 유지합니다.
실제 gradient 크기의 비율을 고정하는 설정은 아닙니다. micro batch 1·soft CE에서만 지원합니다.
SDK 학습 순서·optimizer·scheduler를 유지하며 패키지 파일은 수정하지 않습니다.
실행 폴더는 `outputs/runs/2026-10-10-laya-v2-task-weighted-02/`입니다.
2026-10-10 완료: 긴급도 점수 일치 43.2%→51.4%, 통과·차단 58.1%→56.8%, 긴급 누락 16/17 유지입니다.
검증 충돌 후보 4행을 제외한 보조 정확도는 55.7%로 동일합니다. 이번 모델도 앱에 채택하지 않습니다.
비교와 실제 입출력은 기존 실험 로그 끝, 전체 예측은 실행 폴더에 있습니다.

## 실행 예시

저장소 루트에서 실행합니다. 학습·평가에는 설치된 학습 의존성이 필요합니다.
각 도구의 인자는 `python -m <모듈> --help`로 확인할 수 있습니다.

```powershell
python -m filtering_training.modeling.train --prepared-dir filtering_training/outputs/v3_reviewed_01/prepared --run-dir filtering_training/outputs/runs/qwen06_next --model Qwen/Qwen3-0.6B --max-steps 150 --max-length 768 --qlora
python -m filtering_training.modeling.evaluate --dataset filtering_training/outputs/v3_reviewed_01/dataset.jsonl --prepared-dir filtering_training/outputs/v3_reviewed_01/prepared --split validation --model Qwen/Qwen3-0.6B --adapter filtering_training/outputs/v3_reviewed_01/qwen06_pilot/adapter --load-in-4bit --output filtering_training/outputs/runs/qwen06_validation.json --show-examples 0
```

위 명령은 사용 예시이며 이번 정리 중 재학습하거나 재평가하지 않았습니다.
모델을 0.6B로 명시합니다. 일부 도구의 과거 기본값은 이번 구조 정리에서 변경하지 않았습니다.
새 실험은 별도 실행 폴더를 사용하고 기존 검수·가중치를 덮어쓰지 마세요.

## 호환과 보관

최신 `develop`의 #14에서 추가된 스트림 추론(`--stream`), JSON 입력(`--input-json`),
어댑터 추가학습(`--init-adapter`), 모델 비교·검수 도구를 보존했습니다.
`datasets/`, `checks/`, `external/` 및 기존 `generation/`의 모듈 경로도 호환됩니다.
원격의 기존 import와 로컬의 이전 명령을 모두 유지하기 위한 경로입니다.

예전 `python -m filtering_training.train` 및 기존 import도 유지됩니다.
`__init__.py`에서 옮긴 모듈을 찾을 경로를 제공합니다. 새 코드·명령에서는 역할별
패키지 경로를 사용합니다. 기존 모듈과 새 모듈 이름을 혼용한 monkeypatch는 피하세요.

이전 결과는 `outputs/archive/legacy_20261003/`로 이동했습니다.
공통 데이터 읽기와 모델 학습·평가의 입력 경로는 없어진 이전 경로를 해당 보관 위치에서 찾습니다.
이 기능은 누락된 입력을 찾는 용도이며 결과 저장 경로나 모든 역사 문서 링크를 자동 변경하지 않습니다.
일회성 과거 스크립트와 실행 로그의 경로 문자열은 당시 실행 기록으로 보존했습니다.

과거 명령과 설명은 [legacy/EXPERIMENTS.md](legacy/EXPERIMENTS.md)에 보관했습니다.
이번 구조 정리에서 데이터·가중치를 삭제하지 않았습니다.
