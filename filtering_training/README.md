# 필터링 학습 도구

> 2026-11-04: 현재 신규 학습은 **긴급도5/관련도3 공동 Encoder + v13 데이터**를 사용한다.
> `recent_processes`를 제외하고 `recent_windows`는 현재 창을 제외한 최신순 최대 3개로 학습·검증·평가를 통일한다. 공통코드 분리와 과거 도구 이동은
> [현재 학습 안내](../docs/filtering-windows-only-2026-11-04.md)를 따른다.
> 아래 10월 비교 결과와 Qwen 실험 경로는 당시 기록이며 신규 학습 명령이 아니다.

필터링 데이터 생성, 검수, 학습, 평가에 사용하는 오프라인 도구입니다.
파일별 역할과 현재 사용 여부는 [FILE_GUIDE.md](FILE_GUIDE.md)를 참고하세요.

## 한국어 Encoder CPU 비교

현재 사용 모델은 KLUE이며 Qwen·Laya 전용 비교 도구와 Laya SDK 의존성은 포함하지 않습니다.
기존 고정 자료의 `prepared_laya_v2` 경로·스키마 이름은 데이터 보존을 위해 유지합니다.
CPU 100회 측정은 승인 검증 74건 전체와 앞 26건 반복으로 구성하며 과거 Qwen 실행 결과를 요구하지 않습니다.

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
