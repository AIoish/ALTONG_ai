# 필터링 파일 안내

## 최신 구조 안내 (2026-10-03)

현재 v3 데이터와 실험은 `outputs/v3_seed/`, `outputs/v3_expansion_01/`, `outputs/v3_reviewed_01/`에 있습니다.
이전 결과는 `outputs/archive/legacy_20261003/`로 묶었으며 아래 경로에도 반영했습니다.
4B 후보와 CPU 도구에 관한 아래 설명은 이전 실험 이력입니다. 현재 v3 파일럿은 0.6B를 사용합니다.
로컬 결과·가중치·검수표는 Git에 올리지 않습니다. 공유할 결과 요약은
[실험 기록](../../docs/filtering-experiment-log.md)에 있습니다.

2026-09-30 기준. 아래 경로는 `filtering_training/`을 기준으로 표시한다.
이번 정리는 필터링 영역에 한정했다. 브리핑 코드·데이터·테스트와 공용 환경은 정리 대상에서 제외했다.

## 현재 사용하는 파일

| 경로 | 용도 |
| --- | --- |
| 상위 학습·평가 모듈, `datasets/`, `generation/`, `external/`, `review/`, `checks/`, `requirements.txt` | 데이터 생성·검증·학습·평가·추론 도구. 기존 테스트와 실험 재현에 필요하므로 유지 |
| `data/` | 필터링 원본·샘플·평가 데이터 |
| `outputs/archive/legacy_20261003/candidates/`, `outputs/archive/legacy_20261003/audit/` | 합성 데이터 후보와 품질 점검 결과 |
| `outputs/archive/legacy_20261003/prepared_targeted_5000/` | 5,000행 확장 데이터의 학습·검증·테스트 분할 |
| `outputs/archive/legacy_20261003/prepared/`, `outputs/archive/legacy_20261003/prepared_rapid/`, `outputs/archive/legacy_20261003/holdout/` | 이전 데이터 분할과 실험 재현용 manifest |
| `outputs/archive/legacy_20261003/evaluation/` | 이전 모델의 평가 지표와 예측 예시 |
| `outputs/archive/legacy_20261003/independent_eval_100/`, `outputs/archive/legacy_20261003/independent_stress_80/` | 개발 비교용 평가셋, 예측, 라벨 검수표. 학습에 혼합하지 않음 |
| `outputs/archive/legacy_20261003/lora-rapid-qwen3-1_7b-500/` | 비교 기준인 1.7B 500스텝 어댑터와 설정 |
| `outputs/archive/legacy_20261003/qlora-qwen3-4b-500-20260930/` | 현재 4B 개발 후보의 최종 어댑터, 학습 설정과 로그 |
| `outputs/models/` | 4B NF4 기본 모델. 위 4B 어댑터를 실행하는 데 필요 |
| `outputs/archive/legacy_20261003/portability_probe/`, `outputs/archive/legacy_20261003/portability_probe.py` | CPU 시험용 GGUF 기본 모델, llama.cpp 실행 파일, 다운로드 검증 기록과 시험 스크립트 |
| `outputs/archive/legacy_20261003/external/` | 외부 데이터의 출처·검토 자료. 이용 범위와 라벨 검토 상태를 확인한 뒤 사용 |
| `outputs/archive/` | 과거 실험의 압축 보관본 |
| `outputs/_maintenance/` | 이번 정리에 사용한 로컬 점검·보관 스크립트 |

NF4 기본 모델과 CPU용 GGUF 기본 모델은 서로 다른 실행 형식이다.
GGUF 파일에 4B QLoRA 어댑터의 학습 결과가 포함된 것으로 간주하면 안 된다.
모델·원본 데이터·로컬 산출물은 계속 Git에서 제외한다.

## 이번에 보관한 과거 실험

보관 위치는 `outputs/archive/20260930-before-encoder/historical-experiments.zip`이다.
각 파일의 기존 경로·크기·SHA-256 및 정리 결과는 같은 폴더의 `manifest.json`에 기록했다.
ZIP의 모든 파일을 원본과 대조한 후 기존 위치의 파일을 제거했다.
아래는 최초 보관한 실험 목록이다. 2026-10-09 구형 모델 파일을 정리한 이후에는
각 실험의 설정·로그·학습 경과 기록과 일회성 스크립트만 ZIP에서 복원할 수 있다.

- 초기 0.6B 시험: `lora-smoke`, `lora-smoke-bf16`, `lora-smoke-51`
- 0.6B 파일럿: `lora-rapid-pilot-100`, `lora-rapid-pilot-500`, `lora-rapid-short-500`
- 1.7B 전체 1회 학습: `lora-qwen3-1_7b-epoch1-20260928`
- 4B 시험·추가 학습: `qlora-qwen3-4b-smoke-20260930`, `qlora-qwen3-4b-continue250-20260930`
- 4B 후보의 Trainer 체크포인트: `qlora-qwen3-4b-500-20260930/trainer/`
- 완료된 데이터·문서 작업용 스크립트 9개: `add_experiment_inputs.py`, `append_holdout_log.py`,
  `check_reference_privacy.py`, `compare_real_samples.py`, `create_holdout_draft.py`, `mix_holdout.py`,
  `replace_holdout_log.py`, `rewrite_holdout_realistic.py`, `verify_holdout_log.py`

최초 압축본에는 각 실험의 가중치·설정·로그와 Trainer의 optimizer 상태도 포함했다.
2026-10-09 사용자 승인으로 압축본 안의 구형 가중치·optimizer·토크나이저 등 50개를 제거했다.
학습 설정·로그·`trainer_state.json`·`training_args.bin`·스크립트 50개는 내용이 동일하다.
압축본은 256,766,187바이트에서 178,146바이트로 줄었으며 256,588,041바이트를 확보했다.
`manifest.json`의 현재 파일 목록·압축본 해시를 갱신하고 `model_cleanup`에 삭제 내역과
이전 요약을 보존했다. 압축본과 보관 목록을 제외한 출력 파일 801개의 SHA-256도 동일하다.
과거 모델 가중치는 복원할 수 없다. 최신·직전 0.6B 모델과 현재 데이터셋에는 영향이 없다.

## 보존한 기록 복원 방법

저장소 루트에서 아래 명령으로 새 복원 폴더에 압축을 푼다.
이미 `restored`가 있다면 다른 새 폴더 이름을 사용한다.

```powershell
Expand-Archive -LiteralPath filtering_training/outputs/archive/20260930-before-encoder/historical-experiments.zip -DestinationPath filtering_training/outputs/archive/20260930-before-encoder/restored
```

압축 내부 경로는 원래 `outputs/` 아래의 경로와 동일하다.
필요한 실험 폴더를 복원 폴더에서 원래 `outputs/` 위치로 옮긴다.
같은 이름의 폴더가 새로 생성되어 있다면 덮어쓰지 말고 별도 위치에서 비교한다.
일회성 스크립트는 데이터를 수정하거나 실험 문서를 다시 작성할 수 있으므로, 복원만으로 실행하지 않는다.

## 2026-09-30 최초 보관 당시 삭제 및 검증 결과

- 압축 보관: 실험·체크포인트 폴더 10개와 일회성 스크립트 9개, 총 100파일.
- 압축 전 426,456,592바이트 → 압축 후 256,766,187바이트.
- 중복 다운로드 삭제: `llama-b11269-bin-win-cpu-x64.zip`, `llama-b11282-bin-win-cpu-x64.zip`.
  ZIP 안의 모든 파일을 이미 풀어 둔 해당 버전의 실행 파일과 SHA-256으로 비교했다.
  실행 파일은 보존했다. 다운로드 시험 스크립트를 다시 실행하면 ZIP이 다시 생성될 수 있다.
- 캐시 삭제: `filtering_training`, `src/filtering`, `tests/filtering` 아래의 `__pycache__` 4개, 총 37파일.
  Python 실행 시 다시 생성될 수 있다.
- 확보한 파일 용량: 208,409,050바이트, 약 **199MiB**.
  새 안내 문서·manifest의 소량 용량과 파일시스템 할당 단위는 제외한 값이다.
- 데이터·평가 결과·비교 모델·소스·문서·브리핑 영역 등 보존 대상 234파일의 SHA-256과
  파일 수를 정리 전후 대조했다. 모두 동일했다.

학습 코드와 모델 동작은 변경하지 않았다.
검증은 보관본의 내용 일치 및 보존 파일의 무결성 확인으로 수행했다.
