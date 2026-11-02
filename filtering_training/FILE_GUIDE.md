# 파일 역할과 사용 상태

## 현재 사용 또는 공통 기반

| 파일 | 역할 |
| --- | --- |
| `common/dataset.py` | 스키마와 출력 계약을 검증해 데이터 읽기, SFT 대화 형식 변환 |
| `common/paths.py` | 파일 이동과 무관한 기준 경로, 보관된 이전 입력 경로 찾기 |
| `generation/generate_v3_seed.py` | 새 원문 16개·문맥 변형 48건과 검수 자료 생성 |
| `generation/expand_v3_dataset.py` | 새 원문 48개를 추가해 64개·192건으로 확장, 경계 사례 검수표 생성 |
| `preparation/prepare_reviewed_v3.py` | 승인 기록 저장, 보충 원문 2개 포함, 원문·문맥·지정 유사 사례를 묶어 분할 |
| `preparation/prepare_dataset.py` | 일반 데이터 분할과 SFT 파일 생성, 기존 공통 함수 import 호환 |
| `preparation/prepare_holdout.py` | 별도 평가용 데이터 검사와 평가 대상 목록 작성. 기존 평가 도구 |
| `quality/validate_dataset.py` | 초기 샘플 데이터의 필수 필드·점수 범위 등 형식 검사 |
| `quality/audit_dataset.py` | 카테고리·점수 분포, 중복·모순 검사 |
| `quality/audit_context_shortcut.py` | 창 정보만으로 관련도 정답을 추측할 수 있는 편향 검사 |
| `modeling/train.py` | Qwen LoRA/QLoRA 학습과 어댑터 저장 |
| `modeling/encoder_score_experiment.py` | KLUE·mmBERT 공유 Encoder 전체 학습·점수 헤드 검증·ONNX/INT8 변환·CPU 100회 비교 |
| `modeling/train_linear_baseline.py` | TF-IDF + LinearSVC 분류 기준선 학습 |
| `modeling/evaluate.py` | 기본 모델·어댑터의 점수와 통과/차단 평가, 개별 출력·시간 저장 |
| `modeling/infer_sample.py` | 개별 알림·JSON 파일·지속 실행 스트림 추론 및 출력 JSON 확인 |
| `modeling/compare_evaluations.py` | 원격 #14의 동일 데이터·프롬프트 전체 예측 비교 기능 |
| `review/prepare_label_review.py` | 원격 #14의 정답 비공개 검수표 및 별도 오류 검수 목록 |
| `checks/test_v3_workflow.py` | 새 v3 생성·승인·분할과 기존 승인 보존 검증 |

이 중 일반 분할·holdout·초기 형식 검사 도구는 최근 v3 생성 흐름의 필수 단계는 아니지만
공통 도구 또는 기존 테스트 대상이므로 유지합니다.

## 이전 코드 보관 이력

아래 표는 2026-10-03 보관 당시 목록이다. 2026-10-09 사용하지 않는 코드 일부를 삭제했으며,
현재 남아 있는 코드는 `legacy/README.md`, 정확한 삭제 목록은 이 문서 마지막 기록을 따른다.

| 파일 | 당시 역할 |
| --- | --- |
| `legacy/generation/generate_synthetic_candidates.py` | 초기 소규모 합성 후보 생성 |
| `legacy/generation/generate_rapid_dataset.py` | 패턴 조합으로 한국어 후보 3,000건 생성 |
| `legacy/generation/generate_targeted_dataset.py` | 보강 후보 2,000건 생성 및 이전 5,000건 구성 |
| `legacy/generation/prepare_message_diversity.py` | 채팅 말투 다양화 후보와 검수표 생성. 이름은 prepare지만 실제 역할은 생성 |
| `legacy/preparation/prepare_rapid_dataset.py` | 이전 대량 생성 후보의 시나리오 그룹 분할 |
| `legacy/review/prepare_review_sheet.py` | 이전 5,000건의 대표·경계 사례 검수표 생성 |
| `legacy/review/prepare_review_batch.py` | 큰 검수표를 작은 검수 묶음으로 축소 |
| `legacy/review/prepare_context_counterfactual_review.py` | 같은 창에 다른 알림을 붙인 관련도 대비 검수표 생성 |
| `legacy/review/apply_review_feedback.py` | 이전 데이터에 명시적인 점수 수정과 변경 이력 반영 |
| `legacy/quality/debias_duration.py` | 이전 데이터의 체류 시간에 의한 정답 추측 편향 완화 |
| `legacy/external/select_external_candidates.py` | 외부 NotifAI 원문 후보 선택 |
| `legacy/external/translate_external_candidates.py` | 외부 후보 기계 번역 |
| `legacy/external/audit_external_translations.py` | 번역 후보 검사 |
| `legacy/external/adapt_external_pilot.py` | 외부 알림 아이디어를 한국어 합성 사례로 재작성 |
| `legacy/diagnostics/smoke_test_model.py` | 과거 모델 로딩·GPU·기본 생성 점검 |
| `legacy/diagnostics/demo_policy.py` | 모델 없이 점수에 따른 통과/차단 예제 실행 |
| `legacy/EXPERIMENTS.md` | 정리 전 긴 README의 실험 설명과 실행 방법 보관 |

일반 검수 기능은 다음 데이터 버전에서 다시 활용할 수 있습니다. 이번에는 v3에서 사용하지
않는 이전 기본 경로·양식의 도구를 명시적으로 분리했습니다.

## 입력 자료와 결과 자료

현재 기본 학습 자료는 `outputs/v7_relevance_ladder_01/prepared_laya_v2/`이다. 추가 검수 후보 10건은 `outputs/v8_encoder_boundary_review_01/review.md`이며, 확정·병합 전 상태다.

2026-10-10 안전 보강 모델은 `outputs/runs/2026-10-10-base-safety-continue-01/`이다.
`generation/prepare_boundary_supplement.py`는 소규모 3/4 경계·관련도 대조쌍 84행과 개발·새 평가를 생성한다.
`modeling/continue_encoder_boundary.py`는 부모 모델을 후보 0으로 보존하고 기존 성능이 떨어지는 추가 학습을 채택하지 않는다.
이번 `outputs/runs/2026-10-10-base-boundary-continue-01/`의 학습 에폭 1은 기존 정책 정확도 하락으로 제외했다.
`modeling/prepare_csharp_fixtures.py`는 Python의 입력·토큰·점수를 고정해 C#과 비교한다.
재사용 C# 모듈·실제 앱 계약 어댑터·검증 프로그램·전달 생성기는 저장소의 `integration/csharp/`에 둔다.
현재 전달 ZIP은 `outputs/handoff/ALTONG_filtering_csharp_2026-10-10_actual.zip`이며 기본값은 실제 차단·통과다.
연결 방법은 `integration/csharp/README.md`를 따른다.


`data/`의 세 기존 파일은 내용과 위치를 유지했습니다. 실제 알림 샘플 내용은 이번 정리에서
열거나 학습에 사용하지 않았습니다. 결과 폴더별 설명은 [ARTIFACTS.md](docs/ARTIFACTS.md)를 참고하세요.

| 파일 이름 | 의미 |
| --- | --- |
| `candidates.jsonl` | 검수·확정 전 후보 데이터 |
| `dataset.jsonl` | 원문·문맥·정답을 담은 해당 버전 확정 데이터 |
| `model_io.jsonl` | 모델 입력 및 예상 출력 예시 |
| `review.csv`, `review.md` | 사람이 볼 검수표 |
| `new_notifications.csv` | 확장하면서 추가한 원문 목록 |
| `approval.json`, `seed_approval.json` | 승인 발언과 실제 개별 검수 범위 |
| `manifest.json` | 건수·ID·분할·해시 등 구성 정보 |
| `lineage.json`, `*.lineage.json` | 생성·변형·변경 이력 |
| `audit.json`, `split_audit.json` | 데이터·분할 자동 검사 결과 |
| `prepared/train.jsonl` | 모델 학습용 대화 형식 데이터 |
| `prepared/validation.jsonl` | 개선 판단용 검증 데이터 |
| `prepared/test.jsonl` | 최종 평가를 위해 남겨 둔 데이터 |
| `run_config.json` | 학습 설정·버전·loss·장비 기록 |
| `validation_report.json`, `report.json` | 성능 지표 |
| `validation_predictions.jsonl` | 개별 입력·정답·실제 예측 |
| `policy_errors.csv`, `error_analysis.json` | 틀린 통과/차단과 오류 분석 |
| `pilot_summary.md`, `pilot_comparison.json` | 사람이 읽는 결과와 프로그램용 비교 지표 |
| `adapter_model.safetensors` | 학습된 LoRA 가중치 변경분. 기반 모델과 함께 사용 |
| `adapter_config.json` | 어댑터 구성과 기반 모델 정보 |
| `tokenizer.json`, `tokenizer_config.json` | 토큰 변환 규칙과 설정 |
| `chat_template.jinja` | 모델에 넣는 대화 형식 |
| `*.joblib` | 저장된 선형 분류 모델 |
| `*.stdout.log`, `*.stderr.log` | 실행 출력·오류 기록 |
| `*.gguf`, 기반 모델 `*.safetensors` | 기반 모델 파일 |
| `llama-*.exe`, `*.dll` | 과거 CPU 실행 실험용 실행 파일·라이브러리 |

## 2026-10-03 정리 당시 삭제 여부

이번 정리에서 삭제가 꼭 필요한 파일은 없다고 판단했습니다. 삭제한 파일은 없습니다.

| 구분 | 판단 |
| --- | --- |
| `__pycache__/`, `*.pyc` | 재생성 가능한 캐시. 향후 삭제 후보지만 이번에는 유지 |
| 과거 stdout/stderr 로그 | 결과 지표가 별도 보존됐고 디버깅이 끝났다면 삭제 검토 가능. 이번에는 보관 |
| 중복 버전 CPU 실행 도구 | `portability_probe/` 안의 이전 실행 도구들은 현재 v3에서 미사용. 필요한 버전을 결정한 뒤 삭제 검토 가능 |
| 이전 후보 데이터·검수 의견·가중치·해시·변경 이력 | 실험 근거이므로 현재 미사용이어도 보존 |

실제 삭제를 진행하기 전에는 대상과 영향부터 사용자에게 설명합니다.

## 이동 이력

코드 28개와 과거 결과 353개 파일을 이동했습니다. 현재 v3의 기존 43개 파일은 SHA-256이
동일합니다. 보관 파일 모두 크기를 확인했고 352개는 SHA-256도 확인했습니다.
나머지 한 개는 대형 GGUF 파일이며 같은 파일시스템 안에서 이동하고 크기를 확인했습니다.

상세 이전·이후 경로와 검증 결과는 로컬
`outputs/_maintenance/organization_20261003.json`에 있습니다. 이 결과 파일은 Git에 포함하지 않습니다.
보관 폴더의 과거 기록에 들어 있는 경로 문자열은 당시 이력으로 유지했으므로,
현재 위치는 이동표를 기준으로 찾습니다.

최신 develop의 #14를 통합하면서 스트림 추론·추가학습·모델 비교·검수 기능도 보존했습니다.
`datasets/`, `checks/`, `external/`의 기존 모듈과 이전 생성 모듈은 호환 경로로 유지합니다.
해당 기존 위치의 구현은 역할별·legacy 폴더에 두고, 원격 구현의 로컬 보관본은
`outputs/_maintenance/upstream_layout_20261003/`에 있습니다.

## 2026-10-09 로컬 모델 정리 검토

Laya 다국어 모델과 최신 Qwen 0.6B를 비교하기 위해 프로젝트 안의 모델 파일을 전수 확인했다.
아래는 검토 당시 삭제 후보 목록이다. 승인 후 실행 결과는 이 절 마지막에 기록했다.
용량은 파일 크기 합계의 십진 단위다.

| 대상 | 용량 | 판단·삭제 영향 |
| --- | ---: | --- |
| `outputs/models/qwen3-4b-instruct-2507-bnb-4bit/` | 2.669GB | 과거 4B 기반 모델. 최신 0.6B 비교에는 필요 없고 삭제 후 4B 재실행은 모델 재다운로드 필요 |
| `outputs/archive/legacy_20261003/portability_probe/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2.497GB | 과거 4B CPU 시험용 모델. 삭제 후 해당 시험 재실행은 GGUF 재다운로드 필요 |
| 같은 폴더의 `llama-cpu-b11269/`, `llama-cpu-b11282/` | 98.46MB | 과거 CPU 실행 도구. 현재 숫자 분류 추론에서 사용하지 않음 |
| 아래 과거 어댑터·체크포인트 34곳 | 543.83MB | 삭제하면 해당 과거 가중치로 추론·학습 재개 불가. 실험 결과·설정·데이터는 별도로 보존 |
| `outputs/runs/2026-10-05-qwen06-accumulated-01/`의 두 `adapter/` | 32.07MB | 보존: 최신 학습 결과, Laya 비교 기준·향후 Qwen 누적 학습 시작점 |
| `outputs/runs/2026-10-05-qwen06-continual-01/`의 두 `adapter/` | 32.07MB | 보존 권장: 직전 결과를 재실행하는 비교 기준 |

큰 4B 모델 두 개만 정리하면 약 5.17GB를 확보한다. 과거 CPU 도구와 아래 모델 폴더까지
정리하는 넓은 범위의 합계는 약 5.81GB다. 어댑터 폴더 용량에는 복제된 토크나이저도 포함된다.
실험 상위 폴더 전체를 삭제하는 목록이 아니며 데이터·검수·설정·평가 기록은 유지한다.
체크포인트 안의 `trainer_state.json` 등 경과 기록을 별도로 남기면 실제 확보량은 표보다 조금 작다.

과거 모델 폴더 목록(모두 `outputs/` 기준):

```text
v3_reviewed_01/qwen06_pilot/adapter
runs/2026-10-03-qwen06-separation-smoke-01/urgency/adapter
runs/2026-10-03-qwen06-separation-smoke-01/relevance/adapter
runs/2026-10-03-qwen06-score-pilot-01/urgency/adapter
runs/2026-10-03-qwen06-score-pilot-01/scores/adapter
runs/2026-10-03-qwen06-score-pilot-01/relevance/adapter
runs/2026-10-03-qwen06-score-pilot-01/full/adapter
runs/2026-10-04-qwen06-control-smoke-02/urgency/adapter
runs/2026-10-04-qwen06-score-controls-01/standard_nll/urgency/adapter
runs/2026-10-04-qwen06-score-controls-01/standard_nll/relevance/adapter
runs/2026-10-04-qwen06-score-controls-01/numeric_weight/urgency/adapter
runs/2026-10-04-qwen06-score-controls-01/numeric_weight/relevance/adapter
runs/2026-10-04-qwen06-score-controls-01/balanced/urgency/adapter
runs/2026-10-04-qwen06-score-controls-01/balanced/relevance/adapter
runs/2026-10-04-qwen06-pool-02/urgency/adapter
runs/2026-10-04-qwen06-pool-02/relevance_id_clean/adapter
runs/2026-10-04-qwen06-digit-threshold-01/urgency/adapter
runs/2026-10-04-qwen06-digit-supplement-01/supplement/urgency/adapter
runs/2026-10-04-qwen06-digit-supplement-01/supplement/relevance/adapter
runs/2026-10-04-qwen06-digit-supplement-01/matched_steps/urgency/adapter
runs/2026-10-04-qwen06-digit-supplement-01/matched_steps/relevance/adapter
runs/2026-10-04-qwen06-digit-smoke-01/urgency/adapter
runs/2026-10-04-qwen06-digit-fit-01/urgency/adapter
runs/2026-10-04-qwen06-digit-epochs3-01/urgency/adapter
runs/2026-10-04-qwen06-digit-01/natural/urgency/adapter
runs/2026-10-04-qwen06-digit-01/natural/relevance/adapter
runs/2026-10-04-qwen06-digit-01/balanced/urgency/adapter
runs/2026-10-04-qwen06-digit-01/balanced/relevance/adapter
archive/legacy_20261003/review/qwen06_v2_smoke/adapter
archive/legacy_20261003/review/qwen06_v2_pilot300/adapter
archive/legacy_20261003/review/qwen06_v2_pilot300/trainer/checkpoint-200
archive/legacy_20261003/review/qwen06_v2_pilot300/trainer/checkpoint-300
archive/legacy_20261003/qlora-qwen3-4b-500-20260930/adapter
archive/legacy_20261003/lora-rapid-qwen3-1_7b-500/adapter
```

현재 0.6B 기반 모델은 프로젝트 밖의 Hugging Face 캐시에 있다:
`%USERPROFILE%/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`.
이 기반 모델도 최신 Qwen 비교에 필요하므로 이번 삭제 후보에서 제외한다.

### 승인 후 삭제 결과 및 이전 데이터셋 확인

2026-10-09 사용자 승인에 따라 위 목록의 과거 모델 파일 255개를 삭제했다.
확보한 용량은 **5,808,219,709바이트(5.81GB)**다. 과거 어댑터·체크포인트 34곳의
가중치·토크나이저·옵티마이저 등을 제거하고, `adapter_config.json`, `README.md`,
`trainer_state.json`, `training_args.bin`은 있는 위치에 보존했다.
최신·직전 어댑터 가중치 4개를 포함해 나머지 출력 파일 816개의 SHA-256이 삭제 전과 같다.
데이터셋·검수·학습 설정·평가 기록·실험 로그는 삭제하지 않았다.

데이터 관련 버전 폴더 전체 용량은 약 240.67MB다. 이전 버전이라는 이유로 일괄 삭제하지 않는다.

| 자료 | 확인 결과와 권장 처리 |
| --- | --- |
| `v4_diverse_5000_candidates_01/`, `v4_diverse_5000_candidates_02/` | 초기 미학습 합성 후보, 합계 13.83MB. 현재 학습·검증 및 다른 출력 JSON에서 이 두 폴더를 직접 참조하지 않음. 삭제 후보로 분류하되 과거 후보 내용 재확인은 불가능해짐. 생성 CLI 기본 출력은 01, 과거 계획 문서는 02를 가리키므로 삭제 여부를 기록할 필요가 있음 |
| `v3_reviewed_01/` | 현재 코드가 원본 해시·분할을 확인하며 최종 테스트도 보관하므로 유지 |
| `v5_training_pool_01/`, 특히 `prepared_continual/` | 최신·직전 모델의 실제 학습 자료와 재검수 기준이므로 유지 |
| V3/V4의 승인 완료 검수 묶음 | V5 재학습 자료의 원본·승인 이력이므로 유지 |
| `v5_validation_pool_01/`, `v5_validation_review_01/`, `v5_history_review_01/` | 현재 검증 자료 및 평가 시 원본·승인 해시 확인에 필요하므로 유지 |
| `v4_diverse_5000_candidates_03/`, `v4_training_pool_01/` | 현재 생성·검사 코드 또는 보호 자료 목록에 참조가 남아 있어 폴더 전체 삭제 보류 |
| `v6_targeted_pool_01/` | V7의 중복 검사 및 보존 원본 해시 목록에서 참조하므로 유지 |
| `v7_relevance_ladder_01/` | 신규 3천 건 보강 및 재검수 제안 자료. 아직 기존 학습 데이터에 병합하지 않았으므로 유지 |

V5의 `dataset.jsonl`과 `diversified.jsonl`, `candidates.jsonl`과 `revised.jsonl`은
크기가 같아도 SHA-256이 각각 다르다. 수정 단계별 원본이므로 중복 파일로 취급하지 않는다.
데이터셋 삭제는 이번 승인 범위와 구분해 확인만 했으며 아직 실행하지 않았다.

### 2026-10-09 추가 승인 후 데이터·미사용 코드 삭제

사용자의 추가 승인으로 초기 후보 `v4_diverse_5000_candidates_01/`,
`v4_diverse_5000_candidates_02/`를 삭제했다. 아래 미사용 코드도 삭제했다.
합계 23개 파일, **13,936,477바이트(13.94MB)**를 추가 정리했다.
현재 데이터·승인·평가·모델 등 나머지 출력 파일 802개의 SHA-256은 동일하다.

- `legacy/generation/`: `generate_synthetic_candidates.py`, `prepare_message_diversity.py`
- `legacy/review/`: `prepare_review_sheet.py`, `prepare_review_batch.py`,
  `prepare_context_counterfactual_review.py`, `apply_review_feedback.py`
- `legacy/quality/debias_duration.py`
- `legacy/external/adapt_external_pilot.py`
- `legacy/diagnostics/`: `smoke_test_model.py`, `demo_policy.py`
- `outputs/_maintenance/`: `cleanup-20260930.ps1`, `integrate_remote_20261003.py`, `organize_20261003.py`
- `outputs/archive/legacy_20261003/portability_probe.py`: 기반 모델과 실행 도구를 이미 삭제한 과거 4B CPU 시험 코드

현재 테스트가 직접 사용하는 이전 생성·분할·외부 자료 검사 코드와 그 의존성 6개는 유지했다.
과거 통합 과정의 코드 보관본은 관리 이력으로 남기며 실행 대상이 아니다.
삭제한 초기 후보 내용 및 이전 도구 실행은 현재 작업 폴더에서 제공하지 않는다.

실험 전체의 날짜순 기록은 프로젝트 루트의 `docs/filtering-experiment-log.md`다.
첫 항목은 2026-09-26 LoRA 시험, 최신 학습 항목은 2026-10-05 누적 배치 8·warmup 비교다.
`legacy/EXPERIMENTS.md`는 이전 README에서 옮긴 실행 안내이며 전체 실험 기록을 대체하지 않는다.

정리 후 기존 필터링 테스트 35개와 최근 학습 도구 테스트 48개가 모두 통과했다.
과거 가중치를 참조하던 사전 점검 테스트 두 개는 보존한 최신 모델 또는 임시 LoRA 파일을
사용하도록 수정했으며, 실제 Qwen 모델 학습·평가는 실행하지 않았다.

### 2026-10-09 과거 실험 압축본의 모델 파일 정리

사용자 승인으로 `outputs/archive/20260930-before-encoder/historical-experiments.zip`
안의 구형 가중치·옵티마이저·토크나이저 등 50개만 제거했다.
설정·로그·학습 경과 기록·스크립트 50개는 정리 전후 SHA-256이 같고,
압축본 CRC 검사도 통과했다. 256.77MB에서 178.15KB로 줄여 **256.59MB**를 확보했다.
압축본과 같은 폴더의 `manifest.json`을 갱신했으며, 그 외 출력 파일 801개의
SHA-256이 동일하다. 최신·직전 모델, 데이터셋, 실험 로그, 코드, 캐시 및 Git 작업 트리는
변경하지 않았다. 해당 압축본에서 과거 모델 가중치를 복원하는 기능은 없어졌다.
보관 목록과 `docs/ARTIFACTS.md` 및 이 안내만 현재 상태에 맞춰 갱신했다.

## 2026-10-09 Laya 비교 작업

`modeling/laya_score_experiment.py`는 기존 Qwen 검증 원문·문맥·정답·예측 해시를 확인하고,
같은 승인 검증 74행을 Laya 다국어 모델로 평가한다. `--dry-run`은 모델을 로딩하거나
출력 폴더를 생성하지 않는다. `--reverse-options`는 점수 정의를 유지하며 선택지 순서만 바꾼다.
긴급도·관련도 입력을 각각 전달하고 다섯 `choice` 라벨을 1~5 점수로 해석한다.
`--action train`은 기존 Qwen 학습 입력을 메모리에서 Laya 선택지·정답 형식으로 변환한다.
인코더 attention의 `Wqkv`·`Wo`에 LoRA를 학습하고 분류 head도 함께 업데이트한다.
기존 파일은 덮어쓰지 않으며 검증·최종 테스트와 V7 제안은 학습에 넣지 않는다.
기본 동작은 `--action evaluate`이며 기반 모델과 프로젝트 학습 완료 모델을 모두 평가한다.

모델: `outputs/models/laya-multilingual/`, SDK: `laya==0.4.1`.
고정된 기반 모델 버전은 `1720e3e3357cfe1e281542e223f8273b0890ca34`다.
초기 평가 결과는 `outputs/runs/2026-10-09-laya-multilingual-base-01/report.json`과
`report.predictions.jsonl`에 있다. 기존 자료와 V7 재검수 제안은 변경하거나 병합하지 않았다.

학습 실행 폴더에는 `run_config.json`과 `checkpoint/`를 저장한다. 최종 체크포인트는 LoRA를
합친 일반 Laya 형식이므로 앱 추론에서 어댑터 전환이나 별도의 두 모델을 요구하지 않는다.
학습 자료 건수·원본 해시·변환 내용 해시·기반 버전·설정·시간·최종 가중치 해시를 기록하며,
학습을 마친 체크포인트는 저장된 출처와 원본 자료 해시가 일치할 때만 비교한다.

첫 학습 결과는 `outputs/runs/2026-10-09-laya-multilingual-lora-01/`에 있다.
`validation.json`은 학습된 Laya, `qwen-reference.json`은 같은 날 재평가한 최신 Qwen의 결과다.
각 `.predictions.jsonl`에 전체 입력과 실제 점수를 보존했고 비교표는 기존 실험 로그 끝에 추가했다.
`train-fit.json`과 `train-fit.predictions.jsonl`은 같은 저장 모델의 학습 사례 적합도 진단이다.
점수별 균형 표본 결과이므로 검증·최종 테스트 성능으로 해석하지 않는다.

Laya 전용 질문 v2도 같은 코드의 `LAYA_INSTRUCTIONS_V2`·`LAYA_LEVELS_V2`에서 관리한다.
`--prompt-version v2` 비교 결과는 같은 실행 폴더의 `prompt-v2-validation.json`과 예측 파일에 있다.
기존 모델의 기본 질문은 학습 시 버전으로 유지하며, 새 학습의 기본 질문만 v2다.

`--action train-halves`는 준비된 그룹별 두 절반을 같은 모델·옵티마이저·스케줄러로 이어 학습한다.
이번 실행은 `outputs/runs/2026-10-09-laya-v2-halves-01/`이며 루트 `run_config.json`에
학습 제외 4,161행·그룹 분할·누적 학습 결과를 기록했다. 원본 데이터·라벨은 보존한다.
`phase50/`·`phase100/` 각각에 설정·`checkpoint/`·`validation.json`·전체 예측을 저장했다.
최종 모델은 `phase100/checkpoint/`다. 검증 충돌 후보 4행을 제외한 지표는 같은 예측의 보조 집계다.
같은 폴더의 `phase100/train-fit.json`·`train-fit.predictions.jsonl`은 실제 학습 입력의 점수별 균형 표본 진단이다.
진단은 학습 제외 목록과 단계별 학습 순서로 변환 해시를 재현하므로 제외한 행을 평가에 넣지 않는다.

`outputs/v7_relevance_ladder_01/current_priority_preview.jsonl`은 4,161행 제외 후 준비한
저관련도 대조 원문 10개·최근 창 변형 2행의 미승인 후보다. 같은 폴더 `review.md` 끝에 검수표,
`manifest.json`의 `current_priority_preview`에 상태·해시·중복 검사·묶음 기준을 기록했다.
같은 manifest의 `current_priority_semantic_review`와 기존 `review_decisions.jsonl`의
`current_priority_v2_3000` 회차는 기존 3천 후보의 생성틀·작업/자료 관계 AI 재검토다.
이전 제안·사용자 승인·원본을 보존하며 수정 제안 113행과 판단 불일치 39행은 미적용 상태다.

`outputs/v7_relevance_ladder_01/prepared_laya_v2/`는 저관련도 보강을 반영한 native Laya v2 학습 입력이다.
`urgency/train.jsonl` 4,946행, `relevance/train.jsonl` 10,173행, 공통 `manifest.json`에
원본 선택 인덱스·추가 ID·검수 범위·제외 목록·파일 해시·검증 보존·검사 결과를 모았다.
기존 모델과 데이터는 보존하며 이 자료의 실제 학습은 2026-10-10 완료했다.
기존 Laya 실행 코드의 `--action train --prepared-inputs <이 폴더> --dry-run`으로 사전 검사를 재현한다.
완료된 보강 모델은 `outputs/runs/2026-10-09-laya-v2-low-replenished-01/checkpoint/`이며,
같은 실행 폴더의 `validation.json`·`train-fit.json`과 예측 파일에 비교·적합도 진단을 보존했다.
검증 악화로 앱 채택하지 않는다. 준비 당시 manifest는 입력 출처로 보존하고 V7 상위 manifest에 실제 사용 결과를 기록했다.

`outputs/runs/2026-10-10-laya-v2-task-weighted-02/`는 같은 입력·설정에서 과제별 CE 가중치만 바꾸는 비교 실행이다.
`run_config.json`에 긴급도 명목 손실 비중 39.343%와 과제별 가중치·SDK 루프 해시를 기록한다.
`task-weighted-01/`은 인자 오류로 첫 업데이트 전 종료한 실행 기록만 보존한다.
2026-10-10 완료했으며 `checkpoint/`, `validation.json`·`train-fit.json` 및 예측에 결과를 저장했다.
고정 검증 통과·차단 42/74, 긴급 누락 16/17로 앱에 채택하지 않는다.

`modeling/benchmark_digit_cpu.py`는 기존 Qwen 점수 모델의 GGUF CPU 정확도·100회 지연·로딩된 대기 자원을 측정한다.
`outputs/models/qwen3-06b-cpu/`에 `base-Q8_0.gguf`(805MB)·두 점수 LoRA(각 2.3MB)·CPU runtime·변환 출처 manifest를 보관한다.
`conversion-source/`·변환 로그는 재현용이며 실제 앱에서 읽는 모델은 세 GGUF 파일이다.
`outputs/runs/2026-10-10-qwen06-cpu-01/`의 `report.json`, 예측 100행·자원 표본·`server.log`에 실측을 저장했다.
앱용 API 입력 방법은 README의 Qwen CPU 항목을 참고한다. 원본 모델·자료·최종 테스트는 변경하지 않았다.
