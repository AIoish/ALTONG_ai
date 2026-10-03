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
| `modeling/train_linear_baseline.py` | TF-IDF + LinearSVC 분류 기준선 학습 |
| `modeling/evaluate.py` | 기본 모델·어댑터의 점수와 통과/차단 평가, 개별 출력·시간 저장 |
| `modeling/infer_sample.py` | 개별 알림·JSON 파일·지속 실행 스트림 추론 및 출력 JSON 확인 |
| `modeling/compare_evaluations.py` | 원격 #14의 동일 데이터·프롬프트 전체 예측 비교 기능 |
| `review/prepare_label_review.py` | 원격 #14의 정답 비공개 검수표 및 별도 오류 검수 목록 |
| `checks/test_v3_workflow.py` | 새 v3 생성·승인·분할과 기존 승인 보존 검증 |

이 중 일반 분할·holdout·초기 형식 검사 도구는 최근 v3 생성 흐름의 필수 단계는 아니지만
공통 도구 또는 기존 테스트 대상이므로 유지합니다.

## 이전에 사용했고 현재 v3에서 사용하지 않는 코드

삭제 대상과 동일하지 않습니다. 과거 실험 재현과 기존 테스트를 위해 보관합니다.

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

## 삭제 여부

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
