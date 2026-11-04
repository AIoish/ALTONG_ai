<!-- 용도: Encoder 공통코드 분리·Qwen 도구 보관·최근 창 기반 학습 입력 전환을 설명한다.
생성일: 2026-11-04 -->

# 필터링 Encoder의 현재 학습 경로

현재 신규 학습은 `filtering_training.modeling.joint_urgency5_relevance3`를 사용한다.
긴급도는 5단계, 관련도는 3단계이며 기본 데이터는 `outputs/v13_previous_windows_relevance_01/`이다.
이번 변경은 코드·입력 계약 정리다. 새 모델을 학습하거나 앱에 배포한 결과가 아니다.

## 공통 기능

| 파일 | 역할 |
| --- | --- |
| `common/artifact_io.py` | JSON·JSONL·SHA256·프로젝트 경로·한국 날짜 |
| `common/encoder_input.py` | 최신/과거 입력 계약, 창 정보 정리, 토큰 길이 검사 |
| `common/validation_data.py` | 원본 라벨·승인·해시·분할 검증, 검증 74행 로딩 |
| `common/score_metrics.py` | 프롬프트와 무관한 기존 5단계 점수·통과 정책 지표 |
| `common/encoder_data.py` | 원본 학습 자료와 v13 학습·평가 파일 해시·라벨·입력 문자열 일치 검사 |

현재 Encoder 모듈은 Qwen의 `digit_score_experiment`, `digit_scores`, `score_tasks`, `score_loss`를
import하지 않는다. 검증 데이터를 만드는 데 자연어 프롬프트를 생성했다가 JSON으로 다시 읽는 경로도 제거했다.
원본 승인 데이터를 검증하는 스키마에는 과거 필드가 남아 있다. 이는 승인 당시 데이터의 무결성을 확인하기 위함이며,
새 모델 입력에 그 필드를 전달한다는 뜻은 아니다.

## 입력 계약

`notification-previous-windows-v3`가 새 학습의 입력 계약이다.

- 긴급도 입력: 알림 앱·발신자·제목·본문·시각. 알림 ID와 작업 맥락 제외.
- 관련도 입력: 같은 알림 + `current_window` + `recent_windows`.
- `recent_windows`는 프로그램 이름과 창 제목이다. `recent_processes`는 모델 입력에 넣지 않는다.
- 현재 창과 프로그램 이름·제목이 모두 같은 항목을 먼저 제외하고 남은 최신순 목록에서 최대 3개를 사용한다. 앱과 같이 대소문자를 구분한다. 같은 프로그램의 다른 제목은 유지한다.
- 수집된 목록 순서를 최신순으로 간주한다. 시각을 추측해 재정렬하거나 누락된 과거 창을 만들어 채우지 않는다.
- 원본 데이터의 다른 필드·라벨·행 순서·관련도 복습 표본은 변경하지 않는다.
- 512토큰 초과 입력은 조용히 잘라서 학습하지 않는다.

v13 초기 입력은 15,119행(긴급도4,946/관련도10,173), 안전 보강 입력은
8,946행(긴급도6,446/관련도2,500)이다. v13 파일을 실제 읽고 저장된 `input_text`를
공통 생성 결과와 비교하므로 필드가 남은 입력 문자열이나 바뀐 라벨은 거부한다.

검증74행·안전 개발40행·기존 진단100행·경계 진단14행도 새 모델에는 같은 계약으로 입력한다.
원본 파일을 덮어쓰지 않고 입력용 복사본에서 중복 필드와 현재 창을 제거한다.
이 228행의 실제 입력을 v13의 `evaluation/`에 저장하고, 해시 및 승인 원본에서 같은 변환을 적용한 결과와 대조한다.
실제 학습 실행 시 이 복사본은 새 run의 `evaluation-inputs/`에 해시와 함께 저장한다.

## 기존 모델 비교

과거 모델은 `notification-history-v1` 입력으로 비교한다. 이 계약에는 `recent_processes`가 들어간다.
기존 run_config에 `input_schema`가 없으면 과거 계약으로 간주한다. 알 수 없는 계약은 거부한다.
v12의 `notification-windows-v2`도 과거 입력 재현용으로 보존한다. 이 계약은 `recent_processes`만 제외하고 창 목록은 그대로다. v12를 새 학습에 직접 지정하면 v13 변환을 요구하며 중단한다.
과거 모델에 새 입력을 몰래 적용해 기존 성능 재현 조건을 바꾸지 않는다.

`evaluate_encoder_cases.py`, `prepare_csharp_fixtures.py`, 10월 5/5 이어학습 도구는 기존 모델용이다.
이 도구들은 과거 입력 계약을 명시적으로 유지한다. 새 5/3 모델과 기존 C#의 `[batch,2,5]` 출력은 여전히 다르다.
앱 스키마·C# 입력 생성기·앱 모델은 이번 작업에서 변경하지 않았다.

## 실행

프로젝트 가상환경을 사용한다. 이 PC의 기본 `python`에는 pydantic 등 의존성이 없을 수 있다.

```powershell
.\.venv\Scripts\python.exe -m filtering_training.preparation.prepare_previous_windows

.\.venv\Scripts\python.exe -m filtering_training.modeling.joint_urgency5_relevance3 `
  --dataset-dir filtering_training/outputs/v13_previous_windows_relevance_01 `
  --run filtering_training/outputs/runs/2026-11-04-klue-previous-windows-01 `
  --dry-run
```

`--dry-run`은 행 수·라벨·복습 표본·평가 중복을 검사하고 출력 폴더나 가중치를 만들지 않는다.
변환 도구도 기존 출력 폴더는 덮어쓰지 않는다. v13이 이미 준비돼 있으면 변환 명령을 생략한다.
실제 학습을 결정한 뒤 `--dry-run`을 빼고, 기존에 없는 새 run 경로로 실행한다.
위 명령으로 기존 체크포인트를 이어 학습하지 않는다.

## 과거 도구 정리

기존 학습·추론·대조 CLI 13개를 `legacy/modeling/`으로, Qwen 프롬프트/숫자 loss 헬퍼
3개를 `legacy/common/`으로 옮겼다. 폐기된 관련도 전용 학습·ONNX 변환 코드와 전용 검사·문서는 삭제했다.
구형 생성 도구·원본 데이터·검수·평가·가중치는 삭제하지 않았다.

현재 Encoder 의존성은 `filtering_training/requirements.txt`, 과거 Qwen 추가 의존성은
`filtering_training/legacy/requirements.txt`로 분리했다. 설치된 패키지는 제거하지 않았다.

기존 `filtering_training.modeling.train`, `filtering_training.train`,
`filtering_training.common.score_tasks` 등의 import/CLI는 패키지 검색 경로로 호환한다.
신규 Encoder가 이 호환 경로를 사용할 필요는 없다. 필요하면 과거 도구를
`filtering_training.legacy.modeling.<도구>`로 명시해 실행할 수 있다.

이동 전 로컬 파일 조사표는 개인 작업 자료로 Git에서 제외한다. 현재 코드의 위치는 이 문서를 따른다.

## 공통코드 분리 당시 확인 결과 (v12)

- 학습 도구 회귀 검사 66개, 기존 필터링 회귀 검사 35개 통과.
- 기존 검증74행의 원본·그룹·JSON·토큰·기존 점수 지표가 공통화 전과 일치.
- v12 초기15,119행·안전8,946행의 라벨·순서·복습 표본 확인. 실제 최대 입력371토큰.
- 검증·개발·진단228행은 기존 라벨을 유지하고 중복 맥락 필드만 제거.
- 기존 데이터·모델119개 파일의 SHA256 보존 확인.
- 기존 도구의 import 호환 확인. 최신 Encoder import는 Qwen 도구나 프롬프트를 불러오지 않음.
- 사용자 모델 재학습·앱 수정·모델 배포는 수행하지 않음.

로컬 검증 근거는 `outputs/_maintenance/refactor-windows-only-2026-11-04/verification.json`과
`moves.json`에 있다. 기존 모델 대비 품질 향상 여부는 새 학습·평가를 해야 알 수 있다.

## 앱 기준 창 목록 적용 결과 (v13)

- 현재 창과 같은 항목이 있던 관련도 4,910행에서 해당 항목을 제거했다. 창 목록 필드가 없던 47행은 빈 배열로 명시했다. 총 4,957행이며 라벨과 다른 필드는 보존했다.
- 안전 보강의 관련도 복습 2,500행 중 1,122행도 같은 규칙으로 변환했다. 초기·안전 학습의 긴급도 행은 v12와 완전히 같다.
- 검증·개발·진단 228행을 별도 평가 입력 파일로 저장하고 같은 변환을 적용했다. 학습과 평가 간 정규화 본문 중복은 없다.
- 폐기 실험 코드와 전용 검사 3개 제거 후 학습 도구 검사 67개와 기존 필터링 검사 35개, 총 102개가 통과했다. 새 데이터로 학습 전 점검(`--dry-run`)도 통과했다.
- 실제 학습·평가 입력의 최대 길이는 371토큰이다. 기존 데이터·모델 119개와 변환 원본 16개 파일의 해시가 유지됐다.
- 원본 라벨은 수정하지 않았다. 현재 창 제거로 근거가 달라진 사례의 라벨 적합성 및 모델 품질 개선은 별도 평가가 필요하다.
- 모델 재학습·앱 수정·배포는 실행하지 않았다. 검증 기록은 `outputs/_maintenance/previous-windows-2026-11-04/verification.json`에 있다.
