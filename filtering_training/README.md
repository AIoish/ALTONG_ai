# 필터링 학습 도구

필터링 데이터 생성, 검수, 학습, 평가에 사용하는 오프라인 도구입니다.
파일별 역할과 현재 사용 여부는 [FILE_GUIDE.md](FILE_GUIDE.md)를 참고하세요.

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
