# 필터링 학습·평가 도구

실시간 알림 필터링을 위한 데이터 준비, 학습, 평가 코드를 관리합니다.
앱에서 사용하는 스키마·프롬프트·정책은 `src/filtering`에 있습니다.
명령은 저장소 루트에서 실행합니다.

## 폴더 구성

```text
filtering_training/
  train.py                 Qwen LoRA·QLoRA 학습
  evaluate.py              모델 평가
  infer_sample.py          JSON 입력·스트림 추론
  compare_evaluations.py   평가 결과 비교
  datasets/                데이터 검증·감사·학습/평가 분할
  generation/              합성 데이터 생성·5,000행 확장
  external/                외부 후보 선택·번역·검토
  review/                  라벨 검수표 작성
  checks/                  정책 데모·모델 로딩 시험
  docs/                    상세 실행 안내·산출물 보관 안내
  data/                    로컬 데이터와 추적 중인 샘플
  outputs/                 모델·평가 결과·과거 실험 보관본
```

`__init__.py`의 `TRAINING_ROOT`를 기준으로 기본 데이터·산출물 경로를 계산합니다.
폴더를 옮겨도 데이터와 모델의 저장 위치는 기존 `data/`, `outputs/`를 사용합니다.

## 설치와 기본 확인

공통 환경과 CUDA PyTorch 설치는 저장소 루트 README를 참고합니다.

```powershell
python -m pip install -r filtering_training/requirements.txt
python -m filtering_training.datasets.validate_dataset
python -m filtering_training.checks.demo_policy
python -B -m unittest discover -s tests/filtering -v
```

## 자주 사용하는 명령

| 작업 | 실행 명령 |
| --- | --- |
| 샘플을 SFT 데이터로 변환 | `python -m filtering_training.datasets.prepare_dataset` |
| 데이터 품질 점검 | `python -m filtering_training.datasets.audit_dataset --help` |
| 평가셋 준비 | `python -m filtering_training.datasets.prepare_holdout --help` |
| 기존 3,000행 생성 | `python -m filtering_training.generation.generate_rapid_dataset --help` |
| 5,000행 확장 | `python -m filtering_training.generation.generate_targeted_dataset --help` |
| 상황 유형별 분할 | `python -m filtering_training.datasets.prepare_rapid_dataset --help` |
| 라벨 검수표 작성 | `python -m filtering_training.review.prepare_label_review --help` |
| 학습·평가·추론·비교 | 상위 폴더의 네 모듈에 `--help` 사용 |

생성·분할 명령은 출력 파일을 만들므로, 기존 실험을 보존하려면 상세 안내에서
입력·출력 경로를 확인합니다. 모델 로딩 시험은
`python -m filtering_training.checks.smoke_test_model`이며 모델 다운로드·GPU 사용이 발생할 수 있습니다.

## 상세 문서

- [기존 학습·평가·데이터 작업 안내](docs/WORKFLOWS.md): 이전 README의 설명과 실험 기록을 보존했습니다.
- [모델·데이터 위치와 과거 실험 복원](docs/ARTIFACTS.md)
- [필터링 실험 기록](../docs/filtering-experiment-log.md)
- [필터링 라벨링 기준](../docs/filtering-labeling-guideline.md)

2026-09-30 폴더 정리로 보조 도구의 모듈 경로가 바뀌었습니다.
예: `filtering_training.prepare_dataset` → `filtering_training.datasets.prepare_dataset`.
기존 명령·개인 스크립트는 위 폴더 구성을 기준으로 경로를 바꿔 실행합니다.
압축 보관한 과거 스크립트도 복원 후 import 경로를 갱신해야 합니다.
학습·평가·추론·결과 비교의 네 가지 상위 모듈 경로는 같습니다.

현재 Qwen 기본 모델과 학습 방식은 기존 설정을 사용합니다.
RoBERTa 분류 모델 실험은 별도 후속 작업입니다.
