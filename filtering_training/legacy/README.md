# 이전 필터링 실험 코드

현재 데이터 생성·학습 흐름의 기본 도구는 아니지만 기존 테스트가 사용하는 이전 코드입니다.
2026-10-09 사용자 승인으로 참조가 없는 과거 코드 10개를 삭제했습니다.

- `generation/`: `generate_rapid_dataset.py`, `generate_targeted_dataset.py`
- `preparation/`: 이전 대량 후보 학습 파일 준비
- `external/`: 외부 자료 선택·번역·번역 검사
- [EXPERIMENTS.md](EXPERIMENTS.md): 기존 긴 README의 실험 이력·명령 안내

처음부터의 날짜순 실험 기록은 [filtering-experiment-log.md](../../docs/filtering-experiment-log.md)입니다.

일반 도구로 다시 활용할 때에는 현재 데이터·정책에 맞게 입력과 인자를 지정해야 합니다.
현재 사용 여부와 파일별 역할은 [전체 안내](../FILE_GUIDE.md)에 있습니다.
이전 결과의 로컬 보관 위치는 `outputs/archive/legacy_20261003/`입니다.
