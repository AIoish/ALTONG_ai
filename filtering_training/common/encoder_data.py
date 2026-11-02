"""용도: KLUE 학습용 고정 점수 자료의 스키마와 무결성을 검증한다.
생성일: 2026-11-02
"""

import json
from pathlib import Path
from filtering_training.preparation.prepare_score_experiment import digest

# Retain the frozen source schema and descriptions without requiring its original model SDK.
SCORE_INSTRUCTIONS = {
    "urgency": (
        "알림의 현재 사건·시점·지연 영향을 근거로 긴급도 하나를 선택하세요. "
        "작업 관련성은 별개입니다. 이미 해결된 사건과 앞으로의 일정을 구분하세요. "
        "발신자·광고 마감만으로 높이지 말고 명시된 사실로 판단하세요."
    ),
    "relevance": (
        "알림이 현재 작업 대상·행동에 주는 도움을 선택하세요. 현재 창이 우선입니다. "
        "최근 창은 현재 목적의 보조 근거이며, 현재 제목이 모호할 때 작업 대상을 확인하는 데 쓰세요. "
        "목적을 알 수 없으면 추측하지 마세요. 단어 일치보다 실제 도움을 판단하세요."
    ),
}


SCORE_LEVELS = {
    "urgency": [
        "확인 시점이 자유로운 선택적 정보",
        "나중에 확인·대응 가능, 당장 영향 없음",
        "비교적 빠른 확인 필요, 즉각 중단 근거 없음",
        "현재 확인할 가치, 진행 중인 일이 응답을 기다림",
        "즉시 대응 필요, 진행 중인 안전 위험·중대한 피해",
    ],
    "relevance": [
        "현재 목적과 무관하거나 목적을 확인할 근거 없음",
        "같은 분야의 주변 정보·장식 광고, 작업 도움 거의 없음",
        "같은 주제의 잡담·일반 정보, 직접 도움 불분명",
        "현재 목적에 직접 유용한 자료·방법·비교 정보",
        "정확한 작업 대상·행동의 요청·결과·해결 절차",
    ],
}


def prepared_question(task):
    return {"type": "choice", "instructions": SCORE_INSTRUCTIONS[task],
            "criteria": {str(i): SCORE_LEVELS[task][i-1] for i in range(1, 6)}}

def load_native_prepared(folder):
    """Verify frozen v2 files and keep task labels and context separation intact."""
    manifest_file = folder / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format") != "native_laya_choice_v2" or manifest.get("prompt_version") != "v2":
        raise ValueError("expected native Laya v2 prepared inputs")
    # Model binaries and old experiment reports are provenance, not KLUE data inputs.
    # The manifest hash still protects their recorded values without requiring them on disk.
    hashes = {path: expected for path, expected in manifest["protected_hashes"].items()
              if not {"models", "runs"}.intersection(Path(path.replace("\\", "/")).parts)}
    hashes[str(manifest_file)] = digest(manifest_file)
    sources = {}
    for task in ("urgency", "relevance"):
        source = folder / task / "train.jsonl"
        expected = manifest["files"][task + "/train.jsonl"]
        hashes[str(source)] = expected["sha256"]
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        if len(records) != expected["count"] or len(records) != manifest["counts"][task]["total"]:
            raise ValueError("native prepared count changed")
        for row in records:
            if (set(row) != {"state", "questions", "expected"}
                    or row["questions"] != {task: prepared_question(task)}
                    or set(row["expected"]) != {task}
                    or row["expected"][task] not in {"1", "2", "3", "4", "5"}):
                raise ValueError("native prepared question or target changed")
            if ("id" in row["state"]["notification"]
                    or set(row["state"]) != ({"notification"} if task == "urgency" else {"notification", "context"})):
                raise ValueError("native prepared context separation changed")
        sources[task] = records
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"native prepared protected input changed: {path}")
    return sources, hashes
