"""Stable paths independent of a tool's package location."""

from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = TRAINING_ROOT / "outputs"
LEGACY_OUTPUTS_ROOT = OUTPUTS_ROOT / "archive" / "legacy_20261003"
ARCHIVED_OUTPUT_NAMES = frozenset({
    "audit", "candidates", "evaluation", "external", "holdout",
    "independent_eval_100", "independent_stress_80",
    "lora-rapid-qwen3-1_7b-500", "portability_probe", "portability_probe.py",
    "prepared", "prepared_rapid", "prepared_targeted_5000",
    "qlora-qwen3-4b-500-20260930", "review",
})


def resolve_existing_path(path: Path) -> Path:
    """Resolve a recorded pre-organization input path without rewriting history.

    Only missing inputs below the filtering outputs directory are redirected.
    Existing paths, external paths, and output destinations are left alone.
    """
    path = Path(path)
    if path.exists():
        return path
    try:
        relative = path.resolve().relative_to(OUTPUTS_ROOT.resolve())
    except ValueError:
        return path
    if relative.parts and relative.parts[0] in ARCHIVED_OUTPUT_NAMES:
        archived = LEGACY_OUTPUTS_ROOT / relative
        if archived.exists():
            return archived
    return path
