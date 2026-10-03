"""Offline filtering tools, with compatibility for former flat module names.

New commands use the role packages (for example, filtering_training.modeling.train).
The package search path also exposes the moved modules at their former names so
existing tests and recorded ``python -m filtering_training.train`` commands work
without editing files outside this package. No tool is imported eagerly.
"""

from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parent
_root = TRAINING_ROOT
_compatibility_folders = (
    "generation", "preparation", "quality", "modeling",
    "legacy/generation", "legacy/preparation", "legacy/review",
    "legacy/quality", "legacy/external", "legacy/diagnostics",
)
__path__.extend(str(_root / folder) for folder in _compatibility_folders)
