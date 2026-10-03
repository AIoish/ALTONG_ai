"""Compatibility paths for filtering tools moved into role packages.
Updated: 2026-10-03 (Asia/Seoul).
"""

from filtering_training.common.paths import TRAINING_ROOT

__path__.extend(str(TRAINING_ROOT / folder) for folder in ('legacy/diagnostics',))
