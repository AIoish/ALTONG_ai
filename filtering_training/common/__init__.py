"""Shared dataset and filesystem helpers for filtering training."""

from pathlib import Path

# Existing experiment commands remain importable; current Encoder tools do not use these helpers.
__path__.append(str(Path(__file__).resolve().parents[1] / 'legacy/common'))
