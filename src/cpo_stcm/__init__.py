# SPDX-License-Identifier: AGPL-3.0-only
"""CPO-STCM research software for normal-only animation anomaly detection.

The package intentionally avoids importing PyTorch, OpenCV, Ultralytics, or
other optional runtime dependencies at package import time. Heavy symbols are
resolved lazily so path utilities, conformal calibration, and metadata tools
remain usable in minimal CPU environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0rc1"

__all__ = [
    "LSTMSequenceAutoencoder",
    "TemporalWindowDataset",
    "temporal_reconstruction_scores",
]

_LAZY_EXPORTS = {
    "LSTMSequenceAutoencoder": "cpo_stcm.temporal",
    "TemporalWindowDataset": "cpo_stcm.temporal",
    "temporal_reconstruction_scores": "cpo_stcm.temporal",
}


def __getattr__(name: str) -> Any:
    """Resolve heavy optional symbols only when explicitly requested."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
