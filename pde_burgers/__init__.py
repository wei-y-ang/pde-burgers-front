"""Neural operators for PDEBench 1D Burgers front prediction.

The package is intentionally small: one dataset, two baselines (FNO1D and a
1D U-Net), and a reproducible one-step/rollout evaluation pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Local fallback used during the initial scaffold.  In a clean environment,
# prefer `pip install -r requirements.txt`; `.deps` is git-ignored.
_DEPS = Path(__file__).resolve().parents[1] / ".deps"
if _DEPS.exists() and str(_DEPS) not in sys.path:
    sys.path.append(str(_DEPS))

__version__ = "0.1.0"
