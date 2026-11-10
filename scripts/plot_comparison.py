"""Bar chart: baseline vs data-only fine-tune vs PDE fine-tune."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = {
    "FNO baseline": ["fno40_seed0", "fno40_seed1", "fno40_seed2"],
    "data-only FT": ["finetune_data", "finetune_data_seed1", "finetune_data_seed2"],
    "PDE FT (1e-3)": [
        "finetune_pde1e-3",
        "finetune_pde1e-3_seed1",
        "finetune_pde1e-3_seed2",
    ],
}


def main() -> None:
    root = Path("results/runs")
    labels, means, stds = [], [], []
    for name, dirs in GROUPS.items():
        vals = []
        for d in dirs:
            m = json.loads((root / d / "metrics.json").read_text(encoding="utf-8"))
            vals.append(m["rollout_rel_l2"]["mean"])
        labels.append(name)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("rollout mean relative L2")
    ax.set_title("FNO fine-tuning comparison (3 seeds)")
    fig.tight_layout()
    fig.savefig(out / "finetune_rollout_comparison.png", dpi=160)
    plt.close(fig)
    print("wrote", out / "finetune_rollout_comparison.png")
    for label, m, s in zip(labels, means, stds):
        print(f"{label}: {m:.6f} ± {s:.6f}")


if __name__ == "__main__":
    main()
