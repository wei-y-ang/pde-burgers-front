"""Plot a saved rollout example from training/evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, default="results/runs/fno/rollout_example.npz")
    p.add_argument("--out-dir", type=str, default="results/figures/fno")
    args = p.parse_args()

    z = np.load(args.npz)
    u_true = z["u_true"][0]
    u_pred = z["u_pred"]
    x = z["x"]
    t = z["t"]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, arr, title in [
        (axes[0], u_true, "truth"),
        (axes[1], u_pred, "prediction"),
    ]:
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            extent=[x.min(), x.max(), t.min(), t.max()],
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "rollout_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx in [0, len(t) // 4, len(t) // 2, 3 * len(t) // 4, len(t) - 1]:
        ax.plot(x, u_true[idx], color="k", lw=1.2, alpha=0.7)
        ax.plot(x, u_pred[idx], lw=1.0, ls="--", label=f"t={t[idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("rollout profiles (solid: truth, dashed: prediction)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "rollout_profiles.png", dpi=160)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
