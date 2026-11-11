"""Aggregate FNO/U-Net resolution-generalization evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="results/resolution")
    p.add_argument("--out-dir", type=str, default="results/resolution_summary")
    args = p.parse_args()

    rows = []
    for model in ["fno", "unet"]:
        for seed in range(3):
            for resolution in [1024, 512, 256]:
                path = (
                    Path(args.input_dir)
                    / f"{model}_seed{seed}_x{resolution}"
                    / "metrics.json"
                )
                if not path.exists():
                    raise FileNotFoundError(f"missing {path}; run all resolution evaluations")
                m = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "model": model.upper(),
                        "seed": seed,
                        "resolution": resolution,
                        "rollout_mean": m["rel_l2"]["mean"],
                        "rollout_100": m["rel_l2"]["at_100"],
                        "front_position_mae_aligned": m["front"]["position_mae_aligned"]["mean"],
                        "front_speed_mae": m["front"]["speed_abs_error"]["mean"],
                    }
                )

    per_run = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out / "per_run.csv", index=False)

    summary = (
        per_run.groupby(["model", "resolution"])
        .agg(
            n_seeds=("seed", "count"),
            rollout_mean=("rollout_mean", "mean"),
            rollout_std=("rollout_mean", "std"),
            rollout_median=("rollout_mean", "median"),
            rollout_min=("rollout_mean", "min"),
            rollout_max=("rollout_mean", "max"),
            rollout_100_mean=("rollout_100", "mean"),
            front_position_mae_aligned=("front_position_mae_aligned", "median"),
            front_speed_mae=("front_speed_mae", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    groups = {model: per_run[per_run["model"] == model] for model in ["FNO", "UNET"]}
    x = np.array([256, 512, 1024])

    fno = groups["FNO"]
    fno_stats = fno.groupby("resolution")["rollout_mean"]
    axes[0].errorbar(
        x,
        fno_stats.mean().loc[x].to_numpy(),
        yerr=fno_stats.std().loc[x].to_numpy(),
        marker="o",
        capsize=4,
        color="#4C78A8",
    )
    for resolution in x:
        values = fno.loc[fno["resolution"] == resolution, "rollout_mean"].to_numpy()
        axes[0].scatter(np.full(len(values), resolution), values, color="black", s=16, alpha=0.5)
    axes[0].set_title("FNO: mean ± std and all seeds")
    axes[0].set_ylabel("100-step rollout mean relative L2")

    unet = groups["UNET"]
    unet_stats = unet.groupby("resolution")["rollout_mean"]
    axes[1].plot(x, unet_stats.median().loc[x].to_numpy(), marker="o", color="#F58518")
    for resolution in x:
        values = unet.loc[unet["resolution"] == resolution, "rollout_mean"].to_numpy()
        axes[1].scatter(np.full(len(values), resolution), values, color="#F58518", alpha=0.45, s=18)
    axes[1].set_yscale("log")
    axes[1].set_title("U-Net: median and all seeds")
    axes[1].set_ylabel("100-step rollout mean relative L2 (log)")

    for model, color in [("FNO", "#4C78A8"), ("UNET", "#F58518")]:
        group = groups[model]
        speed = group.groupby("resolution")["front_speed_mae"].median().loc[x].to_numpy()
        axes[2].plot(x, speed, marker="o", label=model, color=color)
    axes[2].set_yscale("log")
    axes[2].set_title("Median front speed error")
    axes[2].set_ylabel("Front speed MAE (log)")
    axes[2].legend()

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([256, 512, 1024], ["256", "512", "1024"])
        ax.set_xlabel("Evaluation grid (trained on 1024)")
        ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out / "resolution_comparison.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
