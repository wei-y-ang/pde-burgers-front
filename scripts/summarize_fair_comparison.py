"""Summarize fair multi-seed comparisons, paired deltas, and rollout curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = {
    "FNO baseline": [f"results/runs/fno40_seed{s}" for s in range(3)],
    "U-Net baseline": [f"results/runs/unet40_seed{s}" for s in range(3)],
    "FNO + data-only FT": [
        "results/runs/finetune_data",
        "results/runs/finetune_data_seed1",
        "results/runs/finetune_data_seed2",
    ],
    "FNO + PDE FT": [
        "results/runs/finetune_pde1e-3",
        "results/runs/finetune_pde1e-3_seed1",
        "results/runs/finetune_pde1e-3_seed2",
    ],
    "FNO + seq data-only": [
        "results/runs/seq_first_data",
        "results/runs/seq_first_data_seed1",
        "results/runs/seq_first_data_seed2",
    ],
}

PAIRS = {
    "PDE FT - data-only FT": ("FNO + PDE FT", "FNO + data-only FT"),
    "seq data-only - FNO baseline": ("FNO + seq data-only", "FNO baseline"),
    "FNO baseline - U-Net baseline": ("FNO baseline", "U-Net baseline"),
}


def load_metrics(run: str) -> dict:
    path = Path(run) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; finish all required seed runs first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="results/fair_comparison")
    p.add_argument("--persistence", type=str, default="results/persistence/id_test/metrics.json")
    args = p.parse_args()

    rows = []
    curves = {}
    for variant, runs in VARIANTS.items():
        variant_curves = []
        for run in runs:
            m = load_metrics(run)
            rows.append(
                {
                    "variant": variant,
                    "seed": int(m["config"]["seed"]),
                    "run": run,
                    "one_step_rel_l2": float(m["test_rel_l2"]),
                    "rollout_mean": float(m["rollout_rel_l2"]["mean"]),
                    "rollout_10": float(m["rollout_rel_l2"]["at_10"]),
                    "rollout_50": float(m["rollout_rel_l2"]["at_50"]),
                    "rollout_100": float(m["rollout_rel_l2"]["at_100"]),
                }
            )
            curve_path = Path(run) / "rollout_error.csv"
            variant_curves.append(pd.read_csv(curve_path)["rel_l2"].to_numpy(dtype=float))
        curves[variant] = np.stack(variant_curves, axis=0)

    per_seed = pd.DataFrame(rows).sort_values(["variant", "seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "per_seed.csv", index=False)

    metrics = ["one_step_rel_l2", "rollout_mean", "rollout_10", "rollout_50", "rollout_100"]
    summary_rows = []
    for variant in VARIANTS:
        group = per_seed[per_seed["variant"] == variant]
        row = {"variant": variant, "n_seeds": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        summary_rows.append(row)

    persistence_path = Path(args.persistence)
    persistence = None
    if persistence_path.exists():
        persistence = json.loads(persistence_path.read_text(encoding="utf-8"))
        row = {"variant": "Persistence", "n_seeds": 1}
        values = {
            "one_step_rel_l2": persistence["one_step_rel_l2"],
            "rollout_mean": persistence["rel_l2"]["mean"],
            "rollout_10": persistence["rel_l2"]["at_10"],
            "rollout_50": persistence["rel_l2"]["at_50"],
            "rollout_100": persistence["rel_l2"]["at_100"],
        }
        for metric in metrics:
            row[f"{metric}_mean"] = values[metric]
            row[f"{metric}_std"] = 0.0
            row[f"{metric}_median"] = values[metric]
            row[f"{metric}_min"] = values[metric]
            row[f"{metric}_max"] = values[metric]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)

    paired_rows = []
    for comparison, (variant, baseline) in PAIRS.items():
        left = per_seed[per_seed["variant"] == variant].set_index("seed")
        right = per_seed[per_seed["variant"] == baseline].set_index("seed")
        for seed in sorted(set(left.index) & set(right.index)):
            paired_rows.append(
                {
                    "comparison": comparison,
                    "seed": seed,
                    "variant_rollout": left.loc[seed, "rollout_mean"],
                    "baseline_rollout": right.loc[seed, "rollout_mean"],
                    "delta_rollout": left.loc[seed, "rollout_mean"] - right.loc[seed, "rollout_mean"],
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_deltas.csv", index=False)
    paired_summary = (
        paired.groupby("comparison")["delta_rollout"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    paired_summary.to_csv(out / "paired_summary.csv", index=False)

    labels = list(VARIANTS)
    medians = [summary.loc[summary["variant"] == v, "rollout_mean_median"].iloc[0] for v in labels]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, medians, color="#4C78A8", alpha=0.75, label="Median across seeds")
    for i, variant in enumerate(labels):
        values = per_seed.loc[per_seed["variant"] == variant, "rollout_mean"].to_numpy()
        ax.scatter(np.full(len(values), i), values, color="black", zorder=3, s=22, label="Individual seeds" if i == 0 else None)
    if persistence is not None:
        ax.axhline(
            persistence["rel_l2"]["mean"],
            color="#E45756",
            linestyle="--",
            label="Persistence",
        )
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("100-step rollout mean relative L2 (log scale)")
    ax.set_title("Fair comparison: medians and all individual seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fair_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for variant, arr in curves.items():
        median = np.median(arr, axis=0)
        lower = arr.min(axis=0)
        upper = arr.max(axis=0)
        steps = np.arange(1, arr.shape[1] + 1)
        ax.plot(steps, median, label=variant)
        ax.fill_between(steps, lower.clip(min=1e-12), upper.clip(min=1e-12), alpha=0.10)
    if persistence is not None:
        curve_path = persistence_path.parent / "rollout_error.csv"
        if curve_path.exists():
            curve = pd.read_csv(curve_path)["rel_l2"].to_numpy(dtype=float)
            ax.plot(np.arange(1, len(curve) + 1), curve, "--", label="Persistence", color="black")
    ax.set_yscale("log")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2 (log scale)")
    ax.set_title("Rollout error: median and seed range")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rollout_error_comparison.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))
    print("\nPaired rollout deltas (negative means improvement):")
    print(paired_summary.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
