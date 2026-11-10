"""Aggregate multiple seed runs and plot the rollout error curve."""

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
    p.add_argument("--run-glob", type=str, default="results/runs/fno40_seed*")
    p.add_argument("--out-dir", type=str, default="results/summary_fno40")
    args = p.parse_args()

    run_dirs = sorted(Path(".").glob(args.run_glob))
    if not run_dirs:
        raise SystemExit(f"no runs matched {args.run_glob!r}")

    rows = []
    curves = []
    for run in run_dirs:
        metrics_path = run / "metrics.json"
        curve_path = run / "rollout_error.csv"
        if not metrics_path.exists():
            print(f"skip {run}: no metrics.json")
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": run.name,
                "seed": m["config"]["seed"],
                "test_rel_l2": m["test_rel_l2"],
                "rollout_mean": m["rollout_rel_l2"]["mean"],
                "rollout_10": m["rollout_rel_l2"].get("at_10"),
                "rollout_50": m["rollout_rel_l2"].get("at_50"),
                "rollout_100": m["rollout_rel_l2"].get("at_100"),
            }
        )
        if curve_path.exists():
            curve = pd.read_csv(curve_path)["rel_l2"].to_numpy()
            curves.append(curve)

    df = pd.DataFrame(rows).sort_values("seed")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_seed.csv", index=False)

    numeric = ["test_rel_l2", "rollout_mean", "rollout_10", "rollout_50", "rollout_100"]
    summary = {}
    for col in numeric:
        values = df[col].dropna().to_numpy(dtype=float)
        if len(values):
            summary[col] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print("\nmean ± std")
    for col, s in summary.items():
        print(f"{col:14s} {s['mean']:.6f} ± {s['std']:.6f}  (n={s['n']})")

    if curves:
        n = min(len(c) for c in curves)
        arr = np.stack([c[:n] for c in curves], axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        steps = np.arange(1, n + 1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, mean, label="mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, label="±1 std")
        ax.set_xlabel("rollout step")
        ax.set_ylabel("relative L2")
        ax.set_title("FNO rollout error across seeds")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "rollout_mean_std.png", dpi=160)
        plt.close(fig)
        print("wrote", out / "rollout_mean_std.png")


if __name__ == "__main__":
    main()
