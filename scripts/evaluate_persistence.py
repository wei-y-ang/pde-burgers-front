"""Evaluate the persistence baseline u(t+k)=u(t) on Burgers trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pde_burgers.data import load_burgers_npz, split_samples
from pde_burgers.front import front_metrics
from pde_burgers.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["all", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=16)
    p.add_argument("--out", type=str, default="results/persistence/id_test")
    return p.parse_args()


def segment(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    data = load_burgers_npz(args.data)
    if args.split == "test":
        _, _, indices = split_samples(data.n_samples, seed=args.seed)
    else:
        indices = np.arange(data.n_samples)
    indices = indices[: args.max_samples]

    steps = min(args.steps, data.n_times - 1)
    initial = data.u[indices, 0]
    pred = np.repeat(initial[:, None, :], steps, axis=1)
    true = data.u[indices, 1 : steps + 1]

    diff = np.linalg.norm(pred - true, axis=2)
    denom = np.linalg.norm(true, axis=2).clip(min=1e-8)
    rel = diff / denom
    per_step = rel.mean(axis=0)

    pair_diff = np.linalg.norm(
        data.u[indices, 1:] - data.u[indices, :-1], axis=2
    )
    pair_denom = np.linalg.norm(data.u[indices, :-1], axis=2).clip(min=1e-8)
    one_step_rel = float((pair_diff / pair_denom).mean())

    rows = []
    for i, sample_idx in enumerate(indices):
        fm = front_metrics(true[i], pred[i], data.x, dt=data.dt, search_window=64)
        rows.append(
            {
                "sample": int(sample_idx),
                "rel_l2_mean": float(rel[i].mean()),
                "rel_l2_final": float(rel[i, -1]),
                "front_position_mae": fm.position_mae,
                "front_position_rmse": fm.position_rmse,
                "front_position_bias": fm.position_bias,
                "front_position_mae_aligned": fm.position_mae_aligned,
                "front_position_rmse_aligned": fm.position_rmse_aligned,
                "front_speed_true": fm.speed_true,
                "front_speed_pred": fm.speed_pred,
                "front_speed_abs_error": fm.speed_abs_error,
            }
        )

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(args.out)
    df.to_csv(out_dir / "per_sample.csv", index=False)
    pd.DataFrame({"step": np.arange(1, steps + 1), "rel_l2": per_step}).to_csv(
        out_dir / "rollout_error.csv", index=False
    )

    summary = {
        "model": "persistence",
        "data": str(args.data),
        "nu": data.nu,
        "split": args.split,
        "seed": args.seed,
        "n_samples": int(len(indices)),
        "steps": int(steps),
        "one_step_rel_l2": one_step_rel,
        "rel_l2": {
            "mean": float(rel.mean()),
            "at_10": float(per_step[9]) if steps >= 10 else None,
            "at_50": float(per_step[49]) if steps >= 50 else None,
            "at_100": float(per_step[99]) if steps >= 100 else None,
        },
        "front": {
            "position_mae": segment(df["front_position_mae"]),
            "position_mae_aligned": segment(df["front_position_mae_aligned"]),
            "position_bias": segment(df["front_position_bias"]),
            "speed_abs_error": segment(df["front_speed_abs_error"]),
            "speed_true": segment(df["front_speed_true"]),
            "speed_pred": segment(df["front_speed_pred"]),
        },
    }
    save_json(summary, out_dir / "metrics.json")
    np.savez_compressed(
        out_dir / "rollout_example.npz",
        u_true=data.u[indices[:1], : steps + 1],
        u_pred=pred[:1],
        x=data.x,
        t=data.t[: steps + 1],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
