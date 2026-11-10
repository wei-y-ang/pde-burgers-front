"""Evaluate a trained FNO/U-Net checkpoint on in-distribution or OOD Burgers data.

Examples
--------
In-distribution test split:
    python scripts/evaluate_ood.py \
        --run checkpoints/fno40_seed0 \
        --data data/burgers_nu0.01_200.npz \
        --split test \
        --out results/ood/id_test

OOD Nu=0.001:
    python scripts/evaluate_ood.py \
        --run checkpoints/fno40_seed0 \
        --data data/burgers_nu0.001_100.npz \
        --split all \
        --out results/ood/nu0.001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pde_burgers.data import load_burgers_npz, split_samples
from pde_burgers.front import front_metrics
from pde_burgers.models import FNO1d, UNet1d
from pde_burgers.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="checkpoints/fno40_seed0")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--split", type=str, default="all", choices=["all", "test"])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--out", type=str, default="results/ood/eval")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def build_model(cfg: dict) -> torch.nn.Module:
    in_channels = 2 if cfg.get("conditional", False) else 1
    if cfg["model"] == "fno":
        return FNO1d(
            in_channels=in_channels,
            out_channels=1,
            width=cfg["width"],
            modes=cfg["modes"],
            n_layers=cfg["n_layers"],
        )
    if cfg["model"] == "unet":
        return UNet1d(in_channels=in_channels, out_channels=1, base=cfg["base"])
    raise ValueError(f"unknown model {cfg['model']!r}")


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    u: np.ndarray,
    indices: np.ndarray,
    mean: float,
    std: float,
    steps: int,
    device: torch.device,
    nu_values: np.ndarray | None = None,
    conditional: bool = False,
    log_nu_mean: float = 0.0,
    log_nu_std: float = 1.0,
) -> np.ndarray:
    model.eval()
    current = torch.as_tensor((u[indices, 0] - mean) / std, dtype=torch.float32, device=device)
    cond = None
    if conditional:
        if nu_values is None:
            raise ValueError("conditional rollout requires nu_values")
        log_nu = np.log(np.asarray(nu_values, dtype=np.float32)[indices])
        cond_np = (log_nu - log_nu_mean) / log_nu_std
        cond = torch.as_tensor(cond_np, dtype=torch.float32, device=device)

    def model_input(profile: torch.Tensor) -> torch.Tensor:
        if conditional:
            c = cond[:, None, None].expand(-1, 1, profile.shape[1])
            return torch.cat([profile[:, None, :], c], dim=1)
        return profile[:, None, :]

    preds = []
    for _ in range(steps):
        pred = model(model_input(current))[:, 0, :]
        preds.append(pred)
        current = pred
    return torch.stack(preds, dim=1).cpu().numpy() * std + mean


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = metrics["config"]
    norm = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))
    mean, std = float(norm["mean"]), float(norm["std"])

    data = load_burgers_npz(args.data)
    if args.split == "test":
        _, _, test_idx = split_samples(
            data.n_samples,
            val_frac=cfg["val_frac"],
            test_frac=cfg["test_frac"],
            seed=cfg["seed"],
        )
        indices = test_idx
    else:
        indices = np.arange(data.n_samples)

    indices = indices[: args.max_samples]
    steps = min(args.steps, data.n_times - 1)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))

    pred = rollout(
        model,
        data.u,
        indices,
        mean,
        std,
        steps,
        device,
        nu_values=data.nu_values,
        conditional=bool(cfg.get("conditional", False)),
        log_nu_mean=float(norm.get("log_nu_mean", 0.0)),
        log_nu_std=float(norm.get("log_nu_std", 1.0)),
    )
    true = data.u[indices, 1 : steps + 1]

    diff = np.linalg.norm(pred - true, axis=2)
    denom = np.linalg.norm(true, axis=2).clip(min=1e-8)
    rel = diff / denom  # (N, steps)
    per_step = rel.mean(axis=0)

    rows = []
    for i, sample_idx in enumerate(indices):
        fm = front_metrics(
            true[i],
            pred[i],
            data.x,
            dt=data.dt,
            search_window=64,
        )
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

    def segment(values: np.ndarray) -> dict:
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    summary = {
        "run": str(run_dir),
        "data": str(args.data),
        "nu": data.nu,
        "split": args.split,
        "n_samples": int(len(indices)),
        "steps": int(steps),
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

    # Save one rollout example for plotting.  Keep the same format as train.py:
    # u_true is (1, T, X), u_pred is (T, X), including the initial frame.
    example_pred = np.concatenate(
        [data.u[indices[:1], 0][:, None, :], pred[:1]], axis=1
    )[0]
    np.savez_compressed(
        out_dir / "rollout_example.npz",
        u_true=data.u[indices[:1], : steps + 1],
        u_pred=example_pred,
        x=data.x,
        t=data.t[: steps + 1],
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
