"""Train FNO1D or U-Net1D on a local PDEBench Burgers subset.

Example
-------
    python -m pde_burgers.train \
        --data data/burgers_nu0.01_200.npz \
        --model fno \
        --out results/runs/fno \
        --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import (
    BurgersPairs,
    BurgersSequences,
    combine_burgers,
    compute_stats,
    filter_by_nu,
    load_burgers_npz,
    split_samples,
    split_samples_by_nu,
)
from .losses import burgers_residual_fd, burgers_residual_step, relative_l2
from .models import FNO1d, UNet1d
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, nargs="+", default=["data/burgers_nu0.01_200.npz"])
    p.add_argument("--out", type=str, default="results/runs/fno")
    p.add_argument("--model", type=str, default="fno", choices=["fno", "unet"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--pde-weight", type=float, default=0.0,
                   help="Weight on the Burgers PDE residual; 0 means data-only.")
    p.add_argument("--seq-len", type=int, default=1,
                   help="Autoregressive training horizon; 1 = one-step pairs.")
    p.add_argument("--seq-data-loss", type=str, default="all", choices=["all", "first"],
                   help="For seq-len > 1: supervise all steps or only the first step.")
    p.add_argument("--detach-seq-pde", action="store_true",
                   help="Detach predicted steps after the first when computing sequence PDE loss.")
    p.add_argument("--conditional", action="store_true",
                   help="Append a normalized log(nu) channel to the model input.")
    p.add_argument("--exclude-nu", type=float, nargs="*", default=[],
                   help="Exclude samples with these viscosity values (for leave-one-nu-out).")
    p.add_argument("--init", type=str, default="",
                   help="Optional checkpoint for fine-tuning.")
    p.add_argument("--width", type=int, default=32, help="FNO width")
    p.add_argument("--modes", type=int, default=16, help="FNO Fourier modes")
    p.add_argument("--n-layers", type=int, default=4, help="FNO blocks")
    p.add_argument("--base", type=int, default=16, help="U-Net base width")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--max-train-pairs", type=int, default=8000)
    p.add_argument("--max-val-pairs", type=int, default=2000)
    p.add_argument("--rollout-steps", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(args: argparse.Namespace) -> nn.Module:
    in_channels = 2 if args.conditional else 1
    if args.model == "fno":
        return FNO1d(
            in_channels=in_channels,
            out_channels=1,
            width=args.width,
            modes=args.modes,
            n_layers=args.n_layers,
        )
    return UNet1d(in_channels=in_channels, out_channels=1, base=args.base)


def subset_pairs(dataset: BurgersPairs, max_pairs: int, seed: int) -> Subset:
    n = len(dataset)
    if max_pairs <= 0 or max_pairs >= n:
        return Subset(dataset, np.arange(n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_pairs, replace=False)
    return Subset(dataset, np.sort(idx))


def predict_horizon(model: nn.Module, xb: torch.Tensor, horizon: int) -> torch.Tensor:
    """Autoregressively predict ``horizon`` future profiles.

    Parameters
    ----------
    xb:
        (B, 1, X) initial profile.
    horizon:
        Number of future steps.

    Returns
    -------
    (B, horizon, X)
    """

    current = xb[:, 0, :]
    cond = xb[:, 1:, :] if xb.shape[1] > 1 else None
    preds = []
    for _ in range(horizon):
        if cond is None:
            model_in = current[:, None, :]
        else:
            model_in = torch.cat([current[:, None, :], cond], dim=1)
        pred = model(model_in)[:, 0, :]
        preds.append(pred)
        current = pred
    return torch.stack(preds, dim=1)


@torch.no_grad()
def rollout(
    model: nn.Module,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive rollout for a few test trajectories.

    Returns
    -------
    errors:
        (steps,) mean relative L2 error at each rollout step.
    example:
        (steps+1, X) true/pred stack for the first selected trajectory.
    """

    model.eval()
    steps = min(steps, u.shape[1] - 1)
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

    pred_norm = torch.stack(preds, dim=1)  # (B, steps, X)
    true_norm = torch.as_tensor(
        (u[indices, 1 : steps + 1] - mean) / std, dtype=torch.float32, device=device
    )
    # Per-step mean error.
    diff = torch.linalg.vector_norm(pred_norm - true_norm, dim=2)
    denom = torch.linalg.vector_norm(true_norm, dim=2).clamp_min(1e-8)
    per_step = (diff / denom).mean(dim=0).cpu().numpy()

    example_pred = np.concatenate(
        [
            u[indices[:1], 0],
            pred_norm[0].cpu().numpy() * std + mean,
        ],
        axis=0,
    )
    return per_step, example_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out)

    data_list = [load_burgers_npz(path) for path in args.data]
    data = combine_burgers(data_list) if len(data_list) > 1 else data_list[0]
    data = filter_by_nu(data, args.exclude_nu)

    if args.conditional or len(np.unique(data.nu_values)) > 1:
        train_idx, val_idx, test_idx = split_samples_by_nu(
            data.nu_values, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
    else:
        train_idx, val_idx, test_idx = split_samples(
            data.n_samples, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
    mean, std = compute_stats(data.u, train_idx)

    log_nu = np.log(data.nu_values)
    log_nu_mean = float(log_nu[train_idx].mean())
    log_nu_std = float(log_nu[train_idx].std())
    if log_nu_std < 1e-12:
        log_nu_std = 1.0

    ds_kwargs = {
        "nu_per_sample": data.nu_values,
        "conditional": args.conditional,
        "log_nu_mean": log_nu_mean,
        "log_nu_std": log_nu_std,
    }
    if args.seq_len == 1:
        train_ds = BurgersPairs(data.u, train_idx, mean, std, **ds_kwargs)
        val_ds = BurgersPairs(data.u, val_idx, mean, std, **ds_kwargs)
    else:
        train_ds = BurgersSequences(
            data.u, train_idx, mean, std, horizon=args.seq_len, **ds_kwargs
        )
        val_ds = BurgersSequences(
            data.u, val_idx, mean, std, horizon=args.seq_len, **ds_kwargs
        )
    train_loader = DataLoader(
        subset_pairs(train_ds, args.max_train_pairs, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        subset_pairs(val_ds, args.max_val_pairs, args.seed + 1),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = make_model(args).to(device)
    if args.init:
        state = torch.load(args.init, map_location=device)
        model.load_state_dict(state)
        print(f"loaded init checkpoint: {args.init}")
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"data={args.data} shape={data.u.shape} nu={data.nu} "
        f"train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
    )
    print(
        f"model={args.model} params={n_params:,} device={device} "
        f"pde_weight={args.pde_weight}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_left = args.patience
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        pde_sum = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if args.seq_len == 1:
                pred = model(xb)
                data_loss = loss_fn(pred, yb)
                pde_scale = data.dt / std
                if args.pde_weight > 0:
                    # PDE residual is computed in physical units.
                    u_t_phys = xb[:, 0, :] * std + mean
                    u_next_phys = pred[:, 0, :] * std + mean
                    residual = burgers_residual_step(
                        u_t_phys,
                        u_next_phys,
                        dx=data.dx,
                        dt=data.dt,
                        nu=data.nu,
                    )
                    pde_loss = (residual * pde_scale).square().mean()
                    loss = data_loss + args.pde_weight * pde_loss
                    pde_sum += float(pde_loss.item()) * len(xb)
                else:
                    loss = data_loss
            else:
                # Sequence-level training: unroll H steps.
                pred_seq = predict_horizon(model, xb, args.seq_len)
                if args.seq_data_loss == "first":
                    data_loss = loss_fn(pred_seq[:, 0, :], yb[:, 0, :])
                else:
                    data_loss = loss_fn(pred_seq, yb)
                pde_scale = data.dt / std
                if args.pde_weight > 0:
                    u0 = xb[:, 0, :]
                    pred_for_pde = pred_seq
                    if args.detach_seq_pde and pred_seq.shape[1] > 1:
                        # Keep the first predicted step differentiable; detach
                        # later steps so the PDE gradient cannot amplify errors.
                        pred_for_pde = torch.cat(
                            [pred_seq[:, :1], pred_seq[:, 1:].detach()], dim=1
                        )
                    phys_seq = torch.cat([u0[:, None, :], pred_for_pde], dim=1) * std + mean
                    residual = burgers_residual_fd(
                        phys_seq,
                        dx=data.dx,
                        dt=data.dt,
                        nu=data.nu,
                    )
                    pde_loss = (residual * pde_scale).square().mean()
                    loss = data_loss + args.pde_weight * pde_loss
                    pde_sum += float(pde_loss.item()) * len(xb)
                else:
                    loss = data_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_sum += float(loss.item()) * len(xb)
            n_train += len(xb)

        model.eval()
        val_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if args.seq_len == 1:
                    pred = model(xb)
                    val_batch = float(loss_fn(pred, yb).item())
                else:
                    pred = predict_horizon(model, xb, args.seq_len)
                    if args.seq_data_loss == "first":
                        val_batch = float(loss_fn(pred[:, 0, :], yb[:, 0, :]).item())
                    else:
                        val_batch = float(loss_fn(pred, yb).item())
                val_sum += val_batch * len(xb)
                n_val += len(xb)

        train_loss = train_sum / max(n_train, 1)
        val_loss = val_sum / max(n_val, 1)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "pde_mse": pde_sum / max(n_train, 1) if args.pde_weight > 0 else 0.0,
            }
        )
        if epoch == 1 or epoch % 5 == 0:
            extra = (
                f" pde_mse={pde_sum / max(n_train, 1):.6e}"
                if args.pde_weight > 0
                else ""
            )
            print(
                f"epoch {epoch:3d} train_mse={train_loss:.6e} "
                f"val_mse={val_loss:.6e}{extra}"
            )

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # One-step test metrics.
    model.eval()
    test_ds = BurgersPairs(data.u, test_idx, mean, std, **ds_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_mse = 0.0
    test_rel = 0.0
    n_test = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            test_mse += float(loss_fn(pred, yb).item()) * len(xb)
            test_rel += float(relative_l2(pred, yb).item()) * len(xb)
            n_test += len(xb)

    # Rollout on a small subset.
    n_roll = min(16, len(test_idx))
    roll_idx = test_idx[:n_roll]
    per_step, example_pred = rollout(
        model,
        data.u,
        roll_idx,
        mean,
        std,
        steps=args.rollout_steps,
        device=device,
        nu_values=data.nu_values,
        conditional=args.conditional,
        log_nu_mean=log_nu_mean,
        log_nu_std=log_nu_std,
    )

    metrics = {
        "data": args.data,
        "nu": data.nu,
        "nus": [float(x) for x in np.unique(data.nu_values)],
        "conditional": bool(args.conditional),
        "shape": list(data.u.shape),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_params": int(n_params),
        "best_val_mse": float(best_val),
        "test_mse": test_mse / max(n_test, 1),
        "test_rel_l2": test_rel / max(n_test, 1),
        "rollout_rel_l2": {
            "mean": float(np.mean(per_step)),
            "at_10": float(per_step[9]) if len(per_step) >= 10 else None,
            "at_50": float(per_step[49]) if len(per_step) >= 50 else None,
            "at_100": float(per_step[99]) if len(per_step) >= 100 else None,
        },
        "config": vars(args),
    }

    torch.save(model.state_dict(), out_dir / "model.pt")
    save_json(metrics, out_dir / "metrics.json")
    save_json(
        {
            "mean": mean,
            "std": std,
            "log_nu_mean": log_nu_mean,
            "log_nu_std": log_nu_std,
            "conditional": bool(args.conditional),
        },
        out_dir / "normalization.json",
    )
    np.savez_compressed(
        out_dir / "rollout_example.npz",
        u_pred=example_pred,
        u_true=data.u[roll_idx[:1], : example_pred.shape[0]],
        x=data.x,
        t=data.t[: example_pred.shape[0]],
    )
    np.savetxt(out_dir / "rollout_error.csv", per_step, delimiter=",", header="rel_l2", comments="")

    print("test rel L2:", metrics["test_rel_l2"])
    print("rollout mean rel L2:", metrics["rollout_rel_l2"]["mean"])
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
