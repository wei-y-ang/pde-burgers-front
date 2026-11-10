"""Dataset utilities for PDEBench 1D Burgers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class BurgersData:
    """A small local subset of PDEBench 1D Burgers.

    Shapes
    ------
    u: (N, T, X)
        N trajectories, T time snapshots, X spatial grid points.
    x: (X,)
    t: (T,)
    nu_per_sample: (N,)
        Viscosity for each trajectory; allows multi-nu joint training.
    """

    u: np.ndarray
    x: np.ndarray
    t: np.ndarray
    nu: float
    source: str = ""
    nu_per_sample: np.ndarray | None = None

    @property
    def nu_values(self) -> np.ndarray:
        if self.nu_per_sample is None:
            return np.full(self.u.shape[0], self.nu, dtype=np.float32)
        return np.asarray(self.nu_per_sample, dtype=np.float32)

    @property
    def n_samples(self) -> int:
        return int(self.u.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.u.shape[1])

    @property
    def n_x(self) -> int:
        return int(self.u.shape[2])

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def load_burgers_npz(path: str | Path) -> BurgersData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        u = np.asarray(z["u"], dtype=np.float32)
        x = np.asarray(z["x"], dtype=np.float32)
        t = np.asarray(z["t"], dtype=np.float32)
        nu = float(np.asarray(z["nu"]).reshape(-1)[0])
        source = str(np.asarray(z["source"]).reshape(-1)[0]) if "source" in z else ""

    if u.ndim != 3:
        raise ValueError(f"u must have shape (N,T,X), got {u.shape}")
    # PDEBench's Burgers file stores T+1 time edges for T snapshots.
    if len(t) == u.shape[1] + 1:
        t = t[: u.shape[1]]
    if u.shape[2] != len(x) or u.shape[1] != len(t):
        raise ValueError(
            f"coordinate mismatch: u={u.shape}, len(x)={len(x)}, len(t)={len(t)}"
        )
    if not np.isfinite(u).all():
        raise ValueError("u contains NaN or Inf")

    return BurgersData(
        u=u,
        x=x,
        t=t,
        nu=nu,
        source=source,
        nu_per_sample=np.full(u.shape[0], nu, dtype=np.float32),
    )


def combine_burgers(data_list: list[BurgersData]) -> BurgersData:
    """Concatenate multiple single-nu subsets along the sample axis."""

    if not data_list:
        raise ValueError("data_list is empty")
    x0 = data_list[0].x
    t0 = data_list[0].t
    for d in data_list[1:]:
        if len(d.x) != len(x0) or not np.allclose(d.x, x0):
            raise ValueError("all datasets must share the same x grid")
        if len(d.t) != len(t0) or not np.allclose(d.t, t0):
            raise ValueError("all datasets must share the same t grid")

    u = np.concatenate([d.u for d in data_list], axis=0)
    nu = np.concatenate([d.nu_values for d in data_list]).astype(np.float32)
    return BurgersData(
        u=u,
        x=x0,
        t=t0,
        nu=float(np.mean(np.unique(nu))),
        source=" + ".join(sorted({d.source for d in data_list if d.source})),
        nu_per_sample=nu,
    )


def filter_by_nu(data: BurgersData, exclude_nu: list[float] | None = None) -> BurgersData:
    """Remove samples whose viscosity is in ``exclude_nu`` (within tolerance)."""

    if not exclude_nu:
        return data
    values = data.nu_values
    keep = np.ones(len(values), dtype=bool)
    for nu in exclude_nu:
        keep &= ~np.isclose(values, nu, rtol=1e-4, atol=1e-6)
    return BurgersData(
        u=data.u[keep],
        x=data.x,
        t=data.t,
        nu=data.nu,
        source=data.source,
        nu_per_sample=values[keep],
    )


def split_samples_by_nu(
    nu_per_sample: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split so every viscosity appears in train/val/test."""

    values = np.asarray(nu_per_sample, dtype=np.float32)
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    for nu in np.unique(values):
        idx = np.flatnonzero(np.isclose(values, nu, rtol=1e-4, atol=1e-6))
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(test_frac * n))) if n >= 3 else 0
        n_val = max(1, int(round(val_frac * n))) if n >= 3 else 0
        if n_test + n_val >= n:
            n_test, n_val = 0, 0
        test_parts.append(idx[:n_test])
        val_parts.append(idx[n_test : n_test + n_val])
        train_parts.append(idx[n_test + n_val :])

    train = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    val = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
    test = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
    if len(train) == 0:
        raise ValueError("no training samples after split")
    return train, val, test


def split_samples(
    n_samples: int,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random split by trajectory, not by space-time point."""

    if n_samples < 3:
        raise ValueError("need at least 3 samples")
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1:
        raise ValueError("val_frac and test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be < 1")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)

    n_test = max(1, int(round(test_frac * n_samples)))
    n_val = max(1, int(round(val_frac * n_samples)))
    if n_test + n_val >= n_samples:
        n_test = max(1, n_samples // 5)
        n_val = max(1, n_samples // 5)

    test = perm[:n_test]
    val = perm[n_test : n_test + n_val]
    train = perm[n_test + n_val :]
    return train, val, test


class BurgersPairs(Dataset):
    """One-step pairs ``(u(t), u(t+1))`` for teacher-forced training."""

    def __init__(
        self,
        u: np.ndarray,
        indices: np.ndarray,
        mean: float,
        std: float,
        nu_per_sample: np.ndarray | None = None,
        conditional: bool = False,
        log_nu_mean: float = 0.0,
        log_nu_std: float = 1.0,
    ):
        self.u = ((u - mean) / std).astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.n_times = int(u.shape[1])
        self.conditional = conditional
        if conditional:
            if nu_per_sample is None:
                raise ValueError("conditional=True requires nu_per_sample")
            log_nu = np.log(np.asarray(nu_per_sample, dtype=np.float32))
            self.nu_channel = ((log_nu - log_nu_mean) / log_nu_std).astype(np.float32)
        else:
            self.nu_channel = None

    def __len__(self) -> int:
        return int(len(self.indices) * (self.n_times - 1))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        per_sample = self.n_times - 1
        sample_pos = idx // per_sample
        t_idx = idx % per_sample
        sample = self.indices[sample_pos]
        x_t = self.u[sample, t_idx][None, :]
        if self.conditional:
            cond = np.full((1, x_t.shape[1]), self.nu_channel[sample], dtype=np.float32)
            x_t = np.concatenate([x_t, cond], axis=0)
        x_next = self.u[sample, t_idx + 1][None, :]
        return torch.from_numpy(x_t), torch.from_numpy(x_next)


class BurgersSequences(Dataset):
    """Multi-step windows ``(u(t0), [u(t0+1), ..., u(t0+H)])``."""

    def __init__(
        self,
        u: np.ndarray,
        indices: np.ndarray,
        mean: float,
        std: float,
        horizon: int = 5,
        nu_per_sample: np.ndarray | None = None,
        conditional: bool = False,
        log_nu_mean: float = 0.0,
        log_nu_std: float = 1.0,
    ):
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        self.u = ((u - mean) / std).astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.horizon = int(horizon)
        self.n_times = int(u.shape[1])
        self.n_windows = self.n_times - self.horizon
        if self.n_windows <= 0:
            raise ValueError("horizon is too large for the time axis")
        self.conditional = conditional
        if conditional:
            if nu_per_sample is None:
                raise ValueError("conditional=True requires nu_per_sample")
            log_nu = np.log(np.asarray(nu_per_sample, dtype=np.float32))
            self.nu_channel = ((log_nu - log_nu_mean) / log_nu_std).astype(np.float32)
        else:
            self.nu_channel = None

    def __len__(self) -> int:
        return int(len(self.indices) * self.n_windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_pos = idx // self.n_windows
        t0 = idx % self.n_windows
        sample = self.indices[sample_pos]
        x_t = self.u[sample, t0][None, :]
        if self.conditional:
            cond = np.full((1, x_t.shape[1]), self.nu_channel[sample], dtype=np.float32)
            x_t = np.concatenate([x_t, cond], axis=0)
        target = self.u[sample, t0 + 1 : t0 + self.horizon + 1]
        return torch.from_numpy(x_t), torch.from_numpy(target)


def compute_stats(u: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    train = u[indices]
    mean = float(train.mean())
    std = float(train.std())
    if std < 1e-12:
        std = 1.0
    return mean, std


def denormalize(u: np.ndarray, mean: float, std: float) -> np.ndarray:
    return u * std + mean
