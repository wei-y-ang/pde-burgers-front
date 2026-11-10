"""Front position and speed metrics for periodic 1D Burgers profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def periodic_distance(a, b, length: float = 1.0):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(d, length - d)


def periodic_gradient(u: np.ndarray, dx: float) -> np.ndarray:
    """Central difference with periodic boundary, along the last axis."""

    return (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2.0 * dx)


def track_front(
    u: np.ndarray,
    x: np.ndarray,
    dt: float = 1.0,
    search_window: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Track the steepest front over time.

    Parameters
    ----------
    u:
        (T, X) profiles.
    x:
        (X,) periodic grid.
    search_window:
        Number of grid points searched around the previous front location.

    Returns
    -------
    positions, speeds
        positions in physical units, speeds from a linear fit.
    """

    if u.ndim != 2:
        raise ValueError(f"u must be (T,X), got {u.shape}")
    if u.shape[1] != len(x):
        raise ValueError("x length does not match u")
    if u.shape[0] < 2:
        raise ValueError("need at least two time slices")

    n_x = len(x)
    dx = float(x[1] - x[0])
    length = float(x[-1] - x[0] + dx)
    grad = np.abs(periodic_gradient(u, dx))

    idx = int(np.argmax(grad[0]))
    positions = [float(x[idx])]
    offsets = np.arange(-search_window, search_window + 1)

    for k in range(1, u.shape[0]):
        candidate_idx = (idx + offsets) % n_x
        values = grad[k, candidate_idx]
        local = int(np.argmax(values))
        idx = int(candidate_idx[local])

        # Unwrap the new position relative to the previous one.
        raw = float(x[idx])
        delta = (raw - positions[-1] + 0.5 * length) % length - 0.5 * length
        positions.append(positions[-1] + delta)

    positions_arr = np.asarray(positions)
    # Speeds from a global linear fit are less noisy than pointwise differences.
    t = np.arange(u.shape[0], dtype=float) * dt
    speed = float(np.polyfit(t, positions_arr, 1)[0]) if len(t) >= 2 else 0.0
    return positions_arr, np.asarray([speed])


@dataclass
class FrontMetrics:
    position_mae: float
    position_rmse: float
    position_bias: float
    position_mae_aligned: float
    position_rmse_aligned: float
    speed_true: float
    speed_pred: float
    speed_abs_error: float


def front_metrics(
    u_true: np.ndarray,
    u_pred: np.ndarray,
    x: np.ndarray,
    dt: float,
    search_window: int = 64,
) -> FrontMetrics:
    """Compare tracked front positions and global front speeds."""

    pos_true, speed_true = track_front(u_true, x, dt=dt, search_window=search_window)
    pos_pred, speed_pred = track_front(u_pred, x, dt=dt, search_window=search_window)

    length = float(x[-1] - x[0] + (x[1] - x[0]))
    d = periodic_distance(pos_true, pos_pred, length)

    # If tracking locks onto a different shock, the raw position error can be
    # ~0.5 even when the predicted speed is correct.  Remove a constant circular
    # offset to measure trajectory drift separately from the choice of shock.
    signed = (pos_pred - pos_true + 0.5 * length) % length - 0.5 * length
    bias = float(np.median(signed))
    residual = (signed - bias + 0.5 * length) % length - 0.5 * length
    d_aligned = np.abs(residual)

    # track_front returns a single global speed per trajectory.
    return FrontMetrics(
        position_mae=float(np.mean(d)),
        position_rmse=float(np.sqrt(np.mean(d * d))),
        position_bias=bias,
        position_mae_aligned=float(np.mean(d_aligned)),
        position_rmse_aligned=float(np.sqrt(np.mean(d_aligned * d_aligned))),
        speed_true=float(speed_true[0]),
        speed_pred=float(speed_pred[0]),
        speed_abs_error=abs(float(speed_true[0] - speed_pred[0])),
    )


def front_speed_from_positions(positions: np.ndarray, dt: float) -> float:
    """Fit a line to front positions and return the speed."""

    if len(positions) < 2:
        return 0.0
    t = np.arange(len(positions), dtype=float) * dt
    return float(np.polyfit(t, positions, 1)[0])
