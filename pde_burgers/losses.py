"""Metrics and a finite-difference Burgers residual for later physics tuning."""

from __future__ import annotations

import torch


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean relative L2 error over the batch."""

    diff = torch.linalg.vector_norm(pred - target, dim=tuple(range(1, pred.ndim)))
    denom = torch.linalg.vector_norm(target, dim=tuple(range(1, target.ndim))).clamp_min(eps)
    return (diff / denom).mean()


def burgers_residual_step(
    u_t: torch.Tensor,
    u_next: torch.Tensor,
    dx: float,
    dt: float,
    nu: float,
) -> torch.Tensor:
    """One-step finite-difference residual of ``u_t + u u_x - nu u_xx``.

    Parameters
    ----------
    u_t, u_next:
        (B, X) profiles at consecutive times.
    dx, dt:
        Grid spacing.
    nu:
        Viscosity.

    Returns
    -------
    residual:
        (B, X-2) interior residual using a forward time difference.
    """

    if u_t.ndim != 2 or u_next.ndim != 2:
        raise ValueError(f"expected (B,X), got {tuple(u_t.shape)} and {tuple(u_next.shape)}")
    if u_t.shape != u_next.shape:
        raise ValueError("u_t and u_next must have the same shape")

    u_t_time = (u_next - u_t) / dt
    u_x = (u_t[:, 2:] - u_t[:, :-2]) / (2.0 * dx)
    u_xx = (u_t[:, 2:] - 2.0 * u_t[:, 1:-1] + u_t[:, :-2]) / (dx * dx)
    u_int = u_t[:, 1:-1]
    return u_t_time[:, 1:-1] + u_int * u_x - nu * u_xx


def burgers_residual_fd(
    u: torch.Tensor,
    dx: float,
    dt: float,
    nu: float,
) -> torch.Tensor:
    """Finite-difference residual of u_t + u u_x - nu u_xx.

    Parameters
    ----------
    u:
        (B, T, X) predicted or reference trajectories.
    dx, dt:
        Grid spacing.
    nu:
        Viscosity.

    Returns
    -------
    residual:
        (B, T-2, X-2) interior residual.
    """

    if u.ndim != 3:
        raise ValueError(f"u must be (B,T,X), got {tuple(u.shape)}")
    if u.shape[1] < 3 or u.shape[2] < 3:
        raise ValueError("need at least 3 time and space points")

    # Time derivative at interior times, all x.
    u_t = (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dt)
    # Spatial derivatives at all times, interior x.
    u_x = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dx)
    u_xx = (u[:, :, 2:] - 2.0 * u[:, :, 1:-1] + u[:, :, :-2]) / (dx * dx)

    # Align to interior time and interior x.
    u_t = u_t[:, :, 1:-1]
    u_x = u_x[:, 1:-1, :]
    u_xx = u_xx[:, 1:-1, :]
    u_int = u[:, 1:-1, 1:-1]

    return u_t + u_int * u_x - nu * u_xx
