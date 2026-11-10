"""FNO1D and U-Net1D baselines for Burgers front prediction."""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    """1D Fourier layer: keep the lowest ``modes`` Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        if modes < 1:
            raise ValueError("modes must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        real = torch.randn(in_channels, out_channels, modes) * scale
        imag = torch.randn(in_channels, out_channels, modes) * scale
        self.weight = nn.Parameter(torch.complex(real, imag))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, n)
        batch, _, n = x.shape
        if self.modes > n // 2 + 1:
            raise ValueError(f"modes={self.modes} too large for n={n}")
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            n // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes], self.weight
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


class FNO1d(nn.Module):
    """A compact Fourier Neural Operator for 1D profiles."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


class UNet1d(nn.Module):
    """A small 1D U-Net baseline."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base, base, 3, padding=1),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(base, base * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base * 2, base * 2, 3, padding=1),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(base * 2, base * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base * 4, base * 4, 3, padding=1),
            nn.SiLU(),
        )
        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv1d(base * 4, base * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base * 2, base * 2, 3, padding=1),
            nn.SiLU(),
        )
        self.up1 = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv1d(base * 2, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base, base, 3, padding=1),
            nn.SiLU(),
        )
        self.out = nn.Conv1d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.up2(e3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = d2[..., : e2.shape[-1]]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = d1[..., : e1.shape[-1]]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "fno":
        return FNO1d(**kwargs)
    if name == "unet":
        return UNet1d(**kwargs)
    raise ValueError(f"unknown model {name!r}")
