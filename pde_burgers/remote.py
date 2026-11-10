"""Read PDEBench HDF5 files with HTTP range requests.

The full 1D Burgers files are ~7.7 GB.  For a course project we usually need
only a few hundred samples, so this module reads just the requested slice and
saves a small local ``.npz`` file.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import requests

# Importing the package adds the local .deps fallback to sys.path.
import pde_burgers  # noqa: F401
import h5py


class HTTPRangeFile(io.RawIOBase):
    """A minimal seekable read-only file backed by HTTP Range requests."""

    def __init__(self, url: str, timeout: float = 120.0):
        super().__init__()
        self.timeout = timeout
        self.session = requests.Session()
        r = self.session.get(
            url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout
        )
        r.raise_for_status()
        self.url = r.url
        content_range = r.headers.get("Content-Range")
        if not content_range or "/" not in content_range:
            raise RuntimeError(f"server did not return Content-Range: {r.headers}")
        self.size = int(content_range.split("/")[1])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self.pos + offset
        elif whence == io.SEEK_END:
            new = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek position")
        self.pos = new
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.pos
        if size == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + size, self.size) - 1
        r = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data

    def readinto(self, b: bytearray) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_burgers_subset(
    file_id: int = 281363,
    n_samples: int = 200,
    nu: float = 0.01,
    out_path: str | Path = "data/burgers_nu0.01_200.npz",
    base_url: str = "https://darus.uni-stuttgart.de/api/access/datafile",
    timeout: float = 300.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Read the first ``n_samples`` trajectories and save a local ``.npz``.

    Returns a dict with arrays and metadata.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    url = f"{base_url}/{file_id}"
    if verbose:
        print(f"opening remote HDF5: {url}")

    remote = HTTPRangeFile(url, timeout=timeout)
    if verbose:
        print(f"remote size: {remote.size / 1e9:.2f} GB")

    with h5py.File(remote, "r") as h:
        t = np.asarray(h["t-coordinate"][:], dtype=np.float32)
        x = np.asarray(h["x-coordinate"][:], dtype=np.float32)
        tensor = h["tensor"]
        if tensor.ndim != 3:
            raise ValueError(f"expected tensor shape (N,T,X), got {tensor.shape}")
        n = min(n_samples, tensor.shape[0])
        u = np.asarray(tensor[:n], dtype=np.float32)
        # Some PDEBench files store T+1 time edges for T snapshots.
        if len(t) == u.shape[1] + 1:
            t = t[: u.shape[1]]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        u=u,
        x=x,
        t=t,
        nu=np.asarray(nu, dtype=np.float32),
        source=np.asarray(f"PDEBench 1D Burgers file_id={file_id}"),
    )

    if verbose:
        print(f"saved {out_path}  u.shape={u.shape}  u.range=({u.min():.4f},{u.max():.4f})")
        print(f"size on disk: {out_path.stat().st_size / 1e6:.1f} MB")

    return {"u": u, "x": x, "t": t, "nu": nu, "path": str(out_path)}
