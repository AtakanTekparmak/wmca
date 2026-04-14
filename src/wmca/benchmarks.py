"""Benchmark data generators for unified ablation.

Each generator returns a BenchmarkData namedtuple with:
    X_train, Y_train, X_val, Y_val, X_test, Y_test, meta

All tensors are torch float32 on the requested device.
Splits are 70/15/15 by trajectory (or by sample for grid_world).
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

BenchmarkData = namedtuple("BenchmarkData",
    ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test", "meta"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_trajectories(trajs: np.ndarray, fracs=(0.70, 0.15, 0.15)):
    """Split along first axis."""
    n = len(trajs)
    n1 = int(fracs[0] * n)
    n2 = int(fracs[1] * n)
    return trajs[:n1], trajs[n1:n1 + n2], trajs[n1 + n2:]


def _make_pairs(trajs: np.ndarray):
    """(N, T+1, ...) -> X (N*T, ...), Y (N*T, ...)."""
    N, Tp1 = trajs.shape[:2]
    rest = trajs.shape[2:]
    X = trajs[:, :-1].reshape(-1, *rest)
    Y = trajs[:, 1:].reshape(-1, *rest)
    return X, Y


def _to_torch(arrays, device):
    """Convert a list of numpy arrays to torch tensors on device."""
    return tuple(torch.from_numpy(a).float().to(device) for a in arrays)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEAT EQUATION
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-compute laplacian kernel
_LAP_KERNEL_NP = np.array([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]], dtype=np.float32)

# Heat PDE parameters
_HEAT_ALPHA = 0.1
_HEAT_DT = 0.01
_HEAT_DX_32 = 1.0 / 32  # default for grid_size=32


def _heat_step(u: np.ndarray, coeff: float) -> np.ndarray:
    """Single heat equation step with zero Dirichlet BCs."""
    from scipy.signal import convolve2d
    lap = convolve2d(u, _LAP_KERNEL_NP, mode="same", boundary="fill", fillvalue=0.0)
    u_new = u + coeff * lap
    return np.clip(u_new, 0.0, 1.0)


def _heat_random_ic(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    """Random sum of 2-5 Gaussian blobs, normalized to [0, 1]."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n_blobs = rng.randint(2, 6)
    u0 = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_blobs):
        cy = rng.uniform(0.2, 0.8) * h
        cx = rng.uniform(0.2, 0.8) * w
        sigma = rng.uniform(1.5, 4.0)
        u0 += np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2)) * rng.uniform(0.3, 1.0)
    u_max = u0.max()
    if u_max > 1e-8:
        u0 = u0 / u_max
    return u0


def generate_heat(grid_size: int = 32, n_trajectories: int = 500,
                  n_steps: int = 50, seed: int = 42,
                  device: str | torch.device = "cpu"):
    """Heat equation benchmark.

    Returns (X_train, Y_train, X_val, Y_val, X_test, Y_test, meta).
    X/Y shapes: (N, 1, H, W) float32 in [0, 1].
    meta: dict with 'name', 'loss_type' ('mse'), 'metric' ('mse'),
          'in_channels', 'out_channels'.
    """
    device = torch.device(device)
    dx = 1.0 / grid_size
    coeff = _HEAT_ALPHA * _HEAT_DT / (dx * dx)

    rng = np.random.RandomState(seed)
    trajs = np.zeros((n_trajectories, n_steps + 1, grid_size, grid_size),
                     dtype=np.float32)
    for i in range(n_trajectories):
        u0 = _heat_random_ic(grid_size, grid_size, rng)
        trajs[i, 0] = u0
        u = u0
        for t in range(n_steps):
            u = _heat_step(u, coeff)
            trajs[i, t + 1] = u

    train_t, val_t, test_t = _split_trajectories(trajs)
    X_tr, Y_tr = _make_pairs(train_t)
    X_v, Y_v = _make_pairs(val_t)
    X_te, Y_te = _make_pairs(test_t)

    # Add channel dim: (N, H, W) -> (N, 1, H, W)
    X_tr = X_tr[:, None]; Y_tr = Y_tr[:, None]
    X_v  = X_v[:, None];  Y_v  = Y_v[:, None]
    X_te = X_te[:, None];  Y_te = Y_te[:, None]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "heat",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": 1,
        "out_channels": 1,
        "grid_size": grid_size,
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "alpha_heat": _HEAT_ALPHA,
        "dt": _HEAT_DT,
        "dx": dx,
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  GAME OF LIFE
# ═══════════════════════════════════════════════════════════════════════════════

_GOL_KERNEL = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=np.float32)


def _gol_step(grid: np.ndarray) -> np.ndarray:
    """Single GoL step. grid: (H, W) binary {0, 1}."""
    from scipy.signal import convolve2d
    neighbors = convolve2d(grid.astype(np.float32), _GOL_KERNEL,
                           mode="same", boundary="wrap")
    born = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return (born | survive).astype(np.float32)


def generate_gol(grid_size: int = 32, n_trajectories: int = 1000,
                 n_steps: int = 20, density: float = 0.3,
                 seed: int = 42, device: str | torch.device = "cpu"):
    """Game of Life benchmark.

    Returns (X_train, Y_train, X_val, Y_val, X_test, Y_test, meta).
    X/Y shapes: (N, 1, H, W) float32 binary.
    loss_type='bce', metric='accuracy'.
    """
    device = torch.device(device)
    rng = np.random.RandomState(seed)

    trajs = np.zeros((n_trajectories, n_steps + 1, grid_size, grid_size),
                     dtype=np.float32)
    for i in range(n_trajectories):
        grid = (rng.rand(grid_size, grid_size) < density).astype(np.float32)
        trajs[i, 0] = grid
        for t in range(n_steps):
            grid = _gol_step(grid)
            trajs[i, t + 1] = grid

    train_t, val_t, test_t = _split_trajectories(trajs)
    X_tr, Y_tr = _make_pairs(train_t)
    X_v, Y_v = _make_pairs(val_t)
    X_te, Y_te = _make_pairs(test_t)

    # Add channel dim
    X_tr = X_tr[:, None]; Y_tr = Y_tr[:, None]
    X_v  = X_v[:, None];  Y_v  = Y_v[:, None]
    X_te = X_te[:, None];  Y_te = Y_te[:, None]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "gol",
        "loss_type": "bce",
        "metric": "accuracy",
        "in_channels": 1,
        "out_channels": 1,
        "grid_size": grid_size,
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "density": density,
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  KURAMOTO-SIVASHINSKY
# ═══════════════════════════════════════════════════════════════════════════════

# KS PDE parameters
_KS_L = 22.0
_KS_DT = 0.05
_KS_SUB = 5
_KS_WARMUP = 200


def _ks_step_spectral(u: np.ndarray, L: float, dt: float,
                      n_sub: int = _KS_SUB) -> np.ndarray:
    """Semi-implicit spectral step for KS equation.

    du/dt = -u*du/dx - d2u/dx2 - d4u/dx4
    """
    N = len(u)
    dt_sub = dt / n_sub
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    k2 = k ** 2
    k4 = k ** 4
    L_op = -k2 - k4

    for _ in range(n_sub):
        u_hat = np.fft.fft(u)
        du_dx = np.real(np.fft.ifft(1j * k * u_hat))
        nl = -u * du_dx
        nl_hat = np.fft.fft(nl)
        u_hat_new = (u_hat + dt_sub * nl_hat) / (1.0 - dt_sub * L_op)
        u = np.real(np.fft.ifft(u_hat_new)).astype(np.float32)
    return u


def _ks_random_ic(N: int, L: float,
                  rng: np.random.RandomState) -> np.ndarray:
    """Small random perturbation — KS generates its own chaos."""
    x = np.linspace(0, L, N, endpoint=False).astype(np.float32)
    u = np.zeros(N, dtype=np.float32)
    n_modes = rng.randint(3, 7)
    for _ in range(n_modes):
        kk = rng.randint(1, 6)
        amp = rng.uniform(0.01, 0.1)
        phase = rng.uniform(0, 2 * np.pi)
        u += amp * np.sin(2 * np.pi * kk * x / L + phase)
    return u


def generate_ks(grid_size: int = 64, n_trajectories: int = 200,
                n_steps: int = 100, seed: int = 42,
                device: str | torch.device = "cpu"):
    """Kuramoto-Sivashinsky equation benchmark.

    ``grid_size`` is used as the 1D spatial width (N).

    Returns (X_train, Y_train, X_val, Y_val, X_test, Y_test, meta).
    X/Y shapes: (N_samples, 1, 1, W) float32 in [0, 1] (globally normalized).
    """
    N = grid_size
    device = torch.device(device)
    rng = np.random.RandomState(seed)

    # Generate raw trajectories with warmup
    raw_trajs = []
    for i in range(n_trajectories):
        u0 = _ks_random_ic(N, _KS_L, rng)
        u = u0.copy()
        # Warm-up: let transients die out
        for _ in range(_KS_WARMUP):
            u = _ks_step_spectral(u, _KS_L, _KS_DT)
        traj = [u.copy()]
        for t in range(n_steps):
            u = _ks_step_spectral(u, _KS_L, _KS_DT)
            traj.append(u.copy())
        raw_trajs.append(np.array(traj, dtype=np.float32))

    # Global normalization to [0, 1]
    all_vals = np.concatenate([t.ravel() for t in raw_trajs])
    g_min, g_max = float(all_vals.min()), float(all_vals.max())

    trajs = np.zeros((n_trajectories, n_steps + 1, 1, N), dtype=np.float32)
    for i, raw in enumerate(raw_trajs):
        normed = (raw - g_min) / (g_max - g_min + 1e-8)
        trajs[i, :, 0, :] = normed

    train_t, val_t, test_t = _split_trajectories(trajs)

    # Make pairs: shape (N_samples, 1, W) -> add H=1 -> (N_samples, 1, 1, W)
    def _pairs_1d(tr):
        X = tr[:, :-1].reshape(-1, 1, N)
        Y = tr[:, 1:].reshape(-1, 1, N)
        # Add H=1 dimension
        X = X[:, :, None, :]  # (N_samples, 1, 1, W)
        Y = Y[:, :, None, :]
        return X, Y

    X_tr, Y_tr = _pairs_1d(train_t)
    X_v, Y_v = _pairs_1d(val_t)
    X_te, Y_te = _pairs_1d(test_t)

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "ks",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": 1,
        "out_channels": 1,
        "spatial_dims": "1d",
        "grid_width": N,
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "L": _KS_L,
        "dt": _KS_DT,
        "n_sub": _KS_SUB,
        "warmup_steps": _KS_WARMUP,
        "norm_min": g_min,
        "norm_max": g_max,
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAY-SCOTT REACTION-DIFFUSION
# ═══════════════════════════════════════════════════════════════════════════════

# Gray-Scott parameters (mitosis pattern)
_GS_D_U = 0.16
_GS_D_V = 0.08
_GS_F_FEED = 0.035
_GS_K_KILL = 0.065
_GS_DT = 1.0
_GS_DX = 1.0
_GS_N_SUBSTEPS = 4
_GS_DT_SUB = _GS_DT / _GS_N_SUBSTEPS


def _build_laplacian_kernel():
    """Standard 5-point 2D Laplacian stencil as conv kernel."""
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    return torch.from_numpy(k).reshape(1, 1, 3, 3)


def _gray_scott_step_torch(u: torch.Tensor, v: torch.Tensor,
                           lap_kernel: torch.Tensor,
                           d_u: float = _GS_D_U, d_v: float = _GS_D_V,
                           f: float = _GS_F_FEED, k: float = _GS_K_KILL,
                           dt_sub: float = _GS_DT_SUB, dx: float = _GS_DX,
                           n_sub: int = _GS_N_SUBSTEPS):
    """One macro-step of Gray-Scott (n_sub sub-steps).

    u, v: (B, 1, H, W). Returns updated (u, v).
    Uses periodic padding for Laplacian.
    """
    import torch.nn.functional as F
    coeff = dt_sub / (dx * dx)
    for _ in range(n_sub):
        u_pad = F.pad(u, (1, 1, 1, 1), mode='circular')
        v_pad = F.pad(v, (1, 1, 1, 1), mode='circular')
        lap_u = F.conv2d(u_pad, lap_kernel)
        lap_v = F.conv2d(v_pad, lap_kernel)
        uvv = u * v * v
        du = d_u * coeff * lap_u - uvv + f * (1.0 - u)
        dv = d_v * coeff * lap_v + uvv - (f + k) * v
        u = u + dt_sub * du
        v = v + dt_sub * dv
        u = u.clamp(0.0, 1.0)
        v = v.clamp(0.0, 1.0)
    return u, v


def generate_gray_scott(grid_size: int = 32, n_trajectories: int = 200,
                        n_steps: int = 100, seed: int = 42,
                        device: str | torch.device = "cpu"):
    """Gray-Scott reaction-diffusion benchmark.

    Returns (X_train, Y_train, X_val, Y_val, X_test, Y_test, meta).
    X/Y shapes: (N, 2, H, W) float32 in [0, 1] (per-channel normalized).
    """
    device = torch.device(device)
    rng = np.random.RandomState(seed)
    lap_k = _build_laplacian_kernel()

    h = w = grid_size
    trajs = np.zeros((n_trajectories, n_steps + 1, 2, h, w), dtype=np.float32)

    # Batched initial conditions
    u = torch.ones(n_trajectories, 1, h, w)
    v = torch.zeros(n_trajectories, 1, h, w)

    ch, cw = h // 2, w // 2
    noise = torch.from_numpy(
        rng.uniform(-0.01, 0.01,
                    (n_trajectories, 1, 4, 4)).astype(np.float32)
    )
    v[:, :, ch - 2:ch + 2, cw - 2:cw + 2] = 0.25 + noise
    u[:, :, ch - 2:ch + 2, cw - 2:cw + 2] = 0.5

    trajs[:, 0, 0] = u.squeeze(1).numpy()
    trajs[:, 0, 1] = v.squeeze(1).numpy()

    for t in range(n_steps):
        u, v = _gray_scott_step_torch(u, v, lap_k)
        trajs[:, t + 1, 0] = u.squeeze(1).numpy()
        trajs[:, t + 1, 1] = v.squeeze(1).numpy()

    # Per-channel normalization to [0, 1]
    norm_stats = {}
    for ch_idx, ch_name in enumerate(["u", "v"]):
        ch_data = trajs[:, :, ch_idx]
        cmin = float(ch_data.min())
        cmax = float(ch_data.max())
        crng = cmax - cmin
        if crng < 1e-8:
            crng = 1.0
        trajs[:, :, ch_idx] = (ch_data - cmin) / crng
        norm_stats[ch_name] = {"min": cmin, "max": cmax, "range": crng}

    train_t, val_t, test_t = _split_trajectories(trajs)

    def _pairs_2ch(tr):
        X = tr[:, :-1].reshape(-1, 2, h, w)
        Y = tr[:, 1:].reshape(-1, 2, h, w)
        return X, Y

    X_tr, Y_tr = _pairs_2ch(train_t)
    X_v, Y_v = _pairs_2ch(val_t)
    X_te, Y_te = _pairs_2ch(test_t)

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "gray_scott",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": 2,
        "out_channels": 2,
        "grid_size": grid_size,
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "d_u": _GS_D_U,
        "d_v": _GS_D_V,
        "f_feed": _GS_F_FEED,
        "k_kill": _GS_K_KILL,
        "dt": _GS_DT,
        "dx": _GS_DX,
        "n_substeps": _GS_N_SUBSTEPS,
        "norm_stats": norm_stats,
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  RULE 110
# ═══════════════════════════════════════════════════════════════════════════════

_RULE_110_TABLE = {
    (1, 1, 1): 0, (1, 1, 0): 1, (1, 0, 1): 1, (1, 0, 0): 0,
    (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): 0,
}


def _rule110_step(row: np.ndarray) -> np.ndarray:
    """One step of Rule 110 with wrap-around boundaries."""
    W = len(row)
    out = np.zeros(W, dtype=np.float32)
    for i in range(W):
        left = int(row[(i - 1) % W])
        centre = int(row[i])
        right = int(row[(i + 1) % W])
        out[i] = _RULE_110_TABLE[(left, centre, right)]
    return out


def generate_rule110(grid_size: int = 64, n_trajectories: int = 500,
                     n_steps: int = 30, density: float = 0.5,
                     seed: int = 42, device: str | torch.device = "cpu"):
    """Rule 110 1D cellular automaton benchmark.

    ``grid_size`` is used as the 1D width.

    Returns (X_train, Y_train, X_val, Y_val, X_test, Y_test, meta).
    X/Y shapes: (N, 1, 1, W) float32 binary.
    loss_type='bce', metric='accuracy'.
    """
    width = grid_size
    device = torch.device(device)
    rng = np.random.RandomState(seed)

    # Shape: (n_traj, T+1, 1, 1, W)
    trajs = np.zeros((n_trajectories, n_steps + 1, 1, 1, width),
                     dtype=np.float32)
    for i in range(n_trajectories):
        row = (rng.rand(width) < density).astype(np.float32)
        trajs[i, 0, 0, 0] = row
        for t in range(n_steps):
            row = _rule110_step(row)
            trajs[i, t + 1, 0, 0] = row

    train_t, val_t, test_t = _split_trajectories(trajs)

    def _pairs(tr):
        N, Tp1 = tr.shape[:2]
        rest = tr.shape[2:]
        X = tr[:, :-1].reshape(-1, *rest)
        Y = tr[:, 1:].reshape(-1, *rest)
        return X, Y

    X_tr, Y_tr = _pairs(train_t)
    X_v, Y_v = _pairs(val_t)
    X_te, Y_te = _pairs(test_t)

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "rule110",
        "loss_type": "bce",
        "metric": "accuracy",
        "in_channels": 1,
        "out_channels": 1,
        "spatial_dims": "1d",
        "grid_width": width,
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "density": density,
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  WIREWORLD
# ═══════════════════════════════════════════════════════════════════════════════
# States: empty=0, head=1, tail=2, conductor=3


def _wireworld_step(grid: np.ndarray) -> np.ndarray:
    """One Wireworld step.  grid: (H, W) int in {0,1,2,3}."""
    H, W = grid.shape
    out = np.zeros_like(grid)
    for y in range(H):
        for x in range(W):
            s = grid[y, x]
            if s == 0:
                out[y, x] = 0
            elif s == 1:
                out[y, x] = 2
            elif s == 2:
                out[y, x] = 3
            elif s == 3:
                heads = 0
                for dy in (-1, 0, 1):
                    for dx_off in (-1, 0, 1):
                        if dy == 0 and dx_off == 0:
                            continue
                        ny, nx = (y + dy) % H, (x + dx_off) % W
                        if grid[ny, nx] == 1:
                            heads += 1
                out[y, x] = 1 if heads in (1, 2) else 3
    return out


def _onehot_ww(grid_int: np.ndarray) -> np.ndarray:
    """(H, W) int -> (4, H, W) float32 one-hot."""
    H, W = grid_int.shape
    oh = np.zeros((4, H, W), dtype=np.float32)
    for c in range(4):
        oh[c] = (grid_int == c).astype(np.float32)
    return oh


def generate_wireworld(grid_size: int = 16, n_trajectories: int = 500,
                       n_steps: int = 20, conductor_density: float = 0.30,
                       n_heads: int = 2, seed: int = 42,
                       device: str | torch.device = "cpu"):
    """Wireworld 2D cellular automaton benchmark.

    Returns (X_train, Y_train, X_val, Y_val, X_test, Y_test, meta).
    X/Y shapes: (N, 4, H, W) float32 one-hot encoded.
    loss_type='cross_entropy', metric='accuracy'.
    """
    device = torch.device(device)
    rng = np.random.RandomState(seed)

    H = W = grid_size
    trajs = np.zeros((n_trajectories, n_steps + 1, 4, H, W), dtype=np.float32)
    for i in range(n_trajectories):
        grid = np.zeros((H, W), dtype=np.int32)
        conductor_mask = rng.rand(H, W) < conductor_density
        grid[conductor_mask] = 3
        conductor_cells = list(zip(*np.where(grid == 3)))
        if len(conductor_cells) >= n_heads:
            head_idxs = rng.choice(len(conductor_cells), size=n_heads,
                                   replace=False)
            for idx in head_idxs:
                y, x = conductor_cells[idx]
                grid[y, x] = 1
        trajs[i, 0] = _onehot_ww(grid)
        for t in range(n_steps):
            grid = _wireworld_step(grid)
            trajs[i, t + 1] = _onehot_ww(grid)

    train_t, val_t, test_t = _split_trajectories(trajs)

    def _pairs(tr):
        N, Tp1 = tr.shape[:2]
        rest = tr.shape[2:]
        X = tr[:, :-1].reshape(-1, *rest)
        Y = tr[:, 1:].reshape(-1, *rest)
        return X, Y

    X_tr, Y_tr = _pairs(train_t)
    X_v, Y_v = _pairs(val_t)
    X_te, Y_te = _pairs(test_t)

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "wireworld",
        "loss_type": "cross_entropy",
        "metric": "accuracy",
        "in_channels": 4,
        "out_channels": 4,
        "grid_size": grid_size,
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "conductor_density": conductor_density,
        "n_heads": n_heads,
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRID WORLD (Planning)
# ═══════════════════════════════════════════════════════════════════════════════

N_CELL_TYPES = 4   # empty=0, wall=1, agent=2, goal=3
N_ACTIONS = 4      # up=0, down=1, left=2, right=3
_GW_WALL_DENSITY = 0.20


class GridWorldEnv:
    """16x16 grid world with walls, agent, and goal.

    Identical to SimpleGridWorld from grid_world_planning.py,
    exposed here for CEM evaluation.
    """

    def __init__(self, grid_size: int = 16,
                 wall_density: float = _GW_WALL_DENSITY,
                 rng: np.random.Generator | None = None):
        self.grid_size = grid_size
        self.wall_density = wall_density
        self.rng = rng or np.random.default_rng(42)
        self.reset()

    def reset(self) -> np.ndarray:
        gs = self.grid_size
        self.grid = np.zeros((gs, gs), dtype=np.int64)
        wall_mask = self.rng.random((gs, gs)) < self.wall_density
        wall_mask[0, :] = False
        wall_mask[-1, :] = False
        wall_mask[:, 0] = False
        wall_mask[:, -1] = False
        self.grid[wall_mask] = 1

        empty = np.argwhere(self.grid == 0)
        idx = self.rng.integers(len(empty))
        self.agent_pos = tuple(empty[idx])
        self.grid[self.agent_pos] = 2

        empty = np.argwhere(self.grid == 0)
        idx = self.rng.integers(len(empty))
        self.goal_pos = tuple(empty[idx])
        self.grid[self.goal_pos] = 3
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = np.zeros((N_CELL_TYPES, self.grid_size, self.grid_size),
                         dtype=np.float32)
        for c in range(N_CELL_TYPES):
            state[c] = (self.grid == c).astype(np.float32)
        return state

    def step(self, action: int):
        ar, ac = self.agent_pos
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = ar + dr, ac + dc

        if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                and self.grid[nr, nc] != 1):
            self.grid[ar, ac] = 0
            self.agent_pos = (nr, nc)
            if self.agent_pos == self.goal_pos:
                self.grid[nr, nc] = 2
                return self._get_state(), 1.0, True
            self.grid[nr, nc] = 2

        if self.grid[self.goal_pos] == 0:
            self.grid[self.goal_pos] = 3
        return self._get_state(), 0.0, False

    def clone(self) -> "GridWorldEnv":
        env = GridWorldEnv.__new__(GridWorldEnv)
        env.grid_size = self.grid_size
        env.wall_density = self.wall_density
        env.rng = np.random.default_rng()
        env.grid = self.grid.copy()
        env.agent_pos = self.agent_pos
        env.goal_pos = self.goal_pos
        return env


def generate_grid_world(grid_size: int = 16, n_trajectories: int = 500,
                        n_transitions: int | None = None,
                        seed: int = 42,
                        device: str | torch.device = "cpu"):
    """Grid world transition benchmark for planning.

    Unlike other benchmarks, this generates action-conditioned transitions.
    ``n_trajectories`` is accepted for API compatibility with the runner;
    the actual count is ``n_transitions`` (defaults to ``n_trajectories * 10``).

    Returns a BenchmarkData where:
      X_train/X_test  = state concatenated with action field  (N, 8, H, W)
      Y_train/Y_test  = next state one-hot                   (N, 4, H, W)

    meta includes 'env_class' for instantiating GridWorldEnv.
    """
    if n_transitions is None:
        n_transitions = n_trajectories * 10

    device = torch.device(device)
    rng = np.random.default_rng(seed)

    states_list = []
    action_fields_list = []
    next_states_list = []

    env = GridWorldEnv(grid_size=grid_size, rng=rng)
    collected = 0

    while collected < n_transitions:
        state = env.reset()
        for _ in range(20):
            action = int(rng.integers(N_ACTIONS))
            # Build action field
            af = np.zeros((N_ACTIONS, grid_size, grid_size), dtype=np.float32)
            agent_ch = state[2]
            agent_pos = np.argwhere(agent_ch > 0.5)
            if len(agent_pos) > 0:
                ar, ac = agent_pos[0]
                af[action, ar, ac] = 1.0

            next_state, _, done = env.step(action)

            states_list.append(state)
            action_fields_list.append(af)
            next_states_list.append(next_state)
            collected += 1

            if collected >= n_transitions or done:
                break
            state = next_state

    states = np.stack(states_list[:n_transitions])
    action_fields = np.stack(action_fields_list[:n_transitions])
    next_states = np.stack(next_states_list[:n_transitions])

    # Concatenate state + action field -> X  (N, 8, H, W)
    X = np.concatenate([states, action_fields], axis=1)
    Y = next_states  # (N, 4, H, W)

    # 70/15/15 split
    n = n_transitions
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_tr = X[:n_train]
    Y_tr = Y[:n_train]
    X_v = X[n_train:n_train + n_val]
    Y_v = Y[n_train:n_train + n_val]
    X_te = X[n_train + n_val:]
    Y_te = Y[n_train + n_val:]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "grid_world",
        "loss_type": "cross_entropy",
        "metric": "accuracy",
        "in_channels": N_CELL_TYPES + N_ACTIONS,  # 8 (state + action field)
        "out_channels": N_CELL_TYPES,  # 4
        "grid_size": grid_size,
        "n_transitions": n_transitions,
        "n_actions": N_ACTIONS,
        "n_cell_types": N_CELL_TYPES,
        "wall_density": _GW_WALL_DENSITY,
        "env_class": GridWorldEnv,
        # Action-conditioned: X has state+action channels, Y has only state
        # channels, so naive autoregressive rollout is not applicable.
        # CEM planning is the proper evaluation for this benchmark.
        "action_conditioned": True,
    }
    return BenchmarkData(*data, meta)


def _exhaustive_action_sequences(n_actions: int, horizon: int) -> torch.Tensor:
    """Generate all possible action sequences.

    Returns tensor of shape (n_actions^horizon, horizon) with dtype long.
    """
    import itertools
    seqs = list(itertools.product(range(n_actions), repeat=horizon))
    return torch.tensor(seqs, dtype=torch.long)


def run_cem_evaluation(
    model,
    grid_world_data,
    n_episodes: int = 200,
    horizon: int = 5,
    population: int = 200,
    elite_k: int = 40,
    cem_iters: int = 3,
    max_steps: int = 50,
    device: str | torch.device = "cpu",
    seed: int = 42,
    use_exhaustive: bool = True,
    use_soft_predictions: bool = True,
) -> dict:
    """CEM / exhaustive planning evaluation for the grid_world benchmark.

    Uses ``grid_world_data.meta['env_class']`` to instantiate fresh
    environments. The model takes an 8-channel input (4 state + 4
    action-field) and outputs a 4-channel prediction of the next state
    (logits over cell types).

    When ``use_exhaustive=True`` (default) and the action/horizon space
    is small enough (n_actions <= 5, horizon <= 6), ALL action sequences
    are enumerated (e.g. 4^5 = 1024), eliminating CEM sampling noise
    entirely.  Falls back to CEM otherwise.

    When ``use_soft_predictions=True`` (default), rollouts keep softmax
    probabilities instead of argmax one-hot states, preventing
    catastrophic agent misplacement from single-pixel prediction errors.

    Evaluation environments use a FIXED seed (12345) independent of the
    training seed so that different model seeds are compared on the same
    set of maps (paired evaluation).

    Returns dict with ``success_rate``, ``avg_steps``, ``avg_reward``,
    and ``planning_method`` ('Exhaustive' or 'CEM').
    """
    import torch.nn.functional as F

    dev = torch.device(device) if isinstance(device, str) else device

    meta = getattr(grid_world_data, "meta", {}) or {}
    env_class = meta.get("env_class")
    if env_class is None:
        return {
            "success_rate": float("nan"),
            "avg_steps": float("nan"),
            "avg_reward": float("nan"),
            "planning_method": "N/A",
        }

    grid_size = meta.get("grid_size", 16)
    n_cell_types = meta.get("n_cell_types", N_CELL_TYPES)
    n_actions = meta.get("n_actions", N_ACTIONS)

    if hasattr(model, "to"):
        model = model.to(dev)
    if hasattr(model, "eval"):
        model.eval()

    # Fixed eval seed — independent of training seed so different
    # model seeds are evaluated on the same set of maps.
    eval_rng = np.random.default_rng(12345)

    do_exhaustive = (
        use_exhaustive
        and n_actions <= 5
        and horizon <= 6
    )
    if do_exhaustive:
        all_action_seqs = _exhaustive_action_sequences(n_actions, horizon).to(dev)
        total_seqs = len(all_action_seqs)
        planning_method = "Exhaustive"
    else:
        total_seqs = population
        planning_method = "CEM"

    print(f"[CEM eval] method={planning_method}, horizon={horizon}, "
          f"n_episodes={n_episodes}, soft={use_soft_predictions}")

    successes = 0
    total_steps_success = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        env = env_class(grid_size=grid_size, rng=eval_rng)
        state = env.reset()
        goal_r, goal_c = env.goal_pos

        ep_reward = 0.0
        for step in range(max_steps):
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(dev)

            if do_exhaustive:
                # --- Exhaustive search: evaluate ALL action sequences ---
                action_seqs = all_action_seqs  # (total_seqs, horizon)
                n_seqs = total_seqs

                cur_states = state_t.expand(n_seqs, -1, -1, -1).clone()
                chunk_size = 256  # process in chunks to avoid OOM
                for t in range(horizon):
                    acts = action_seqs[:, t]  # (n_seqs,)

                    agent_ch = cur_states[:, 2]  # (n_seqs, H, W)
                    agent_flat = agent_ch.reshape(n_seqs, -1).argmax(dim=1)
                    ar = agent_flat // grid_size
                    ac = agent_flat % grid_size

                    af = torch.zeros(n_seqs, n_actions, grid_size, grid_size,
                                     device=dev)
                    bidx = torch.arange(n_seqs, device=dev)
                    af[bidx, acts, ar, ac] = 1.0

                    x = torch.cat([cur_states, af], dim=1)  # (n_seqs, 8, H, W)
                    # Forward in chunks to keep memory manageable
                    logits_list = []
                    with torch.no_grad():
                        for ci in range(0, n_seqs, chunk_size):
                            logits_list.append(model(x[ci:ci + chunk_size]))
                    logits = torch.cat(logits_list, dim=0)

                    if use_soft_predictions:
                        cur_states = F.softmax(logits, dim=1)
                    else:
                        pred_cls = logits.argmax(dim=1)  # (n_seqs, H, W)
                        cur_states = F.one_hot(pred_cls, n_cell_types).permute(
                            0, 3, 1, 2).float()

                # Score: distance to goal
                agent_ch = cur_states[:, 2]
                agent_flat = agent_ch.reshape(n_seqs, -1).argmax(dim=1)
                pred_ar = agent_flat // grid_size
                pred_ac = agent_flat % grid_size

                reached = (pred_ar == goal_r) & (pred_ac == goal_c)
                dist = (pred_ar - goal_r).abs() + (pred_ac - goal_c).abs()
                rewards = torch.where(
                    reached,
                    torch.ones_like(dist, dtype=torch.float32),
                    -dist.float(),
                )

                best_idx = rewards.argmax()
                best_action = int(action_seqs[best_idx, 0].item())

            else:
                # --- CEM planning ---
                action_probs = torch.ones(horizon, n_actions, device=dev) / n_actions
                elite_idx = None
                action_seqs = None

                for _ in range(cem_iters):
                    action_seqs = torch.zeros(population, horizon,
                                              dtype=torch.long, device=dev)
                    for t in range(horizon):
                        action_seqs[:, t] = torch.multinomial(
                            action_probs[t].unsqueeze(0).expand(population, -1),
                            1,
                        ).squeeze(-1)

                    # Batched rollout through the world model
                    cur_states = state_t.expand(population, -1, -1, -1).clone()
                    for t in range(horizon):
                        acts = action_seqs[:, t]  # (population,)

                        agent_ch = cur_states[:, 2]  # (population, H, W)
                        agent_flat = agent_ch.reshape(population, -1).argmax(dim=1)
                        ar = agent_flat // grid_size
                        ac = agent_flat % grid_size

                        af = torch.zeros(population, n_actions, grid_size, grid_size,
                                         device=dev)
                        bidx = torch.arange(population, device=dev)
                        af[bidx, acts, ar, ac] = 1.0

                        x = torch.cat([cur_states, af], dim=1)  # (pop, 8, H, W)
                        with torch.no_grad():
                            logits = model(x)

                        if use_soft_predictions:
                            cur_states = F.softmax(logits, dim=1)
                        else:
                            pred_cls = logits.argmax(dim=1)  # (pop, H, W)
                            cur_states = F.one_hot(pred_cls, n_cell_types).permute(
                                0, 3, 1, 2).float()

                    # Reward: reached goal -> +1; else negative manhattan distance
                    agent_ch = cur_states[:, 2]
                    agent_flat = agent_ch.reshape(population, -1).argmax(dim=1)
                    pred_ar = agent_flat // grid_size
                    pred_ac = agent_flat % grid_size

                    reached = (pred_ar == goal_r) & (pred_ac == goal_c)
                    dist = (pred_ar - goal_r).abs() + (pred_ac - goal_c).abs()
                    rewards = torch.where(
                        reached,
                        torch.ones_like(dist, dtype=torch.float32),
                        -dist.float(),
                    )

                    _, elite_idx = rewards.topk(elite_k)
                    elite_actions = action_seqs[elite_idx]  # (elite_k, horizon)
                    for t in range(horizon):
                        counts = torch.zeros(n_actions, device=dev)
                        for a in range(n_actions):
                            counts[a] = (elite_actions[:, t] == a).float().sum()
                        action_probs[t] = (counts + 0.1) / (elite_k + 0.1 * n_actions)

                best_action = int(action_seqs[elite_idx[0], 0].item())

            # Execute chosen action in real environment
            state, reward, done = env.step(best_action)
            ep_reward += reward

            if done:
                successes += 1
                total_steps_success += step + 1
                break

        total_reward += ep_reward

    return {
        "success_rate": successes / max(n_episodes, 1),
        "avg_steps": total_steps_success / max(successes, 1),
        "avg_reward": total_reward / max(n_episodes, 1),
        "planning_method": planning_method,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HEAT CONTROL (Planning -- CML-native physics)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_heat_control(grid_size: int = 16, n_trajectories: int = 500,
                          n_transitions: int | None = None,
                          episode_length: int = 50, alpha: float = 0.1,
                          seed: int = 42,
                          device: str | torch.device = "cpu"):
    """Heat control RL benchmark for world-model-based planning.

    A 2D heat equation environment where the agent toggles point heat
    sources to match a target temperature distribution.  The dynamics
    ARE the discrete heat equation -- exactly the inductive bias baked
    into our CML reservoir.

    Returns BenchmarkData where:
      X_train/X_test  = [temp, sources, target, action_map]  (N, 4, H, W)
      Y_train/Y_test  = [next_temperature]                   (N, 1, H, W)

    meta includes 'env_class' for CEM evaluation.
    """
    from wmca.envs.heat_control import HeatControlEnv, generate_heat_control_transitions

    if n_transitions is None:
        n_transitions = n_trajectories  # n_trajectories = n_episodes

    device = torch.device(device)

    env = HeatControlEnv(
        grid_size=grid_size,
        alpha=alpha,
        episode_length=episode_length,
        seed=seed,
    )

    X_all, Y_all = generate_heat_control_transitions(
        env, n_episodes=n_transitions, seed=seed,
    )

    # 70/15/15 split
    n = len(X_all)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_tr = X_all[:n_train]
    Y_tr = Y_all[:n_train]
    X_v = X_all[n_train:n_train + n_val]
    Y_v = Y_all[n_train:n_train + n_val]
    X_te = X_all[n_train + n_val:]
    Y_te = Y_all[n_train + n_val:]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "heat_control",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": 4,   # temp + sources + target + action_map
        "out_channels": 1,  # predicted next temperature
        "grid_size": grid_size,
        "n_episodes": n_transitions,
        "episode_length": episode_length,
        "alpha_heat": alpha,
        "action_conditioned": True,
        "env_class": HeatControlEnv,
        "env_kwargs": {
            "grid_size": grid_size,
            "alpha": alpha,
            "episode_length": episode_length,
        },
    }
    return BenchmarkData(*data, meta)


def run_heat_control_cem(
    model,
    heat_control_data,
    n_episodes: int = 100,
    horizon: int = 5,
    population: int = 200,
    elite_k: int = 40,
    cem_iters: int = 3,
    max_steps: int = 50,
    device: str | torch.device = "cpu",
    mse_threshold: float = 0.01,
) -> dict:
    """CEM planning evaluation for the heat_control benchmark.

    Uses the world model to plan action sequences that minimize the MSE
    between the temperature field and the target.

    Success is defined as final MSE(temp, target) < mse_threshold.

    Returns dict with success_rate, avg_final_mse, avg_reward,
    planning_method.
    """
    from wmca.envs.heat_control import HeatControlEnv

    dev = torch.device(device) if isinstance(device, str) else device

    meta = getattr(heat_control_data, "meta", {}) or {}
    env_kwargs = meta.get("env_kwargs", {})
    grid_size = env_kwargs.get("grid_size", 16)

    if hasattr(model, "to"):
        model = model.to(dev)
    if hasattr(model, "eval"):
        model.eval()

    # Fixed eval seed for reproducibility
    eval_rng = np.random.default_rng(12345)

    n_actions = grid_size * grid_size + 1  # toggle cells + no-op

    planning_method = "CEM"
    print(f"[Heat CEM] method={planning_method}, horizon={horizon}, "
          f"n_episodes={n_episodes}, pop={population}")

    successes = 0
    total_mse = 0.0
    total_reward = 0.0

    for ep in range(n_episodes):
        env = HeatControlEnv(**env_kwargs, rng=eval_rng)
        state = env.reset()

        ep_reward = 0.0
        for step_idx in range(max_steps):
            # CEM planning over the world model
            # Represent action distributions as categorical over all cells + no-op
            # For efficiency, sample from a smaller candidate set each CEM iter
            action_logits = torch.zeros(horizon, n_actions, device=dev)
            best_seq = None

            for _cem_it in range(cem_iters):
                # Sample action sequences
                probs = torch.softmax(action_logits, dim=-1)  # (horizon, n_actions)
                action_seqs = torch.multinomial(
                    probs.unsqueeze(0).expand(population, -1, -1).reshape(
                        population * horizon, n_actions
                    ),
                    1,
                ).reshape(population, horizon)  # (pop, horizon)

                # Batched rollout through world model
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(dev)
                cur_temp = state_t[:, 0:1].expand(population, -1, -1, -1).clone()
                cur_sources = state_t[:, 1:2].expand(population, -1, -1, -1).clone()
                cur_target = state_t[:, 2:3].expand(population, -1, -1, -1).clone()

                for t in range(horizon):
                    acts = action_seqs[:, t]  # (pop,)

                    # Toggle sources based on actions
                    action_map = torch.zeros(
                        population, 1, grid_size, grid_size, device=dev
                    )
                    valid = acts < grid_size * grid_size
                    if valid.any():
                        valid_acts = acts[valid]
                        r = valid_acts // grid_size
                        c = valid_acts % grid_size
                        bidx = torch.arange(population, device=dev)[valid]
                        action_map[bidx, 0, r, c] = 1.0

                    # Update sources (toggle)
                    cur_sources = torch.where(
                        action_map > 0.5,
                        1.0 - cur_sources,  # toggle
                        cur_sources,
                    )

                    # Build input: [temp, sources, target, action_map] = (pop, 4, H, W)
                    x = torch.cat([cur_temp, cur_sources, cur_target, action_map], dim=1)

                    with torch.no_grad():
                        pred_temp = model(x)  # (pop, 1, H, W)

                    cur_temp = pred_temp.clamp(0.0, 1.0)

                # Score: negative MSE to target
                mse_per_sample = (cur_temp - cur_target).pow(2).mean(dim=(1, 2, 3))
                rewards = -mse_per_sample  # (pop,)

                # Elite selection
                _, elite_idx = rewards.topk(elite_k)
                elite_actions = action_seqs[elite_idx]  # (elite_k, horizon)

                # Update action logits from elite counts
                for t in range(horizon):
                    counts = torch.zeros(n_actions, device=dev)
                    for a_idx in range(n_actions):
                        counts[a_idx] = (elite_actions[:, t] == a_idx).float().sum()
                    # Smooth and convert to logits
                    action_logits[t] = torch.log((counts + 0.1) / (elite_k + 0.1 * n_actions))

                best_seq = action_seqs[elite_idx[0]]

            # Execute first action from best sequence
            best_action = int(best_seq[0].item())
            state, reward, done, info = env.step(best_action)
            ep_reward += reward

            if done:
                break

        # Final MSE after episode
        final_mse = float(np.mean((env.temperature - env.target) ** 2))
        total_mse += final_mse
        total_reward += ep_reward
        if final_mse < mse_threshold:
            successes += 1

    return {
        "success_rate": successes / max(n_episodes, 1),
        "avg_final_mse": total_mse / max(n_episodes, 1),
        "avg_reward": total_reward / max(n_episodes, 1),
        "planning_method": planning_method,
        "mse_threshold": mse_threshold,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DMCONTROL (cartpole-swingup)
# ═══════════════════════════════════════════════════════════════════════════════

# Default path for pre-generated DMControl data
_DMCONTROL_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def generate_dmcontrol(grid_size: int = 16, n_trajectories: int = 500,
                       seed: int = 42,
                       device: str | torch.device = "cpu"):
    """DMControl cartpole-swingup benchmark (flat state vectors).

    Loads pre-generated transition data from data/dmcontrol_cartpole.npz.
    The data consists of (state, action) -> next_state pairs where
    state is 5D and action is 1D.

    To match the unified ablation tensor format (N, C, H, W), the input
    is packed as channels: X = (N, state_dim + action_dim, 1, 1) and
    Y = (N, state_dim, 1, 1). This mirrors how grid_world handles
    action-conditioned data with different in/out channel counts.

    Note: ``grid_size`` is accepted for API compatibility but ignored —
    DMControl states have fixed dimensionality.

    Requires pre-generated data from experiments/dmcontrol_data_gen.py.
    Skips gracefully if the data file is not found.
    """
    data_path = _DMCONTROL_DATA_DIR / "dmcontrol_cartpole.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"DMControl data not found at {data_path}. "
            f"Generate it first with: python experiments/dmcontrol_data_gen.py"
        )

    device = torch.device(device)
    data = np.load(data_path)

    states = data["states"]           # (500, 100, 5)
    actions = data["actions"]         # (500, 100, 1)
    next_states = data["next_states"] # (500, 100, 5)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    state_dim = int(data["state_dim"])
    action_dim = actions.shape[-1]

    # Normalize states to [0, 1] using training data stats
    train_s = states[train_idx]
    train_ns = next_states[train_idx]
    all_train = np.concatenate([
        train_s.reshape(-1, state_dim),
        train_ns.reshape(-1, state_dim),
    ], axis=0)
    s_min = all_train.min(axis=0)  # (D,)
    s_max = all_train.max(axis=0)  # (D,)
    s_range = s_max - s_min
    s_range[s_range < 1e-8] = 1.0

    def norm_s(s):
        return ((s - s_min) / s_range).astype(np.float32)

    def norm_a(a):
        return ((a + 1.0) / 2.0).astype(np.float32)

    def make_pairs(idx):
        """Flatten trajectories and pack as (N, C, 1, 1) tensors."""
        s = norm_s(states[idx]).reshape(-1, state_dim)      # (N, D)
        a = norm_a(actions[idx]).reshape(-1, action_dim)     # (N, A)
        ns = norm_s(next_states[idx]).reshape(-1, state_dim) # (N, D)

        # X: concat state and action as channels -> (N, D+A, 1, 1)
        sa = np.concatenate([s, a], axis=1)  # (N, D+A)
        X = sa[:, :, None, None]             # (N, D+A, 1, 1)
        Y = ns[:, :, None, None]             # (N, D, 1, 1)
        return X, Y

    X_tr, Y_tr = make_pairs(train_idx)
    X_v, Y_v = make_pairs(val_idx)
    X_te, Y_te = make_pairs(test_idx)

    tensors = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "dmcontrol",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": state_dim + action_dim,  # 6 for cartpole
        "out_channels": state_dim,              # 5 for cartpole
        "grid_size": 1,
        "spatial_dims": "0d",
        "action_conditioned": True,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "task": "cartpole-swingup",
        "note": "Flat state vectors — MLP expected to dominate over CML/conv",
    }
    return BenchmarkData(*tensors, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  CRAFTER LITE (Mixed spatial + symbolic dynamics)
# ═══════════════════════════════════════════════════════════════════════════════

# Cell types (terrain + entities)
_CL_GRASS = 0
_CL_WATER = 1
_CL_STONE = 2
_CL_TREE  = 3
_CL_ORE   = 4
_CL_AGENT = 5
_CL_ZOMBIE = 6
_CL_N_TYPES = 7

# Actions
_CL_UP    = 0
_CL_DOWN  = 1
_CL_LEFT  = 2
_CL_RIGHT = 3
_CL_HARVEST = 4
_CL_PLACE_WALL = 5
_CL_N_ACTIONS = 6

_CL_TREE_GROWTH_RATE = 0.10  # P(tree spreads to adjacent grass per step)
_CL_MOVE_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


class ResourceGrid:
    """Crafter-inspired grid with mixed spatial + symbolic dynamics.

    Terrain: grass(0), water(1), stone(2), tree(3), ore(4)
    Entities: agent(5), zombie(6)

    Spatial dynamics (CML should help):
      - Trees spread to adjacent grass cells (~10% per neighbor per step)
      - Zombies move toward agent (Manhattan greedy, 1 step/tick)

    Symbolic dynamics (CML wastes computation):
      - Harvesting removes a single adjacent tree/ore cell (no propagation)
      - Placing wall converts grass under agent to stone (point change)
    """

    def __init__(self, grid_size: int = 16, n_zombies: int = 2,
                 tree_growth_rate: float = _CL_TREE_GROWTH_RATE,
                 seed: int = 42):
        self.grid_size = grid_size
        self.n_zombies = n_zombies
        self.tree_growth_rate = tree_growth_rate
        self.rng = np.random.RandomState(seed)
        self.grid = None
        self.agent_pos = None
        self.zombie_positions = []
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        gs = self.grid_size

        # Base terrain: mostly grass with water bodies and stone clusters
        self.grid = np.full((gs, gs), _CL_GRASS, dtype=np.int32)

        # Water: 2-3 small rectangular pools
        n_pools = self.rng.randint(2, 4)
        for _ in range(n_pools):
            py = self.rng.randint(1, gs - 4)
            px = self.rng.randint(1, gs - 4)
            ph = self.rng.randint(2, 4)
            pw = self.rng.randint(2, 4)
            self.grid[py:py+ph, px:px+pw] = _CL_WATER

        # Stone clusters: 3-5 small patches
        n_stone = self.rng.randint(3, 6)
        for _ in range(n_stone):
            sy = self.rng.randint(0, gs - 2)
            sx = self.rng.randint(0, gs - 2)
            sh = self.rng.randint(1, 3)
            sw = self.rng.randint(1, 3)
            self.grid[sy:sy+sh, sx:sx+sw] = _CL_STONE

        # Trees: scattered on grass cells (~15% of remaining grass)
        grass_mask = self.grid == _CL_GRASS
        tree_mask = grass_mask & (self.rng.rand(gs, gs) < 0.15)
        self.grid[tree_mask] = _CL_TREE

        # Ore: scattered on grass cells (~5%)
        grass_mask = self.grid == _CL_GRASS
        ore_mask = grass_mask & (self.rng.rand(gs, gs) < 0.05)
        self.grid[ore_mask] = _CL_ORE

        # Place agent on a random grass cell
        grass_cells = list(zip(*np.where(self.grid == _CL_GRASS)))
        if len(grass_cells) == 0:
            # Fallback: clear center
            self.grid[gs // 2, gs // 2] = _CL_GRASS
            grass_cells = [(gs // 2, gs // 2)]
        idx = self.rng.randint(len(grass_cells))
        self.agent_pos = grass_cells[idx]
        self.grid[self.agent_pos] = _CL_AGENT

        # Place zombies on grass cells away from agent
        self.zombie_positions = []
        grass_cells = list(zip(*np.where(self.grid == _CL_GRASS)))
        # Sort by distance from agent (descending) to place zombies far away
        if len(grass_cells) > 0:
            dists = [abs(r - self.agent_pos[0]) + abs(c - self.agent_pos[1])
                     for r, c in grass_cells]
            sorted_idx = np.argsort(dists)[::-1]
            for i in range(min(self.n_zombies, len(grass_cells))):
                zpos = grass_cells[sorted_idx[i]]
                self.zombie_positions.append(zpos)
                self.grid[zpos] = _CL_ZOMBIE

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return one-hot encoded grid: (N_TYPES, H, W) float32."""
        gs = self.grid_size
        state = np.zeros((_CL_N_TYPES, gs, gs), dtype=np.float32)
        for c in range(_CL_N_TYPES):
            state[c] = (self.grid == c).astype(np.float32)
        return state

    def step(self, action: int) -> np.ndarray:
        """Execute one step: agent action -> tree growth -> zombie movement."""
        # --- 1. Agent action (symbolic: only 1-2 cells change) ---
        self._do_agent_action(action)

        # --- 2. Tree growth (SPATIAL: local coupling, CML should help) ---
        self._grow_trees()

        # --- 3. Zombie movement (SPATIAL: local, CML should help) ---
        self._move_zombies()

        return self._get_state()

    def _do_agent_action(self, action: int):
        ar, ac = self.agent_pos
        gs = self.grid_size

        if action < 4:
            # Movement
            dr, dc = _CL_MOVE_DELTAS[action]
            nr, nc = ar + dr, ac + dc
            if (0 <= nr < gs and 0 <= nc < gs
                    and self.grid[nr, nc] not in (_CL_WATER, _CL_STONE,
                                                  _CL_ZOMBIE)):
                # Move: old position becomes grass, new position becomes agent
                self.grid[ar, ac] = _CL_GRASS
                self.grid[nr, nc] = _CL_AGENT
                self.agent_pos = (nr, nc)

        elif action == _CL_HARVEST:
            # Harvest: remove one adjacent tree or ore (symbolic, no propagation)
            for dr, dc in _CL_MOVE_DELTAS:
                nr, nc = ar + dr, ac + dc
                if (0 <= nr < gs and 0 <= nc < gs
                        and self.grid[nr, nc] in (_CL_TREE, _CL_ORE)):
                    self.grid[nr, nc] = _CL_GRASS
                    break  # harvest only one

        elif action == _CL_PLACE_WALL:
            # Place wall: convert the cell the agent stands on to stone,
            # agent moves to an adjacent grass cell if possible
            for dr, dc in _CL_MOVE_DELTAS:
                nr, nc = ar + dr, ac + dc
                if (0 <= nr < gs and 0 <= nc < gs
                        and self.grid[nr, nc] == _CL_GRASS):
                    self.grid[ar, ac] = _CL_STONE
                    self.grid[nr, nc] = _CL_AGENT
                    self.agent_pos = (nr, nc)
                    break

    def _grow_trees(self):
        """Trees spread to adjacent grass cells with probability tree_growth_rate.

        This is LOCAL SPATIAL coupling: each tree independently tries to
        spread to its 4-connected grass neighbors. CML's local coupling
        kernels should capture this well.
        """
        gs = self.grid_size
        tree_cells = list(zip(*np.where(self.grid == _CL_TREE)))
        for tr, tc in tree_cells:
            for dr, dc in _CL_MOVE_DELTAS:
                nr, nc = tr + dr, tc + dc
                if (0 <= nr < gs and 0 <= nc < gs
                        and self.grid[nr, nc] == _CL_GRASS
                        and self.rng.rand() < self.tree_growth_rate):
                    self.grid[nr, nc] = _CL_TREE

    def _move_zombies(self):
        """Each zombie moves 1 step toward agent (Manhattan greedy).

        This is LOCAL SPATIAL: zombie position depends on relative
        position of agent. CML should partially capture this.
        """
        gs = self.grid_size
        ar, ac = self.agent_pos
        new_positions = []

        for zr, zc in self.zombie_positions:
            # Greedy: pick the move that minimizes Manhattan distance to agent
            best_pos = (zr, zc)
            best_dist = abs(zr - ar) + abs(zc - ac)

            # Shuffle move order for tie-breaking
            moves = list(_CL_MOVE_DELTAS)
            self.rng.shuffle(moves)

            for dr, dc in moves:
                nr, nc = zr + dr, zc + dc
                if (0 <= nr < gs and 0 <= nc < gs
                        and self.grid[nr, nc] in (_CL_GRASS, _CL_AGENT)):
                    dist = abs(nr - ar) + abs(nc - ac)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (nr, nc)

            # Update grid
            if best_pos != (zr, zc):
                if self.grid[zr, zc] == _CL_ZOMBIE:
                    self.grid[zr, zc] = _CL_GRASS
                self.grid[best_pos] = _CL_ZOMBIE

            new_positions.append(best_pos)

        self.zombie_positions = new_positions


def generate_crafter_lite(grid_size: int = 16, n_trajectories: int = 500,
                          n_steps: int = 20, n_zombies: int = 2,
                          seed: int = 42,
                          device: str | torch.device = "cpu"):
    """CrafterLite: mixed spatial + symbolic dynamics benchmark.

    A Crafter-inspired 16x16 grid with:
      - Tree growth (spatial, local coupling -- CML helps)
      - Zombie movement toward agent (spatial -- CML helps)
      - Harvesting / wall placement (symbolic, point changes -- CML wastes)

    This creates a spectrum between heat (100% spatial, CML dominates)
    and minigrid (0% coupling, MLP sufficient). CML should be BETTER
    than MLP (spatial components) but NOT perfect (symbolic components).

    Returns BenchmarkData where:
      X = [state_one_hot(7ch) + action_one_hot(1ch)]  (N, 8, H, W)
      Y = next_state_one_hot                           (N, 7, H, W)

    Action encoding: single channel with action_id / N_ACTIONS
    broadcast at the agent's position, 0 elsewhere.
    """
    device = torch.device(device)
    rng_master = np.random.RandomState(seed)

    states_list = []
    action_fields_list = []
    next_states_list = []

    env = ResourceGrid(grid_size=grid_size, n_zombies=n_zombies, seed=seed)

    for traj_i in range(n_trajectories):
        traj_seed = rng_master.randint(0, 2**31)
        env.reset(seed=traj_seed)
        action_rng = np.random.RandomState(traj_seed + 1)

        state = env._get_state()
        for t in range(n_steps):
            action = action_rng.randint(_CL_N_ACTIONS)

            # Build action field: 1 channel, agent position gets
            # normalized action value, rest is 0
            af = np.zeros((1, grid_size, grid_size), dtype=np.float32)
            ar, ac = env.agent_pos
            af[0, ar, ac] = (action + 1.0) / _CL_N_ACTIONS  # in (0, 1]

            next_state = env.step(action)

            states_list.append(state)
            action_fields_list.append(af)
            next_states_list.append(next_state)

            state = next_state

    # Stack all transitions
    states = np.stack(states_list)                # (N, 7, H, W)
    action_fields = np.stack(action_fields_list)  # (N, 1, H, W)
    next_states = np.stack(next_states_list)       # (N, 7, H, W)

    # Input: state + action field = 8 channels
    X = np.concatenate([states, action_fields], axis=1)  # (N, 8, H, W)
    Y = next_states  # (N, 7, H, W)

    # 70/15/15 split by trajectory (contiguous blocks)
    n_per_traj = n_steps
    n_train_traj = int(0.70 * n_trajectories)
    n_val_traj = int(0.15 * n_trajectories)

    n_train = n_train_traj * n_per_traj
    n_val = n_val_traj * n_per_traj

    X_tr = X[:n_train]
    Y_tr = Y[:n_train]
    X_v = X[n_train:n_train + n_val]
    Y_v = Y[n_train:n_train + n_val]
    X_te = X[n_train + n_val:]
    Y_te = Y[n_train + n_val:]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "crafter_lite",
        "loss_type": "cross_entropy",
        "metric": "accuracy",
        "in_channels": _CL_N_TYPES + 1,  # 8 (7 state + 1 action)
        "out_channels": _CL_N_TYPES,     # 7 (next state one-hot)
        "grid_size": grid_size,
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "n_zombies": n_zombies,
        "n_cell_types": _CL_N_TYPES,
        "n_actions": _CL_N_ACTIONS,
        "tree_growth_rate": _CL_TREE_GROWTH_RATE,
        "action_conditioned": True,
        "dynamics_mix": {
            "spatial": ["tree_growth", "zombie_movement"],
            "symbolic": ["harvesting", "wall_placement"],
        },
        "note": "Mixed spatial+symbolic: CML helps partially but not fully",
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  MINIGRID — Negative control (no inter-cell physics)
# ═══════════════════════════════════════════════════════════════════════════════

# MiniGrid-like cell types
_MG_EMPTY = 0
_MG_WALL  = 1
_MG_GOAL  = 2
_MG_AGENT = 3
_MG_N_CELL_TYPES = 4

# Actions: 0=up, 1=right, 2=down, 3=left
_MG_N_ACTIONS = 4
_MG_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class _MiniGridNav:
    """Minimal 8x8 grid navigator (no external dependency).

    Layout: border walls, a single goal at a random interior cell,
    agent starts at a random empty interior cell.  No obstacles
    beyond the border — matches MiniGrid-Empty-8x8.
    """

    def __init__(self, grid_size: int = 8, rng: np.random.Generator | None = None):
        self.H = self.W = grid_size
        self.rng = rng or np.random.default_rng()
        self.grid = None          # (H, W) int array of cell types
        self.agent_r = 0
        self.agent_c = 0
        self.goal_r = 0
        self.goal_c = 0

    def reset(self) -> np.ndarray:
        """Reset environment, return one-hot state (C, H, W)."""
        g = np.full((self.H, self.W), _MG_EMPTY, dtype=np.int32)
        # Border walls
        g[0, :] = _MG_WALL; g[-1, :] = _MG_WALL
        g[:, 0] = _MG_WALL; g[:, -1] = _MG_WALL

        # Place goal at random interior cell
        interior = [(r, c) for r in range(1, self.H - 1)
                    for c in range(1, self.W - 1)]
        idx = self.rng.integers(len(interior))
        self.goal_r, self.goal_c = interior[idx]
        g[self.goal_r, self.goal_c] = _MG_GOAL

        # Place agent at a different random interior cell
        interior.pop(idx)
        idx2 = self.rng.integers(len(interior))
        self.agent_r, self.agent_c = interior[idx2]
        g[self.agent_r, self.agent_c] = _MG_AGENT

        self.grid = g
        return self._observe()

    def step(self, action: int):
        """Take action, return (next_state, reward, done)."""
        dr, dc = _MG_DELTAS[action]
        nr, nc = self.agent_r + dr, self.agent_c + dc
        # Move only if target cell is not a wall
        if self.grid[nr, nc] != _MG_WALL:
            self.grid[self.agent_r, self.agent_c] = _MG_EMPTY
            self.agent_r, self.agent_c = nr, nc
            reached_goal = (nr == self.goal_r and nc == self.goal_c)
            self.grid[nr, nc] = _MG_AGENT
            if reached_goal:
                return self._observe(), 1.0, True
        return self._observe(), 0.0, False

    def _observe(self) -> np.ndarray:
        """Return one-hot encoded grid: (_MG_N_CELL_TYPES, H, W)."""
        obs = np.zeros((_MG_N_CELL_TYPES, self.H, self.W), dtype=np.float32)
        for t in range(_MG_N_CELL_TYPES):
            obs[t] = (self.grid == t).astype(np.float32)
        return obs


def generate_minigrid(grid_size: int = 8, n_trajectories: int = 500,
                      n_transitions: int | None = None,
                      episode_length: int = 30,
                      seed: int = 42,
                      device: str | torch.device = "cpu"):
    """MiniGrid navigation benchmark — NEGATIVE CONTROL.

    A simple 8x8 bordered grid with a single agent and goal.
    Only the agent cell changes per step; no inter-cell physics
    coupling.  CML's spatial convolution bias wastes parameters here.
    An MLP should match or beat CML.

    Returns BenchmarkData where:
      X  = [state_one_hot + action_field]  (N, C_state + C_action, H, W)
      Y  = next state one-hot              (N, C_state, H, W)

    C_state = 4 (empty, wall, goal, agent)
    C_action = 4 (one-hot action at agent position)
    """
    if n_transitions is None:
        n_transitions = n_trajectories * 10

    device = torch.device(device)
    rng = np.random.default_rng(seed)

    states, action_fields, next_states = [], [], []

    env = _MiniGridNav(grid_size=grid_size, rng=rng)
    collected = 0

    while collected < n_transitions:
        state = env.reset()
        for _ in range(episode_length):
            action = int(rng.integers(_MG_N_ACTIONS))

            # Action field: one-hot at agent position
            af = np.zeros((_MG_N_ACTIONS, grid_size, grid_size), dtype=np.float32)
            af[action, env.agent_r, env.agent_c] = 1.0

            ns, _, done = env.step(action)

            states.append(state)
            action_fields.append(af)
            next_states.append(ns)
            collected += 1

            if collected >= n_transitions or done:
                break
            state = ns

    states = np.stack(states[:n_transitions])           # (N, 4, H, W)
    action_fields = np.stack(action_fields[:n_transitions])  # (N, 4, H, W)
    next_states = np.stack(next_states[:n_transitions])  # (N, 4, H, W)

    X = np.concatenate([states, action_fields], axis=1)  # (N, 8, H, W)
    Y = next_states                                       # (N, 4, H, W)

    # 70/15/15 split
    n = n_transitions
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_v,  Y_v  = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_te, Y_te = X[n_train + n_val:], Y[n_train + n_val:]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    in_ch = _MG_N_CELL_TYPES + _MG_N_ACTIONS   # 8
    out_ch = _MG_N_CELL_TYPES                    # 4

    meta = {
        "name": "minigrid",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": in_ch,
        "out_channels": out_ch,
        "grid_size": grid_size,
        "spatial_dims": "2d",
        "n_transitions": n_transitions,
        "n_actions": _MG_N_ACTIONS,
        "n_cell_types": _MG_N_CELL_TYPES,
        "action_conditioned": True,
        "note": "Negative control: sparse discrete transitions, no inter-cell physics coupling",
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  ATARI PONG — Self-contained Pong (local ball/paddle physics)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_atari_pong(grid_size: int = 16, n_trajectories: int = 500,
                        n_transitions: int | None = None,
                        episode_length: int = 50,
                        seed: int = 42,
                        device: str | torch.device = "cpu"):
    """Atari Pong benchmark — local 2D physics for world model testing.

    Self-contained Pong: ball bounces off walls/paddles, agent controls
    left paddle (up/down/stay), right paddle tracks ball (AI).

    Great for CML: ball trajectory is LOCAL and predictable,
    paddle collisions are local spatial interactions.

    Returns BenchmarkData where:
      X = [state_one_hot(4ch) + action_broadcast(1ch)]  (N, 5, H, W)
      Y = next_state_one_hot                             (N, 4, H, W)

    Action encoding: single channel broadcast with (action+1)/N_ACTIONS.
    Grid: 16x32 by default (H x W).
    """
    from wmca.envs.atari_pong import PongEnv, _PONG_N_CHANNELS, _PONG_N_ACTIONS

    if n_transitions is None:
        n_transitions = n_trajectories * episode_length

    device = torch.device(device)
    rng_master = np.random.RandomState(seed)

    # Pong uses non-square grid: H=grid_size, W=grid_size*2
    grid_h = grid_size
    grid_w = grid_size * 2

    states, action_fields, next_states = [], [], []

    env = PongEnv(grid_h=grid_h, grid_w=grid_w, seed=seed)
    collected = 0

    while collected < n_transitions:
        traj_seed = rng_master.randint(0, 2**31)
        state = env.reset(seed=traj_seed)
        action_rng = np.random.RandomState(traj_seed + 1)

        for _ in range(episode_length):
            action = action_rng.randint(_PONG_N_ACTIONS)

            # Action field: single channel, broadcast normalized action
            af = np.full((1, grid_h, grid_w), (action + 1.0) / _PONG_N_ACTIONS,
                         dtype=np.float32)

            ns = env.step(action)

            states.append(state)
            action_fields.append(af)
            next_states.append(ns)
            collected += 1

            if collected >= n_transitions:
                break
            state = ns

    states = np.stack(states[:n_transitions])
    action_fields = np.stack(action_fields[:n_transitions])
    next_states = np.stack(next_states[:n_transitions])

    X = np.concatenate([states, action_fields], axis=1)  # (N, 5, H, W)
    Y = next_states                                       # (N, 4, H, W)

    # 70/15/15 split
    n = n_transitions
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_v,  Y_v  = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_te, Y_te = X[n_train + n_val:], Y[n_train + n_val:]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    in_ch = _PONG_N_CHANNELS + 1   # 5 (4 state + 1 action)
    out_ch = _PONG_N_CHANNELS      # 4

    meta = {
        "name": "atari_pong",
        "loss_type": "cross_entropy",
        "metric": "accuracy",
        "in_channels": in_ch,
        "out_channels": out_ch,
        "grid_size": grid_size,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "n_transitions": n_transitions,
        "n_actions": _PONG_N_ACTIONS,
        "n_channels": _PONG_N_CHANNELS,
        "action_conditioned": True,
        "non_square": True,
        "note": "Atari Pong: local ball physics + paddle collisions, CML should excel",
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  ATARI BREAKOUT — Self-contained Breakout (brick destruction + ball physics)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_atari_breakout(grid_size: int = 16, n_trajectories: int = 500,
                            n_transitions: int | None = None,
                            episode_length: int = 50,
                            seed: int = 42,
                            device: str | torch.device = "cpu"):
    """Atari Breakout benchmark — brick destruction + local ball physics.

    Self-contained Breakout: ball bounces, bricks disappear on hit,
    paddle moves left/right/stay.

    Even better for CML than Pong: bricks breaking = local state change,
    ball trajectory = local physics, paddle = agent-controlled.

    Returns BenchmarkData where:
      X = [state_one_hot(4ch) + action_broadcast(1ch)]  (N, 5, H, W)
      Y = next_state_one_hot                             (N, 4, H, W)

    Action encoding: single channel broadcast with (action+1)/N_ACTIONS.
    Grid: 20x16 by default (H x W).
    """
    from wmca.envs.atari_pong import BreakoutEnv, _BK_N_CHANNELS, _BK_N_ACTIONS

    if n_transitions is None:
        n_transitions = n_trajectories * episode_length

    device = torch.device(device)
    rng_master = np.random.RandomState(seed)

    # Breakout uses non-square grid: taller than wide
    grid_h = max(grid_size, 20)  # need room for bricks + play area
    grid_w = grid_size

    states, action_fields, next_states = [], [], []

    env = BreakoutEnv(grid_h=grid_h, grid_w=grid_w, seed=seed)
    collected = 0

    while collected < n_transitions:
        traj_seed = rng_master.randint(0, 2**31)
        state = env.reset(seed=traj_seed)
        action_rng = np.random.RandomState(traj_seed + 1)

        for _ in range(episode_length):
            action = action_rng.randint(_BK_N_ACTIONS)

            # Action field: single channel, broadcast normalized action
            af = np.full((1, grid_h, grid_w), (action + 1.0) / _BK_N_ACTIONS,
                         dtype=np.float32)

            ns = env.step(action)

            states.append(state)
            action_fields.append(af)
            next_states.append(ns)
            collected += 1

            if collected >= n_transitions:
                break
            state = ns

    states = np.stack(states[:n_transitions])
    action_fields = np.stack(action_fields[:n_transitions])
    next_states = np.stack(next_states[:n_transitions])

    X = np.concatenate([states, action_fields], axis=1)  # (N, 5, H, W)
    Y = next_states                                       # (N, 4, H, W)

    # 70/15/15 split
    n = n_transitions
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_v,  Y_v  = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_te, Y_te = X[n_train + n_val:], Y[n_train + n_val:]

    data = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    in_ch = _BK_N_CHANNELS + 1   # 5 (4 state + 1 action)
    out_ch = _BK_N_CHANNELS      # 4

    meta = {
        "name": "atari_breakout",
        "loss_type": "cross_entropy",
        "metric": "accuracy",
        "in_channels": in_ch,
        "out_channels": out_ch,
        "grid_size": grid_size,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "n_transitions": n_transitions,
        "n_actions": _BK_N_ACTIONS,
        "n_channels": _BK_N_CHANNELS,
        "action_conditioned": True,
        "non_square": True,
        "note": "Atari Breakout: brick destruction (local state change) + ball physics",
    }
    return BenchmarkData(*data, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def _lazy_gs_control(*args, **kwargs):
    """Lazy import to avoid circular dependency."""
    from wmca.envs.gray_scott_control import generate_gs_control
    return generate_gs_control(*args, **kwargs)


def _generate_autumn_disease(*args, **kwargs):
    """Lazy import for autumn disease spreading benchmark."""
    from wmca.envs.autumn import make_autumn_benchmark
    gen = make_autumn_benchmark("autumn_disease")
    return gen(*args, **kwargs)


def _generate_autumn_gravity(*args, **kwargs):
    """Lazy import for autumn gravity benchmark."""
    from wmca.envs.autumn import make_autumn_benchmark
    gen = make_autumn_benchmark("autumn_gravity")
    return gen(*args, **kwargs)


def _generate_autumn_water(*args, **kwargs):
    """Lazy import for autumn water flow benchmark."""
    from wmca.envs.autumn import make_autumn_benchmark
    gen = make_autumn_benchmark("autumn_water")
    return gen(*args, **kwargs)


BENCHMARKS = {
    "heat": generate_heat,
    "gol": generate_gol,
    "ks": generate_ks,
    "gray_scott": generate_gray_scott,
    "rule110": generate_rule110,
    "wireworld": generate_wireworld,
    "grid_world": generate_grid_world,
    "heat_control": generate_heat_control,
    "gs_control": _lazy_gs_control,
    "dmcontrol": generate_dmcontrol,
    "crafter_lite": generate_crafter_lite,
    "minigrid": generate_minigrid,
    "autumn_disease": _generate_autumn_disease,
    "autumn_gravity": _generate_autumn_gravity,
    "autumn_water": _generate_autumn_water,
    "atari_pong": generate_atari_pong,
    "atari_breakout": generate_atari_breakout,
}
