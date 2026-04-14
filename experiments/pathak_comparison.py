"""Direct comparison with Pathak et al. 2018 (PRL 120, 024102).

"Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data"

KS equation: du/dt = -u*u_x - u_xx - u_xxxx
L=22, N=64, periodic BCs, spectral method

CORRECTED: Pathak's dt=0.25 is in TIME UNITS (not Lyapunov times).
VPT is REPORTED in Lyapunov times: VPT_lyap = n_steps * dt_save / tau_lyap

Usage:
    cd ~/wmca && PYTHONPATH=src python3 experiments/pathak_comparison.py --no-wandb
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wmca.utils import pick_device
from wmca.modules.paralesn import ParalESNLayer

def _get_plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42

# =============================================================================
# KS Parameters -- Matching Pathak et al. 2018
# =============================================================================
KS_L = 22.0
KS_N = 64
KS_DX = KS_L / KS_N
LAMBDA_MAX = 0.047             # max Lyapunov exponent (1/time_unit)
TAU_LYAP = 1.0 / LAMBDA_MAX   # ~21.28 time units

# KEY: Pathak saves data every dt=0.25 TIME UNITS (not Lyapunov times!)
DT_SAVE = 0.25                 # time units between saved snapshots

# Derived
DT_LYAP_PER_STEP = DT_SAVE / TAU_LYAP  # ~0.01175 Lyapunov times per step

WARMUP_TIME = 500.0            # time units of warmup

# Training: 500 Lyapunov times of data
TRAIN_LYAP = 500
N_TRAIN = int(TRAIN_LYAP * TAU_LYAP / DT_SAVE)  # ~42553 snapshots

# VPT evaluation
VPT_THRESHOLD = 0.4
N_TEST_WINDOWS = 100
MAX_ROLLOUT_STEPS = 800        # ~9.4 Lyapunov times (past Pathak's 8.2)

# Training hyperparams
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 256

# ESN
ESN_HIDDEN = 256
ESN_SPECTRAL_RADIUS = 0.9
ESN_INPUT_SCALE = 0.1
ESN_RIDGE_ALPHA = 1e-4
ESN_WASHOUT = 500

# Data cache
KS_DATA_CACHE = Path("/tmp/ks_data.npz")


# ParalESN config
@dataclass
class ParalESNCfg:
    hidden_size: int = 256
    rho_min: float = 0.95
    rho_max: float = 0.999
    theta_min: float = 0.0
    theta_max: float = 6.2832
    tau: float = 0.5
    mix_kernel_size: int = 5
    omega_in: float = 1.0
    omega_b: float = 0.1
    use_fft: bool = True


# =============================================================================
# KS Solver
# =============================================================================

def _make_ks_rhs(N: int, L: float):
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    k[N // 2] = 0.0
    ik = 1j * k
    lin_k = k ** 2 - k ** 4

    def rhs(t, u):
        u_hat = np.fft.fft(u)
        nl = np.fft.ifft(-0.5 * ik * np.fft.fft(u ** 2)).real
        return nl + np.fft.ifft(lin_k * u_hat).real

    return rhs


def generate_ks_trajectory(n_save: int, warmup_time: float = WARMUP_TIME,
                           seed: int = SEED) -> np.ndarray:
    from scipy.integrate import solve_ivp

    rng = np.random.RandomState(seed)
    rhs = _make_ks_rhs(KS_N, KS_L)

    x = np.linspace(0, KS_L, KS_N, endpoint=False)
    u0 = np.cos(x * 2 * np.pi / KS_L) * (1 + np.sin(x * 2 * np.pi / KS_L))
    u0 += rng.randn(KS_N) * 0.01

    print(f"  Warmup: {warmup_time:.0f} time units ...")
    t0 = time.time()
    sol = solve_ivp(rhs, [0, warmup_time], u0, method="LSODA",
                    rtol=1e-6, atol=1e-8, max_step=0.5)
    u_w = sol.y[:, -1]
    print(f"    Done in {time.time() - t0:.1f}s")

    print(f"  Collecting {n_save} snapshots ({n_save * DT_SAVE:.0f} time units) ...")
    chunk_size = 10000
    trajectory = np.zeros((n_save, KS_N), dtype=np.float64)
    u_curr = u_w.copy()
    collected = 0

    while collected < n_save:
        n_chunk = min(chunk_size, n_save - collected)
        t_eval = np.arange(1, n_chunk + 1) * DT_SAVE
        sol = solve_ivp(rhs, [0, n_chunk * DT_SAVE], u_curr, method="LSODA",
                        t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.5)
        if not sol.success:
            sol = solve_ivp(rhs, [0, n_chunk * DT_SAVE], u_curr, method="Radau",
                            t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.5)
        trajectory[collected:collected + n_chunk] = sol.y.T
        u_curr = sol.y[:, -1].copy()
        collected += n_chunk
        elapsed = time.time() - t0
        print(f"    {collected}/{n_save} ({elapsed:.0f}s)")

    return trajectory


# =============================================================================
# ESN
# =============================================================================

class SimpleESN:
    def __init__(self, input_dim, hidden_dim=ESN_HIDDEN,
                 spectral_radius=ESN_SPECTRAL_RADIUS,
                 input_scale=ESN_INPUT_SCALE,
                 ridge_alpha=ESN_RIDGE_ALPHA,
                 leak_rate=0.3, seed=SEED):
        rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.leak_rate = leak_rate

        self.W_in = np.zeros((hidden_dim, input_dim), dtype=np.float64)
        for i in range(hidden_dim):
            nc = max(1, int(0.1 * input_dim))
            idx = rng.choice(input_dim, nc, replace=False)
            self.W_in[i, idx] = rng.randn(nc) * input_scale

        density = min(10.0 / hidden_dim, 1.0)
        W = rng.randn(hidden_dim, hidden_dim) * (rng.rand(hidden_dim, hidden_dim) < density)
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_res = W * (spectral_radius / max(sr, 1e-10))

        self.bias = rng.randn(hidden_dim) * 0.1
        self.ridge_alpha = ridge_alpha
        self.W_out = None
        self.state = np.zeros(hidden_dim)

    def _step(self, u):
        pre = self.W_in @ u + self.W_res @ self.state + self.bias
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre)
        return self.state.copy()

    def _feat(self, s, u):
        return np.concatenate([s, s ** 2, u])

    def train(self, data, washout=ESN_WASHOUT):
        T = len(data)
        fd = 2 * self.hidden_dim + self.input_dim
        states = np.zeros((T, fd))
        self.state[:] = 0
        for t in range(T):
            states[t] = self._feat(self._step(data[t]), data[t])
            if t % 10000 == 0 and t > 0:
                print(f"      Driving: {t}/{T}")

        X, Y = states[washout:-1], data[washout + 1:]
        print(f"    Ridge: X={X.shape}")
        self.W_out = np.linalg.solve(X.T @ X + self.ridge_alpha * np.eye(fd), X.T @ Y).T
        print(f"    Train MSE: {np.mean((X @ self.W_out.T - Y) ** 2):.6e}")

    def predict_step(self, u):
        return self.W_out @ self._feat(self._step(u), u)

    def warm_state(self, data):
        self.state[:] = 0
        for t in range(len(data)):
            self._step(data[t])


# =============================================================================
# 1D Neural Models
# =============================================================================

class CML1D(nn.Module):
    """1D CML with circular padding."""
    def __init__(self, steps=15, r=3.90, eps=0.3, beta=0.15, seed=42):
        super().__init__()
        self.steps = steps
        self.register_buffer("r", torch.tensor(r))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        rng = torch.Generator().manual_seed(seed)
        K = torch.rand(1, 1, 3, generator=rng).abs()
        self.register_buffer("K_local", K / K.sum())

    def forward(self, drive):
        grid = drive
        for _ in range(self.steps):
            mapped = self.r * grid * (1.0 - grid)
            mapped_p = F.pad(mapped, (1, 1), mode="circular")
            local = F.conv1d(mapped_p, self.K_local)
            physics = (1 - self.eps) * mapped + self.eps * local
            grid = (1 - self.beta) * physics + self.beta * drive
        return grid.clamp(1e-4, 1 - 1e-4)


class ResidualCorrectionWM1D(nn.Module):
    """CML1D base + learned NCA correction."""
    def __init__(self, hidden_ch=16, cml_steps=15):
        super().__init__()
        self.cml = CML1D(steps=cml_steps)
        self.nca = nn.Sequential(
            nn.Conv1d(2, hidden_ch, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_ch, 1, 1),
        )

    def forward(self, x):
        cml_out = self.cml(x)
        cat = F.pad(torch.cat([x, cml_out], dim=1), (1, 1), mode="circular")
        return torch.clamp(cml_out + self.nca(cat), 0, 1)

    def param_count(self):
        return {"trained": sum(p.numel() for p in self.parameters()),
                "frozen": sum(b.numel() for b in self.cml.buffers())}


class PureNCA1D(nn.Module):
    def __init__(self, hidden_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_ch, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(F.pad(x, (1, 1), mode="circular"))

    def param_count(self):
        return {"trained": sum(p.numel() for p in self.parameters()), "frozen": 0}


class Conv1DBaseline(nn.Module):
    """3-layer Conv1D with circular padding and skip connection."""
    def __init__(self, hidden_ch=64):
        super().__init__()
        self.c1 = nn.Conv1d(1, hidden_ch, 5, padding=0)
        self.c2 = nn.Conv1d(hidden_ch, hidden_ch, 5, padding=0)
        self.c3 = nn.Conv1d(hidden_ch, 1, 5, padding=0)

    def forward(self, x):
        h = F.relu(self.c1(F.pad(x, (2, 2), mode="circular")))
        h = F.relu(self.c2(F.pad(h, (2, 2), mode="circular")))
        return x + self.c3(F.pad(h, (2, 2), mode="circular"))  # skip connection

    def param_count(self):
        return {"trained": sum(p.numel() for p in self.parameters()), "frozen": 0}


# =============================================================================
# ParalESN + Ridge (pure reservoir baseline — closest to Pathak's ESN)
# =============================================================================

class ParalESNRidge:
    """ParalESN temporal features → Ridge regression readout.

    No CML, no NCA — just diagonal recurrence + linear readout.
    Closest analog to Pathak's ESN but with parallel (FFT) recurrence.
    """
    def __init__(self, input_size=KS_N, hidden_size=256, ridge_alpha=1e-4,
                 washout=500, seed=SEED):
        cfg = ParalESNCfg(hidden_size=hidden_size)
        rng = torch.Generator().manual_seed(seed)
        self.layer = ParalESNLayer(cfg, layer_idx=0, input_size=input_size, rng=rng)
        self.layer.eval()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ridge_alpha = ridge_alpha
        self.washout = washout
        self.W_out = None
        self.h_state = None  # complex hidden state for token-by-token rollout

    def _to_device(self, device):
        self.layer = self.layer.to(device)
        return self

    def _extract_features(self, data_tensor):
        """Run full sequence through ParalESN, return mixed features (B, T, N_h)."""
        with torch.no_grad():
            h, _z = self.layer.forward(data_tensor)
            # Use _mix(h) to bypass zero-init out_proj
            feats = self.layer._mix(h)
        return feats

    def train(self, data_norm, device=None):
        """data_norm: (T, N) numpy float32 normalized KS trajectory."""
        if device is None:
            device = torch.device("cpu")
        self._to_device(device)

        T = len(data_norm)
        print(f"    ParalESN-Ridge: processing {T} timesteps ...")

        # Process full sequence: (1, T, N)
        x_tensor = torch.from_numpy(data_norm).float().unsqueeze(0).to(device)
        feats = self._extract_features(x_tensor)  # (1, T, hidden_size)
        feats_np = feats.squeeze(0).cpu().numpy()  # (T, hidden_size)

        # Augment features: [mix_features, mix_features^2, input]
        feats_aug = np.concatenate([feats_np, feats_np ** 2, data_norm], axis=1)
        fd = feats_aug.shape[1]

        # Ridge regression: predict next state from current features
        X = feats_aug[self.washout:-1]
        Y = data_norm[self.washout + 1:]
        print(f"    Ridge: X={X.shape}, Y={Y.shape}")
        self.W_out = np.linalg.solve(
            X.T @ X + self.ridge_alpha * np.eye(fd), X.T @ Y
        ).T
        train_mse = np.mean((X @ self.W_out.T - Y) ** 2)
        print(f"    Train MSE: {train_mse:.6e}")

    def warm_state(self, data_norm, device=None):
        """Warm up hidden state by driving with data sequence."""
        if device is None:
            device = torch.device("cpu")
        self._to_device(device)

        x_tensor = torch.from_numpy(data_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            h, _ = self.layer.forward(x_tensor)
            self.h_state = h[0, -1]  # (hidden_size,) complex

    def predict_step(self, u_norm, device=None):
        """Single-step prediction using forward_token with hidden state."""
        if device is None:
            device = torch.device("cpu")

        x = torch.from_numpy(u_norm).float().unsqueeze(0).to(device)  # (1, N)
        with torch.no_grad():
            self.h_state, _z = self.layer.forward_token(x, self.h_state)
            mix_feat = self.layer._mix_single(self.h_state)  # (1, hidden_size)
        mix_np = mix_feat.cpu().numpy().reshape(-1)
        feat = np.concatenate([mix_np, mix_np ** 2, u_norm])
        return (self.W_out @ feat).astype(np.float32)


# =============================================================================
# ParalESN + ResCor(D) — temporal backbone feeding CML+NCA
# =============================================================================

class ParalESNResCor(nn.Module):
    """ParalESN temporal memory → CML drive → NCA correction.

    ParalESN provides temporal recurrence (hidden state across time).
    Its mixed output drives the CML-1D, which is then corrected by NCA.
    """
    def __init__(self, input_size=KS_N, hidden_size=256, hidden_ch=16,
                 cml_steps=15, seed=SEED):
        super().__init__()
        self.cfg = ParalESNCfg(hidden_size=hidden_size)
        rng = torch.Generator().manual_seed(seed)
        self.esn_layer = ParalESNLayer(self.cfg, layer_idx=0,
                                       input_size=input_size, rng=rng)
        # Freeze ESN parameters (reservoir is fixed)
        for p in self.esn_layer.parameters():
            p.requires_grad = False

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Learned linear projection: ESN features → drive signal
        self.drive_proj = nn.Linear(hidden_size, input_size)

        # CML + NCA correction (same as ResCor(D))
        self.cml = CML1D(steps=cml_steps)
        self.nca = nn.Sequential(
            nn.Conv1d(2, hidden_ch, 3, padding=0),
            nn.ReLU(),
            nn.Conv1d(hidden_ch, 1, 1),
        )

        # Hidden state for token-by-token rollout
        self.h_state = None

    def _esn_to_drive(self, esn_mix):
        """Convert ESN mixed output to CML drive in [0,1].

        esn_mix: (..., hidden_size) from _mix (tanh, range [-1,1])
        Returns: (..., 1, input_size) drive signal in [0,1]
        """
        proj = self.drive_proj(esn_mix)  # (..., input_size)
        drive = (proj + 1.0) / 2.0      # shift tanh range to [0,1]
        return drive.unsqueeze(-2)       # (..., 1, input_size)

    def forward(self, x):
        """x: (B, 1, N) single-step input (for training compatibility).

        For training, we don't use temporal recurrence — just feed single
        frames through the ESN input projection + mix, then CML + NCA.
        This trains the drive_proj and NCA weights.
        """
        B, _, N = x.shape
        # Direct input projection through ESN (no recurrence for single-step)
        x_flat = x.squeeze(1)  # (B, N)
        with torch.no_grad():
            projected = self.esn_layer._input_projection(x_flat)
            drive_complex = self.cfg.tau * (projected + self.esn_layer.b)
            # For single-step, h = drive (no previous state)
            h = drive_complex
            # Mix: need (B, 1, hidden_size) for _mix
            h_seq = h.unsqueeze(1)
            mix = self.esn_layer._mix(h_seq)  # (B, 1, hidden_size)
        mix = mix.squeeze(1).float()  # (B, hidden_size) — detached from ESN

        drive = self._esn_to_drive(mix)  # (B, 1, N)
        cml_out = self.cml(drive)

        cat = F.pad(torch.cat([drive, cml_out], dim=1), (1, 1), mode="circular")
        return torch.clamp(cml_out + self.nca(cat), 0, 1)

    def forward_sequence(self, x_seq, device=None):
        """Process full sequence with temporal recurrence for VPT eval.

        x_seq: (T, N) numpy array, normalized
        Returns: sets self.h_state for subsequent predict_step calls
        """
        if device is None:
            device = next(self.parameters()).device
        x_t = torch.from_numpy(x_seq).float().unsqueeze(0).to(device)  # (1, T, N)
        with torch.no_grad():
            h, _ = self.esn_layer.forward(x_t)
            self.h_state = h[0, -1]  # (hidden_size,) complex

    def predict_step_recurrent(self, u_norm, device=None):
        """Single recurrent step: update hidden, project → CML → NCA."""
        if device is None:
            device = next(self.parameters()).device
        x = torch.from_numpy(u_norm).float().unsqueeze(0).to(device)  # (1, N)
        with torch.no_grad():
            self.h_state, _ = self.esn_layer.forward_token(x, self.h_state)
            mix = self.esn_layer._mix_single(self.h_state)  # (1, hidden_size)
        mix = mix.float()

        drive = self._esn_to_drive(mix)  # (1, 1, N)
        with torch.no_grad():
            cml_out = self.cml(drive)
            cat = F.pad(torch.cat([drive, cml_out], dim=1), (1, 1), mode="circular")
            out = torch.clamp(cml_out + self.nca(cat), 0, 1)
        return out.cpu().numpy().reshape(self.input_size)

    def param_count(self):
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_esn = sum(b.numel() for b in self.esn_layer.buffers())
        frozen_cml = sum(b.numel() for b in self.cml.buffers())
        return {"trained": trained, "frozen": frozen_esn + frozen_cml}


# =============================================================================
# VPT
# =============================================================================

def nrmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2) / max(np.mean(true ** 2), 1e-12))

def compute_vpt(pred_seq, true_seq, threshold=VPT_THRESHOLD):
    for t in range(len(pred_seq)):
        if nrmse(pred_seq[t], true_seq[t]) > threshold:
            return t
    return len(pred_seq)


# =============================================================================
# Training
# =============================================================================

def train_model(model, X_tr, Y_tr, X_v, Y_v, epochs=EPOCHS, lr=LR,
                bs=BATCH_SIZE, device=None):
    if device is None: device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    crit = nn.MSELoss()

    Xt = torch.from_numpy(X_tr).float().to(device)
    Yt = torch.from_numpy(Y_tr).float().to(device)
    Xv = torch.from_numpy(X_v).float().to(device)
    Yv = torch.from_numpy(Y_v).float().to(device)

    best_val, best_st = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        tl, nb = 0., 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            loss = crit(model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vs, vn = 0., 0
            for vi in range(0, len(Xv), bs):
                vx, vy = Xv[vi:vi+bs], Yv[vi:vi+bs]
                vs += crit(model(vx), vy).item() * len(vx); vn += len(vx)
            vl = vs / vn

        if vl < best_val:
            best_val = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 25 == 0 or ep == 0:
            print(f"    Ep {ep+1:3d}/{epochs}  tr={tl/nb:.6e}  val={vl:.6e}  lr={opt.param_groups[0]['lr']:.1e}")

    model.load_state_dict(best_st); model.to(device)
    print(f"    Best val: {best_val:.6e}")
    del Xt, Yt, Xv, Yv; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return model


def train_multistep(model, data, val_data, steps=4, epochs=80, lr=3e-4,
                    bs=64, device=None):
    if device is None: device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)
    crit = nn.MSELoss()
    sl = steps + 1
    nt = len(data) - sl; nv = len(val_data) - sl
    best_val, best_st = float("inf"), None

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(nt)
        tl, nb = 0., 0
        for bi in range(0, min(len(perm), 8000), bs):
            idx = perm[bi:bi+bs]
            seqs = torch.from_numpy(np.stack([data[i:i+sl] for i in idx])).float().to(device)
            loss = torch.tensor(0., device=device)
            xc = seqs[:, 0].unsqueeze(1)
            for s in range(steps):
                pred = model(xc)
                loss = loss + crit(pred, seqs[:, s+1].unsqueeze(1))
                xc = pred.detach().clamp(0, 1)
            loss /= steps
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1
        sched.step()

        model.eval(); vs, vn = 0., 0
        with torch.no_grad():
            for vi in range(0, min(nv, 3000), bs):
                iv = np.arange(vi, min(vi+bs, nv))
                sv = torch.from_numpy(np.stack([val_data[i:i+sl] for i in iv])).float().to(device)
                xc = sv[:, 0].unsqueeze(1)
                vl = torch.tensor(0., device=device)
                for s in range(steps):
                    pred = model(xc)
                    vl += crit(pred, sv[:, s+1].unsqueeze(1))
                    xc = pred.clamp(0, 1)
                vs += (vl/steps).item() * len(iv); vn += len(iv)
        vl = vs / max(vn, 1)
        if vl < best_val:
            best_val = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"    [MS] Ep {ep+1:3d}/{epochs}  tr={tl/max(nb,1):.6e}  val={vl:.6e}")

    if best_st: model.load_state_dict(best_st); model.to(device)
    print(f"    [MS] Best val: {best_val:.6e}")
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return model


# =============================================================================
# Main
# =============================================================================

def run_experiment(args):
    device = pick_device()

    print("=" * 80)
    print("  PATHAK et al. 2018 COMPARISON -- KS VPT")
    print("=" * 80)
    print(f"  L={KS_L}, N={KS_N}, lambda_max={LAMBDA_MAX}, tau_Lyap={TAU_LYAP:.2f} tu")
    print(f"  dt_save={DT_SAVE} tu = {DT_LYAP_PER_STEP:.5f} Lyap/step")
    print(f"  VPT threshold={VPT_THRESHOLD}, {N_TEST_WINDOWS} windows, "
          f"max {MAX_ROLLOUT_STEPS} steps ({MAX_ROLLOUT_STEPS * DT_LYAP_PER_STEP:.1f} Lyap)")
    print(f"  Training: {TRAIN_LYAP} Lyap = {N_TRAIN} snapshots")

    # ----- Data -----
    print(f"\n{'=' * 80}\n  STEP 1: Generate KS data\n{'=' * 80}")
    total_needed = N_TRAIN + N_TEST_WINDOWS * (MAX_ROLLOUT_STEPS + 100) + 1000
    t0 = time.time()
    if KS_DATA_CACHE.exists():
        print(f"  Loading cached data from {KS_DATA_CACHE} ...")
        _cached = np.load(KS_DATA_CACHE)
        raw = _cached["raw"]
        if len(raw) < total_needed:
            print(f"  Cache too small ({len(raw)} < {total_needed}), regenerating ...")
            raw = generate_ks_trajectory(total_needed, seed=SEED)
            np.savez(KS_DATA_CACHE, raw=raw)
        else:
            raw = raw[:total_needed]
        print(f"  Loaded {len(raw)} snapshots")
    else:
        raw = generate_ks_trajectory(total_needed, seed=SEED)
        np.savez(KS_DATA_CACHE, raw=raw)
        print(f"  Cached data to {KS_DATA_CACHE}")
    print(f"  Range: [{raw.min():.3f}, {raw.max():.3f}], std={raw.std():.4f}")

    gmin, gmax = raw.min(), raw.max()
    gr = gmax - gmin + 1e-12
    traj = ((raw - gmin) / gr).astype(np.float32)
    raw = raw.astype(np.float64)

    train_norm = traj[:N_TRAIN]
    test_start = N_TRAIN

    # Pairs for 1-step training: (B, 1, N)
    X_all = train_norm[:-1].reshape(-1, 1, KS_N)
    Y_all = train_norm[1:].reshape(-1, 1, KS_N)
    nv = int(0.1 * len(X_all))
    X_tr, X_v = X_all[:-nv], X_all[-nv:]
    Y_tr, Y_v = Y_all[:-nv], Y_all[-nv:]
    val_seq = train_norm[N_TRAIN - nv - 10:]
    print(f"  Pairs: train={len(X_tr)}, val={len(X_v)}")

    # ----- VPT helpers -----
    def eval_vpt_nn(model, name):
        model.eval()
        dev = next(model.parameters()).device if list(model.parameters()) else device
        vpts = []
        for w in range(N_TEST_WINDOWS):
            st = test_start + w * (MAX_ROLLOUT_STEPS + 50)
            if st + MAX_ROLLOUT_STEPS + 1 > len(traj): break
            x = traj[st].copy()
            true_raw = raw[st+1:st+1+MAX_ROLLOUT_STEPS]
            preds = []
            with torch.no_grad():
                for t in range(MAX_ROLLOUT_STEPS):
                    xt = torch.from_numpy(x).float().reshape(1, 1, KS_N).to(dev)
                    out = model(xt).cpu().numpy().reshape(KS_N)
                    preds.append(out.copy())
                    x = np.clip(out, 0, 1).astype(np.float32)
            pred_raw = np.array(preds) * gr + gmin
            vpts.append(compute_vpt(pred_raw, true_raw))
        vpt_lyap = np.mean(vpts) * DT_LYAP_PER_STEP
        std_lyap = np.std(vpts) * DT_LYAP_PER_STEP
        med_lyap = np.median(vpts) * DT_LYAP_PER_STEP
        print(f"    {name}: {vpt_lyap:.2f} +/- {std_lyap:.2f} Lyap (med={med_lyap:.2f}, n={len(vpts)})")
        return vpt_lyap, vpts

    def eval_vpt_esn(esn, name):
        vpts = []
        for w in range(N_TEST_WINDOWS):
            st = test_start + w * (MAX_ROLLOUT_STEPS + 50)
            if st + MAX_ROLLOUT_STEPS + 1 > len(traj): break
            wl = min(500, st)
            esn.warm_state(traj[st-wl:st])
            u = traj[st].copy()
            true_raw = raw[st+1:st+1+MAX_ROLLOUT_STEPS]
            preds = []
            for t in range(MAX_ROLLOUT_STEPS):
                u = esn.predict_step(u)
                preds.append(u.copy())
            pred_raw = np.array(preds) * gr + gmin
            vpts.append(compute_vpt(pred_raw, true_raw))
        vpt_lyap = np.mean(vpts) * DT_LYAP_PER_STEP
        std_lyap = np.std(vpts) * DT_LYAP_PER_STEP
        med_lyap = np.median(vpts) * DT_LYAP_PER_STEP
        print(f"    {name}: {vpt_lyap:.2f} +/- {std_lyap:.2f} Lyap (med={med_lyap:.2f}, n={len(vpts)})")
        return vpt_lyap, vpts

    def eval_vpt_paralesn_ridge(pesn, name):
        """VPT eval for ParalESNRidge: warm state then token-by-token rollout."""
        vpts = []
        for w in range(N_TEST_WINDOWS):
            st = test_start + w * (MAX_ROLLOUT_STEPS + 50)
            if st + MAX_ROLLOUT_STEPS + 1 > len(traj): break
            wl = min(500, st)
            pesn.warm_state(traj[st-wl:st], device=device)
            u = traj[st].copy()
            true_raw = raw[st+1:st+1+MAX_ROLLOUT_STEPS]
            preds = []
            for t in range(MAX_ROLLOUT_STEPS):
                u = pesn.predict_step(u, device=device)
                preds.append(u.copy())
                u = np.clip(u, 0, 1).astype(np.float32)
            pred_raw = np.array(preds) * gr + gmin
            vpts.append(compute_vpt(pred_raw, true_raw))
        vpt_lyap = np.mean(vpts) * DT_LYAP_PER_STEP
        std_lyap = np.std(vpts) * DT_LYAP_PER_STEP
        med_lyap = np.median(vpts) * DT_LYAP_PER_STEP
        print(f"    {name}: {vpt_lyap:.2f} +/- {std_lyap:.2f} Lyap (med={med_lyap:.2f}, n={len(vpts)})")
        return vpt_lyap, vpts

    def eval_vpt_paralesn_rescor(model, name):
        """VPT eval for ParalESNResCor: warm ESN state then recurrent rollout."""
        model.eval()
        dev = next(model.parameters()).device
        vpts = []
        for w in range(N_TEST_WINDOWS):
            st = test_start + w * (MAX_ROLLOUT_STEPS + 50)
            if st + MAX_ROLLOUT_STEPS + 1 > len(traj): break
            wl = min(500, st)
            model.forward_sequence(traj[st-wl:st], device=dev)
            u = traj[st].copy()
            true_raw = raw[st+1:st+1+MAX_ROLLOUT_STEPS]
            preds = []
            for t in range(MAX_ROLLOUT_STEPS):
                out = model.predict_step_recurrent(u, device=dev)
                preds.append(out.copy())
                u = np.clip(out, 0, 1).astype(np.float32)
            pred_raw = np.array(preds) * gr + gmin
            vpts.append(compute_vpt(pred_raw, true_raw))
        vpt_lyap = np.mean(vpts) * DT_LYAP_PER_STEP
        std_lyap = np.std(vpts) * DT_LYAP_PER_STEP
        med_lyap = np.median(vpts) * DT_LYAP_PER_STEP
        print(f"    {name}: {vpt_lyap:.2f} +/- {std_lyap:.2f} Lyap (med={med_lyap:.2f}, n={len(vpts)})")
        return vpt_lyap, vpts

    results = {}

    # ----- ESN -----
    print(f"\n{'=' * 80}\n  MODEL 1: ESN-{ESN_HIDDEN}\n{'=' * 80}")
    t0 = time.time()
    esn = SimpleESN(KS_N, ESN_HIDDEN, ESN_SPECTRAL_RADIUS, ESN_INPUT_SCALE,
                    ESN_RIDGE_ALPHA, 0.3, SEED)
    esn.train(train_norm, ESN_WASHOUT)
    et = time.time() - t0
    fd = 2 * ESN_HIDDEN + KS_N
    ep = KS_N * fd
    results["Our ESN (256)"] = {"time": et, "params": ESN_HIDDEN*KS_N + ESN_HIDDEN**2 + ESN_HIDDEN + ep, "trained": ep}
    vpt, vl = eval_vpt_esn(esn, "ESN-256")
    results["Our ESN (256)"].update({"vpt_lyap": vpt, "vpt_list": vl})

    # ----- ResCor(D) -----
    print(f"\n{'=' * 80}\n  MODEL 2: ResCor(D)\n{'=' * 80}")
    t0 = time.time()
    rescor = ResidualCorrectionWM1D(hidden_ch=16, cml_steps=15)
    pc = rescor.param_count()
    print(f"  trained={pc['trained']}, frozen={pc['frozen']}")
    rescor = train_model(rescor, X_tr, Y_tr, X_v, Y_v, device=device)
    rescor = train_multistep(rescor, train_norm[:N_TRAIN-nv], val_seq, steps=4,
                             epochs=80, lr=3e-4, device=device)
    et = time.time() - t0
    vpt, vl = eval_vpt_nn(rescor, "ResCor(D)")
    results["ResCor(D)"] = {"vpt_lyap": vpt, "vpt_list": vl, "params": pc["trained"],
                            "trained": pc["trained"], "time": et}

    # ----- PureNCA -----
    print(f"\n{'=' * 80}\n  MODEL 3: PureNCA\n{'=' * 80}")
    t0 = time.time()
    pnca = PureNCA1D(hidden_ch=16)
    pc = pnca.param_count()
    print(f"  trained={pc['trained']}")
    pnca = train_model(pnca, X_tr, Y_tr, X_v, Y_v, device=device)
    pnca = train_multistep(pnca, train_norm[:N_TRAIN-nv], val_seq, steps=4,
                           epochs=80, lr=3e-4, device=device)
    et = time.time() - t0
    vpt, vl = eval_vpt_nn(pnca, "PureNCA")
    results["PureNCA"] = {"vpt_lyap": vpt, "vpt_list": vl, "params": pc["trained"],
                          "trained": pc["trained"], "time": et}

    # ----- Conv1D -----
    print(f"\n{'=' * 80}\n  MODEL 4: Conv1D\n{'=' * 80}")
    t0 = time.time()
    cnn = Conv1DBaseline(hidden_ch=64)
    pc = cnn.param_count()
    print(f"  trained={pc['trained']}")
    cnn = train_model(cnn, X_tr, Y_tr, X_v, Y_v, device=device)
    cnn = train_multistep(cnn, train_norm[:N_TRAIN-nv], val_seq, steps=4,
                          epochs=80, lr=3e-4, device=device)
    et = time.time() - t0
    vpt, vl = eval_vpt_nn(cnn, "Conv1D")
    results["Conv1D"] = {"vpt_lyap": vpt, "vpt_list": vl, "params": pc["trained"],
                         "trained": pc["trained"], "time": et}

    # ----- ParalESN + Ridge -----
    print(f"\n{'=' * 80}\n  MODEL 5: ParalESN + Ridge\n{'=' * 80}")
    t0 = time.time()
    pesn_ridge = ParalESNRidge(input_size=KS_N, hidden_size=256,
                               ridge_alpha=ESN_RIDGE_ALPHA, washout=ESN_WASHOUT,
                               seed=SEED)
    pesn_ridge.train(train_norm, device=device)
    et = time.time() - t0
    n_esn_buf = sum(b.numel() for b in pesn_ridge.layer.buffers())
    fd_pesn = 2 * 256 + KS_N
    n_readout = KS_N * fd_pesn
    results["ParalESN+Ridge"] = {"time": et, "params": n_esn_buf + n_readout,
                                  "trained": n_readout}
    vpt, vl = eval_vpt_paralesn_ridge(pesn_ridge, "ParalESN+Ridge")
    results["ParalESN+Ridge"].update({"vpt_lyap": vpt, "vpt_list": vl})

    # ----- ParalESN + ResCor(D) -----
    print(f"\n{'=' * 80}\n  MODEL 6: ParalESN + ResCor(D)\n{'=' * 80}")
    t0 = time.time()
    pesn_rescor = ParalESNResCor(input_size=KS_N, hidden_size=256,
                                  hidden_ch=16, cml_steps=15, seed=SEED)
    pc = pesn_rescor.param_count()
    print(f"  trained={pc['trained']}, frozen={pc['frozen']}")
    pesn_rescor = train_model(pesn_rescor, X_tr, Y_tr, X_v, Y_v, device=device)
    pesn_rescor = train_multistep(pesn_rescor, train_norm[:N_TRAIN-nv], val_seq,
                                   steps=4, epochs=80, lr=3e-4, device=device)
    et = time.time() - t0
    vpt, vl = eval_vpt_paralesn_rescor(pesn_rescor, "ParalESN+ResCor(D)")
    results["ParalESN+ResCor(D)"] = {"vpt_lyap": vpt, "vpt_list": vl,
                                      "params": pc["trained"] + pc["frozen"],
                                      "trained": pc["trained"], "time": et}

    # ----- Results Table -----
    print(f"\n\n{'=' * 100}")
    print(f"  PATHAK et al. 2018 COMPARISON -- RESULTS")
    print(f"{'=' * 100}")
    print(f"  KS: L={KS_L}, N={KS_N}, dt={DT_SAVE} tu, threshold={VPT_THRESHOLD}")
    print(f"  Training: {TRAIN_LYAP} Lyap times, {N_TEST_WINDOWS} test windows\n")
    print(f"  {'Model':<25s}  {'VPT (Lyap)':<18s}  {'Params':<12s}  {'Trained':<12s}  {'Time'}")
    print("  " + "-" * 90)
    print(f"  {'Pathak ESN (2018)':<25s}  {'~8.2':<18s}  {'~36M+rout':<12s}  {'readout':<12s}  --")

    for name, r in results.items():
        std = np.std(r['vpt_list']) * DT_LYAP_PER_STEP
        vstr = f"{r['vpt_lyap']:.2f} +/- {std:.2f}"
        print(f"  {name:<25s}  {vstr:<18s}  {r['params']:>10,}  {r['trained']:>10,}  {r['time']:.0f}s")

    print(f"\n{'=' * 100}")

    # ----- Plots -----
    print("\n  Generating plots ...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt = _get_plt()

    # VPT boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    vd = [np.array(results[n]["vpt_list"]) * DT_LYAP_PER_STEP for n in names]
    bp = ax.boxplot(vd, tick_labels=names, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63"]
    for p, c in zip(bp["boxes"], colors[:len(names)]):
        p.set_facecolor(c); p.set_alpha(0.7)
    ax.axhline(y=8.2, color="purple", ls="--", lw=2, label="Pathak ~8.2")
    ax.set_ylabel("VPT (Lyapunov times)")
    ax.set_title(f"KS VPT: L={KS_L}, N={KS_N}")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pathak_vpt_comparison.png", dpi=150)
    plt.close(fig)

    # Spacetime
    fig3, ax3 = plt.subplots(1, 3, figsize=(18, 5))
    ns = min(200, MAX_ROLLOUT_STEPS)
    ex = test_start
    true_st = raw[ex:ex+ns]
    tax = np.arange(ns) * DT_LYAP_PER_STEP

    im0 = ax3[0].imshow(true_st.T, aspect="auto", cmap="RdBu_r",
                         extent=[0, tax[-1], 0, KS_L], origin="lower")
    ax3[0].set_title("True"); ax3[0].set_xlabel("Time (Lyap)"); ax3[0].set_ylabel("x")
    plt.colorbar(im0, ax=ax3[0])

    best_name = max(results.keys(), key=lambda n: results[n]["vpt_lyap"])
    pred_st = np.zeros((ns, KS_N))
    if best_name == "ParalESN+Ridge":
        pesn_ridge.warm_state(traj[max(0, ex-200):ex], device=device)
        ue = traj[ex].copy()
        pred_st[0] = ue * gr + gmin
        for t in range(1, ns):
            ue = pesn_ridge.predict_step(ue, device=device)
            ue = np.clip(ue, 0, 1).astype(np.float32)
            pred_st[t] = ue * gr + gmin
    elif best_name == "ParalESN+ResCor(D)":
        pesn_rescor.eval()
        pesn_rescor.forward_sequence(traj[max(0, ex-200):ex], device=device)
        ue = traj[ex].copy()
        pred_st[0] = ue * gr + gmin
        for t in range(1, ns):
            out = pesn_rescor.predict_step_recurrent(ue, device=device)
            ue = np.clip(out, 0, 1).astype(np.float32)
            pred_st[t] = ue * gr + gmin
    elif best_name == "Our ESN (256)":
        esn.warm_state(traj[max(0, ex-200):ex])
        ue = traj[ex].copy()
        pred_st[0] = ue * gr + gmin
        for t in range(1, ns):
            ue = esn.predict_step(ue)
            pred_st[t] = ue * gr + gmin
    else:
        best_m = {"ResCor(D)": rescor, "PureNCA": pnca, "Conv1D": cnn}[best_name]
        best_m.eval()
        xc = traj[ex].copy()
        pred_st[0] = xc * gr + gmin
        with torch.no_grad():
            for t in range(1, ns):
                xt = torch.from_numpy(xc).float().reshape(1, 1, KS_N).to(device)
                out = best_m(xt).cpu().numpy().reshape(KS_N)
                xc = np.clip(out, 0, 1).astype(np.float32)
                pred_st[t] = xc * gr + gmin

    im1 = ax3[1].imshow(pred_st.T, aspect="auto", cmap="RdBu_r",
                         extent=[0, tax[-1], 0, KS_L], origin="lower")
    ax3[1].set_title(f"Predicted ({best_name})"); ax3[1].set_xlabel("Time (Lyap)")
    plt.colorbar(im1, ax=ax3[1])

    im2 = ax3[2].imshow(np.abs(true_st - pred_st).T, aspect="auto", cmap="hot",
                         extent=[0, tax[-1], 0, KS_L], origin="lower")
    ax3[2].set_title("|Error|"); ax3[2].set_xlabel("Time (Lyap)")
    plt.colorbar(im2, ax=ax3[2])
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "pathak_spacetime.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # NRMSE over time
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    na = min(20, N_TEST_WINDOWS)
    nrmse_models = [
        ("Our ESN (256)", "#2196F3", "-"),
        ("ResCor(D)", "#F44336", "-"),
        ("PureNCA", "#4CAF50", "--"),
        ("Conv1D", "#FF9800", "--"),
        ("ParalESN+Ridge", "#9C27B0", "-"),
        ("ParalESN+ResCor(D)", "#E91E63", "-"),
    ]
    nn_models = {"ResCor(D)": rescor, "PureNCA": pnca, "Conv1D": cnn}
    for name, color, ls in nrmse_models:
        nr = np.zeros(MAX_ROLLOUT_STEPS); cnt = 0
        for w in range(na):
            st = test_start + w * (MAX_ROLLOUT_STEPS + 50)
            if st + MAX_ROLLOUT_STEPS + 1 > len(traj): break
            if name in nn_models:
                mobj = nn_models[name]
                mobj.eval()
                xc = traj[st].copy()
                with torch.no_grad():
                    for t in range(MAX_ROLLOUT_STEPS):
                        xt = torch.from_numpy(xc).float().reshape(1, 1, KS_N).to(device)
                        out = mobj(xt).cpu().numpy().reshape(KS_N)
                        nr[t] += nrmse(out * gr + gmin, raw[st+1+t])
                        xc = np.clip(out, 0, 1).astype(np.float32)
            elif name == "Our ESN (256)":
                esn.warm_state(traj[max(0, st-500):st])
                ue = traj[st].copy()
                for t in range(MAX_ROLLOUT_STEPS):
                    ue = esn.predict_step(ue)
                    nr[t] += nrmse(ue * gr + gmin, raw[st+1+t])
            elif name == "ParalESN+Ridge":
                pesn_ridge.warm_state(traj[max(0, st-500):st], device=device)
                ue = traj[st].copy()
                for t in range(MAX_ROLLOUT_STEPS):
                    ue = pesn_ridge.predict_step(ue, device=device)
                    nr[t] += nrmse(ue * gr + gmin, raw[st+1+t])
                    ue = np.clip(ue, 0, 1).astype(np.float32)
            elif name == "ParalESN+ResCor(D)":
                pesn_rescor.eval()
                pesn_rescor.forward_sequence(traj[max(0, st-500):st], device=device)
                ue = traj[st].copy()
                for t in range(MAX_ROLLOUT_STEPS):
                    out = pesn_rescor.predict_step_recurrent(ue, device=device)
                    nr[t] += nrmse(out * gr + gmin, raw[st+1+t])
                    ue = np.clip(out, 0, 1).astype(np.float32)
            cnt += 1
        nr /= max(cnt, 1)
        ax4.plot(np.arange(MAX_ROLLOUT_STEPS) * DT_LYAP_PER_STEP, nr,
                 color=color, ls=ls, label=name, lw=1.5)

    ax4.axhline(y=VPT_THRESHOLD, color="gray", ls=":", lw=1.5, label=f"Threshold ({VPT_THRESHOLD})")
    ax4.set_xlabel("Time (Lyapunov times)"); ax4.set_ylabel("NRMSE")
    ax4.set_title("KS: Average NRMSE Over Time")
    ax4.legend(fontsize=9); ax4.set_ylim(0, 1.5); ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(PLOTS_DIR / "pathak_nrmse_over_time.png", dpi=150)
    plt.close(fig4)

    print(f"  Plots saved to {PLOTS_DIR}/")
    print("  DONE!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-wandb", action="store_true", default=True)
    run_experiment(p.parse_args())
