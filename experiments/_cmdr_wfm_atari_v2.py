"""Atari WFM v2 — joint Pong+Breakout pre-train, two-direction LOO transfer test.

CML2DMultiR with gate_mode='uniform' is fully frozen and deterministic, so we
precompute CML(X) once per dataset and only the NCA gradient step runs each
epoch — ~10x faster than v1's per-step CML re-evaluation, mathematically
identical training.

Hypothesis: joint pre-training on Pong+Breakout transfers better than single-env
(Pong-only) pre-training, beating from-scratch on at least one direction.
"""
import sys, time, numpy as np, torch, torch.nn as nn, os
sys.path.insert(0, 'src')
from wmca.modules.hybrid import CML2DMultiR

DEVICE = torch.device('mps')
EPOCHS_PRE = 50
EPOCHS_FT = 20
EPOCHS_SCRATCH = 50
BATCH = 32
SEED = 42
GRID_WFM = 32
N_ACTIONS = 3
LR_PRE = 1e-3
LR_FT = 1e-4
LR_SCRATCH = 1e-3
EVAL_N = 256
PRECOMPUTE_BATCH = 128

torch.manual_seed(SEED)
np.random.seed(SEED)


def make_action_field(actions, H, W):
    vals = (actions + 1.0) / N_ACTIONS
    return np.tile(vals[:, None, None, None], (1, 1, H, W)).astype(np.float32)


def pad_to(X, target_h, target_w):
    ph = (target_h - X.shape[2]) // 2
    pw = (target_w - X.shape[3]) // 2
    return np.pad(
        X,
        ((0, 0), (0, 0), (ph, target_h - X.shape[2] - ph), (pw, target_w - X.shape[3] - pw)),
    )


def load_env_padded(game):
    latents = np.load(f'experiments/atari_data/{game}_latents.npy')
    next_latents = np.load(f'experiments/atari_data/{game}_next_latents.npy')
    actions = np.load(f'experiments/atari_data/{game}_actions.npy')[: len(latents)]
    H, W = latents.shape[2], latents.shape[3]
    act_field = make_action_field(actions, H, W)
    X = np.concatenate([latents, act_field], axis=1)  # (N, 2, H, W)
    Y = next_latents
    X = pad_to(X, GRID_WFM, GRID_WFM)
    Y = pad_to(Y, GRID_WFM, GRID_WFM)
    return X, Y, H, W


print("Loading Atari data...")
X_p, Y_p, Hp, Wp = load_env_padded('pong')
X_b, Y_b, Hb, Wb = load_env_padded('breakout')
print(f"Pong:     orig grid {Hp}x{Wp} -> padded {X_p.shape}")
print(f"Breakout: orig grid {Hb}x{Wb} -> padded {X_b.shape}")

n_p = len(X_p); n_b = len(X_b)
n_p_tr = int(n_p * 0.7); n_b_tr = int(n_b * 0.7)
Xp_tr = torch.from_numpy(X_p[:n_p_tr]).float().to(DEVICE); Yp_tr = torch.from_numpy(Y_p[:n_p_tr]).float().to(DEVICE)
Xp_te = torch.from_numpy(X_p[n_p_tr:]).float().to(DEVICE); Yp_te = torch.from_numpy(Y_p[n_p_tr:]).float().to(DEVICE)
Xb_tr = torch.from_numpy(X_b[:n_b_tr]).float().to(DEVICE); Yb_tr = torch.from_numpy(Y_b[:n_b_tr]).float().to(DEVICE)
Xb_te = torch.from_numpy(X_b[n_b_tr:]).float().to(DEVICE); Yb_te = torch.from_numpy(Y_b[n_b_tr:]).float().to(DEVICE)
print(f"Pong:     train={n_p_tr}  test={n_p - n_p_tr}")
print(f"Breakout: train={n_b_tr}  test={n_b - n_b_tr}")


# ────────────────────────────────────────────────────────────────────────────
# Shared CML (gate_mode='uniform' = no trainable params, deterministic)
# Precompute CML(X) once for every (X) we'll ever pass through.
# ────────────────────────────────────────────────────────────────────────────
print("\nPre-computing CML outputs (gate_mode='uniform' is deterministic)...")
SHARED_CML = CML2DMultiR(in_channels=2, K=32, steps=15, seed=SEED, gate_mode='uniform').to(DEVICE)
SHARED_CML.eval()


def precompute_cml(X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), PRECOMPUTE_BATCH):
            out.append(SHARED_CML(X[i:i + PRECOMPUTE_BATCH]))
    return torch.cat(out, dim=0)


t0 = time.time()
CXp_tr = precompute_cml(Xp_tr); CXp_te = precompute_cml(Xp_te)
CXb_tr = precompute_cml(Xb_tr); CXb_te = precompute_cml(Xb_te)
print(f"  Precomputed CML for all 4 datasets in {time.time()-t0:.1f}s")
print(f"  Cached: pong_train={CXp_tr.shape}, pong_test={CXp_te.shape}, "
      f"breakout_train={CXb_tr.shape}, breakout_test={CXb_te.shape}")


class NCAOnly(nn.Module):
    """The trainable part of RescorWFM. Identical computation to v1's
    RescorWFM but accepts pre-computed CML output, skipping the CML pass."""

    def __init__(self, in_ch=2, out_ch=1, hid=64):
        super().__init__()
        self.nca = nn.Sequential(
            nn.Conv2d(in_ch * 2, hid, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hid, out_ch, 1),
        )
        self.out_ch = out_ch

    def forward(self, x, cml_out):
        return cml_out[:, :self.out_ch] + self.nca(torch.cat([x, cml_out], dim=1))


def make_model():
    torch.manual_seed(SEED)
    return NCAOnly(in_ch=2, out_ch=1, hid=64).to(DEVICE)


def eval_mse(model, X, CML_X, Y, n=EVAL_N):
    model.eval()
    with torch.no_grad():
        return nn.functional.mse_loss(model(X[:n], CML_X[:n]), Y[:n]).item()


def train_single(model, X, CML_X, Y, epochs, lr, label, log_every=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            pred = model(X[idx], CML_X[idx])
            loss = nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % log_every == 0 or ep == epochs - 1:
            train_mse = eval_mse(model, X, CML_X, Y)
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (ep + 1) * (epochs - ep - 1)
            print(f"  [{label}] ep {ep + 1:3d}/{epochs} train_mse={train_mse:.2e} elapsed={elapsed:.1f}m eta={eta:.1f}m")
    return model


def train_joint(model, datasets, epochs, lr, label, log_every=5):
    """datasets: list of (X, CML_X, Y, name). Round-robin batches per epoch."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perms = [(name, torch.randperm(len(X), device=DEVICE), X, CX, Y) for X, CX, Y, name in datasets]
        ptrs = [0] * len(datasets)
        active = True
        while active:
            active = False
            for k in range(len(datasets)):
                name, perm, X, CX, Y = perms[k]
                if ptrs[k] < len(perm):
                    idx = perm[ptrs[k]:ptrs[k] + BATCH]
                    ptrs[k] += BATCH
                    pred = model(X[idx], CX[idx])
                    loss = nn.functional.mse_loss(pred, Y[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
                    active = True
        if ep % log_every == 0 or ep == epochs - 1:
            vals = []
            for X, CX, Y, name in datasets:
                vals.append(f"{name[:5]}={eval_mse(model, X, CX, Y):.2e}")
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (ep + 1) * (epochs - ep - 1)
            print(f"  [{label}] ep {ep + 1:3d}/{epochs} {' '.join(vals)} elapsed={elapsed:.1f}m eta={eta:.1f}m")
    return model


os.makedirs('experiments/results', exist_ok=True)
RESULTS_DIR = 'experiments/results'
T_total = time.time()

# ────────────────────────────────────────────────────────────────────────────
# STAGE 1: Joint pre-train on Pong(70%) + Breakout(70%)
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSTAGE 1: Joint pre-train (Pong+Breakout, {EPOCHS_PRE} ep)\n{'='*60}")
m_joint = make_model()
pc = sum(p.numel() for p in m_joint.parameters() if p.requires_grad)
print(f"NCA: trained={pc:,} (CML buffers={sum(b.numel() for b in SHARED_CML.buffers()):,})")
train_joint(
    m_joint,
    [(Xp_tr, CXp_tr, Yp_tr, 'pong'), (Xb_tr, CXb_tr, Yb_tr, 'breakout')],
    EPOCHS_PRE, LR_PRE, 'joint',
)
joint_path = f'{RESULTS_DIR}/wfm_joint_v2.pt'
torch.save(m_joint.state_dict(), joint_path)

# ────────────────────────────────────────────────────────────────────────────
# STAGE 2: Pong-only pre-train (v1 baseline replicated in same code path)
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSTAGE 2: Pong-only pre-train ({EPOCHS_PRE} ep)\n{'='*60}")
m_pong_only = make_model()
train_single(m_pong_only, Xp_tr, CXp_tr, Yp_tr, EPOCHS_PRE, LR_PRE, 'pong-only')
pong_only_path = f'{RESULTS_DIR}/wfm_pong_only_v2.pt'
torch.save(m_pong_only.state_dict(), pong_only_path)

# ────────────────────────────────────────────────────────────────────────────
# STAGE 3: Zero-shot evals
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSTAGE 3: Zero-shot evaluations\n{'='*60}")
zs = {
    'joint_on_pong':         eval_mse(m_joint,     Xp_te, CXp_te, Yp_te),
    'joint_on_breakout':     eval_mse(m_joint,     Xb_te, CXb_te, Yb_te),
    'pong_only_on_pong':     eval_mse(m_pong_only, Xp_te, CXp_te, Yp_te),
    'pong_only_on_breakout': eval_mse(m_pong_only, Xb_te, CXb_te, Yb_te),
}
for k, v in zs.items():
    print(f"  {k:<28s}: {v:.2e}")

# ────────────────────────────────────────────────────────────────────────────
# STAGE 4: FT branches
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSTAGE 4: Fine-tuning branches\n{'='*60}")

print(f"\n[FT-1] Joint -> FT Pong ({EPOCHS_FT} ep)")
m_ft_jp = make_model(); m_ft_jp.load_state_dict(torch.load(joint_path))
train_single(m_ft_jp, Xp_tr, CXp_tr, Yp_tr, EPOCHS_FT, LR_FT, 'joint+ft_pong')
ft_joint_pong = eval_mse(m_ft_jp, Xp_te, CXp_te, Yp_te)
print(f"  Joint+FT_pong test MSE: {ft_joint_pong:.2e}")
del m_ft_jp

print(f"\n[FT-2] Joint -> FT Breakout ({EPOCHS_FT} ep)")
m_ft_jb = make_model(); m_ft_jb.load_state_dict(torch.load(joint_path))
train_single(m_ft_jb, Xb_tr, CXb_tr, Yb_tr, EPOCHS_FT, LR_FT, 'joint+ft_break')
ft_joint_break = eval_mse(m_ft_jb, Xb_te, CXb_te, Yb_te)
print(f"  Joint+FT_breakout test MSE: {ft_joint_break:.2e}")
del m_ft_jb

print(f"\n[FT-3] Pong-only -> FT Breakout (v1 replicate, {EPOCHS_FT} ep)")
m_ft_pb = make_model(); m_ft_pb.load_state_dict(torch.load(pong_only_path))
train_single(m_ft_pb, Xb_tr, CXb_tr, Yb_tr, EPOCHS_FT, LR_FT, 'pong+ft_break')
ft_pong_break = eval_mse(m_ft_pb, Xb_te, CXb_te, Yb_te)
print(f"  Pong-only+FT_breakout test MSE: {ft_pong_break:.2e}")
del m_ft_pb

# ────────────────────────────────────────────────────────────────────────────
# STAGE 5: From-scratch baselines
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSTAGE 5: From-scratch baselines\n{'='*60}")

print(f"\n[SCRATCH-1] From-scratch Pong ({EPOCHS_SCRATCH} ep)")
m_s_p = make_model()
train_single(m_s_p, Xp_tr, CXp_tr, Yp_tr, EPOCHS_SCRATCH, LR_SCRATCH, 'scratch_pong')
scratch_pong = eval_mse(m_s_p, Xp_te, CXp_te, Yp_te)
print(f"  Scratch_pong test MSE: {scratch_pong:.2e}")
del m_s_p

print(f"\n[SCRATCH-2] From-scratch Breakout ({EPOCHS_SCRATCH} ep)")
m_s_b = make_model()
train_single(m_s_b, Xb_tr, CXb_tr, Yb_tr, EPOCHS_SCRATCH, LR_SCRATCH, 'scratch_break')
scratch_break = eval_mse(m_s_b, Xb_te, CXb_te, Yb_te)
print(f"  Scratch_breakout test MSE: {scratch_break:.2e}")
del m_s_b

# ────────────────────────────────────────────────────────────────────────────
# REPORT
# ────────────────────────────────────────────────────────────────────────────
total_min = (time.time() - T_total) / 60
print(f"\n\n{'='*60}\nFINAL RESULTS  (total runtime {total_min:.1f}m)\n{'='*60}")

print(f"\n--- Pre-train MSE (zero-shot on test slices) ---")
print(f"  Joint        on Pong test:     {zs['joint_on_pong']:.2e}")
print(f"  Joint        on Breakout test: {zs['joint_on_breakout']:.2e}")
print(f"  Pong-only    on Pong test:     {zs['pong_only_on_pong']:.2e}")
print(f"  Pong-only    on Breakout test: {zs['pong_only_on_breakout']:.2e}    (v1 zero-shot 1.38e-02 reference)")

print(f"\n--- Fine-tuned MSE (20ep FT) ---")
print(f"  Joint+FT_pong:                 {ft_joint_pong:.2e}")
print(f"  Joint+FT_breakout:             {ft_joint_break:.2e}")
print(f"  Pong-only+FT_breakout:         {ft_pong_break:.2e}    (v1 reference 2.56e-04)")

print(f"\n--- From-scratch MSE (50ep) ---")
print(f"  Scratch_pong:                  {scratch_pong:.2e}")
print(f"  Scratch_breakout:              {scratch_break:.2e}    (v1 reference 2.15e-04)")

print(f"\n--- TRANSFER RATIOS (scratch / FT, >1 = transfer WIN) ---")
ratio_pong = scratch_pong / ft_joint_pong
ratio_break_joint = scratch_break / ft_joint_break
ratio_break_pong = scratch_break / ft_pong_break
print(f"  Pong:     scratch/joint+FT       = {ratio_pong:.2f}x  {'WIN' if ratio_pong > 1 else 'LOSS'}")
print(f"  Breakout: scratch/joint+FT       = {ratio_break_joint:.2f}x  {'WIN' if ratio_break_joint > 1 else 'LOSS'}")
print(f"  Breakout: scratch/pong-only+FT   = {ratio_break_pong:.2f}x  {'WIN' if ratio_break_pong > 1 else 'LOSS'}  (cf v1 0.84x)")

print(f"\n--- MULTI-ENV ADVANTAGE (joint vs single-env pre-train, on Breakout FT) ---")
multi_env_advantage = ft_pong_break / ft_joint_break
print(f"  pong-only+FT / joint+FT (Breakout) = {multi_env_advantage:.2f}x  {'JOINT WINS' if multi_env_advantage > 1 else 'JOINT LOSES'}")

print("\nDone.")
