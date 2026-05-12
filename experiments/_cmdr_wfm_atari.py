"""Atari WFM — pre-train on Pong, test transfer to Breakout."""
import sys, time, numpy as np, torch, torch.nn as nn, os
sys.path.insert(0, 'src')
from wmca.modules.hybrid import CML2DMultiR

DEVICE = torch.device('mps')
EPOCHS = 50; BATCH = 8; SEED = 42
GRID_WFM = 32  # pad everything to 32x32

N_ACTIONS = 3

def make_action_field(actions, H, W):
    vals = (actions + 1.0) / N_ACTIONS
    return np.tile(vals[:, None, None, None], (1, 1, H, W)).astype(np.float32)

def load_env(game):
    latents = np.load(f'experiments/atari_data/{game}_latents.npy')
    next_latents = np.load(f'experiments/atari_data/{game}_next_latents.npy')
    actions = np.load(f'experiments/atari_data/{game}_actions.npy')[:len(latents)]
    H, W = latents.shape[2], latents.shape[3]
    act_field = make_action_field(actions, H, W)
    X = np.concatenate([latents, act_field], axis=1)  # (N, 2, H, W)
    Y = next_latents
    return X, Y, H, W

def pad_to(X, target_h, target_w):
    """Pad spatial dims to target. (N, C, H, W) -> (N, C, target_h, target_w)"""
    ph = (target_h - X.shape[2]) // 2
    pw = (target_w - X.shape[3]) // 2
    return np.pad(X, ((0,0),(0,0),(ph, target_h - X.shape[2] - ph),(pw, target_w - X.shape[3] - pw)))

print("Loading Atari data...")
X_p, Y_p, Hp, Wp = load_env('pong')
X_b, Y_b, Hb, Wb = load_env('breakout')
print(f"Pong: {X_p.shape} (grid {Hp}x{Wp})")
print(f"Breakout: {X_b.shape} (grid {Hb}x{Wb})")

# Pad both to 32x32
X_p = pad_to(X_p, GRID_WFM, GRID_WFM)
Y_p = pad_to(Y_p, GRID_WFM, GRID_WFM)
X_b = pad_to(X_b, GRID_WFM, GRID_WFM)
Y_b = pad_to(Y_b, GRID_WFM, GRID_WFM)
print(f"Padded to {GRID_WFM}x{GRID_WFM}")

# Train on Pong (70/15 split), test on Breakout
n_pt = int(len(X_p) * 0.7); n_pv = int(len(X_p) * 0.15)
Xp_tr = torch.from_numpy(X_p[:n_pt]).float().to(DEVICE); Yp_tr = torch.from_numpy(Y_p[:n_pt]).float().to(DEVICE)
Xp_val = torch.from_numpy(X_p[n_pt:n_pt+n_pv]).float().to(DEVICE); Yp_val = torch.from_numpy(Y_p[n_pt:n_pt+n_pv]).float().to(DEVICE)

# Breakout: use last 3750 for test
n_bt = int(len(X_b) * 0.7)
Xb_tr = torch.from_numpy(X_b[:n_bt]).float(); Yb_tr = torch.from_numpy(Y_b[:n_bt]).float()
Xb_ts = torch.from_numpy(X_b[n_bt:]).float(); Yb_ts = torch.from_numpy(Y_b[n_bt:]).float()

in_ch, out_ch = 2, 1  # latent + action field -> next latent

class RescorWFM(nn.Module):
    def __init__(self, in_ch, out_ch, K=32, hid=64, depth=2, seed=42):
        super().__init__()
        self.cml = CML2DMultiR(in_channels=in_ch, K=K, steps=15, seed=seed, gate_mode='uniform')
        self.nca = nn.Sequential(
            nn.Conv2d(in_ch*2, hid, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hid, out_ch, 1),
        )
        self.out_ch = out_ch
    def forward(self, x):
        with torch.no_grad(): cml_out = self.cml(x)
        return cml_out[:, :self.out_ch] + self.nca(torch.cat([x, cml_out], dim=1))

model = RescorWFM(in_ch, out_ch, K=32, hid=64, depth=2, seed=SEED).to(DEVICE)
pc = {'trained': sum(p.numel() for p in model.parameters() if p.requires_grad),
      'frozen': sum(b.numel() for b in model.buffers())}
print(f"\nRescorWFM: trained={pc['trained']:,} frozen={pc['frozen']:,}")

# Pre-train on Pong
print(f"\nPre-training on Pong ({EPOCHS} epochs)...")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(Xp_tr), device=DEVICE)
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        loss = nn.functional.mse_loss(model(Xp_tr[idx]), Yp_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 10 == 0 or ep == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            val_mse = nn.functional.mse_loss(model(Xp_val[:256]), Yp_val[:256]).item()
        eta = (time.time()-t0)/(ep+1)*(EPOCHS-ep-1)
        print(f"  ep {ep+1:2d}/{EPOCHS} val={val_mse:.2e} eta={eta/60:.0f}m")

os.makedirs('experiments/results', exist_ok=True)
torch.save(model.state_dict(), 'experiments/results/wfm_pong.pt')

# Zero-shot on Breakout
print(f"\n{'='*50}\nZERO-SHOT + FINE-TUNE on Breakout\n{'='*50}")
model.eval()
with torch.no_grad():
    zero = nn.functional.mse_loss(model(Xb_ts[:256].to(DEVICE)), Yb_ts[:256].to(DEVICE)).item()
print(f"Zero-shot: {zero:.2e}")

# Fine-tune on Breakout (20 epochs)
n_ft = n_bt
Xft, Yft = Xb_tr.to(DEVICE), Yb_tr.to(DEVICE)
Xfv, Yfv = Xb_ts[:256].to(DEVICE), Yb_ts[:256].to(DEVICE)
model_ft = RescorWFM(in_ch, out_ch, K=32, hid=64, depth=2, seed=SEED).to(DEVICE)
model_ft.load_state_dict(torch.load('experiments/results/wfm_pong.pt'))
opt_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-4)
for ep in range(20):
    model_ft.train(); perm = torch.randperm(n_ft, device=DEVICE)
    for i in range(0, n_ft, BATCH):
        loss = nn.functional.mse_loss(model_ft(Xft[perm[i:i+BATCH]]), Yft[perm[i:i+BATCH]])
        opt_ft.zero_grad(); loss.backward(); opt_ft.step()
model_ft.eval()
with torch.no_grad():
    ft = nn.functional.mse_loss(model_ft(Xfv), Yfv).item()
print(f"Fine-tuned (20ep): {ft:.2e}")

# From-scratch
model_s = RescorWFM(in_ch, out_ch, K=32, hid=64, depth=2, seed=SEED).to(DEVICE)
opt_s = torch.optim.Adam(model_s.parameters(), lr=1e-3)
for ep in range(50):
    model_s.train(); perm = torch.randperm(n_ft, device=DEVICE)
    for i in range(0, n_ft, BATCH):
        loss = nn.functional.mse_loss(model_s(Xft[perm[i:i+BATCH]]), Yft[perm[i:i+BATCH]])
        opt_s.zero_grad(); loss.backward(); opt_s.step()
model_s.eval()
with torch.no_grad():
    s = nn.functional.mse_loss(model_s(Xfv), Yfv).item()
print(f"From-scratch (50ep): {s:.2e}")

print(f"\n{'='*50}\nRESULTS:")
print(f"  Zero-shot:           {zero:.2e}")
print(f"  Fine-tuned (20ep):   {ft:.2e}")
print(f"  From-scratch (50ep): {s:.2e}")
print(f"  FT vs scratch:       {s/ft:.1f}x {'WIN!' if ft < s else 'LOSS'}")
