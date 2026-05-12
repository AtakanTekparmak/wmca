"""Multi-env WFM — train K=32/d=2/h=64 on Heat + Gray-Scott, test transfer."""
import sys, time, numpy as np, torch, torch.nn as nn, os
sys.path.insert(0, 'src')
from wmca.modules.hybrid import CML2DMultiR
from wmca.benchmarks import generate_heat, generate_gray_scott

DEVICE = torch.device('mps')
EPOCHS = 50; BATCH = 8; SEED = 42; GRID = 32

def to_np(t): return t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

print("Generating data...")
raw = {
    'heat': generate_heat(grid_size=GRID, n_trajectories=100, n_steps=30, seed=SEED),
    'gray_scott': generate_gray_scott(grid_size=GRID, n_trajectories=100, n_steps=30, seed=SEED),
}

max_ch = max(to_np(d.X_train).shape[1] for d in raw.values())
out_ch = max(to_np(d.Y_train).shape[1] for d in raw.values())

def pad_ch(X_np, target):
    if X_np.shape[1] < target:
        return np.pad(X_np, ((0,0),(0,target-X_np.shape[1]),(0,0),(0,0)))
    return X_np

envs = {}
for name, d in raw.items():
    Xt = pad_ch(to_np(d.X_train), max_ch); Yt = pad_ch(to_np(d.Y_train), out_ch)
    Xv = pad_ch(to_np(d.X_test), max_ch); Yv = pad_ch(to_np(d.Y_test), out_ch)
    envs[name] = {
        'X_tr': torch.from_numpy(Xt).float().to(DEVICE),
        'Y_tr': torch.from_numpy(Yt).float().to(DEVICE),
        'X_ts': torch.from_numpy(Xv).float().to(DEVICE),
        'Y_ts': torch.from_numpy(Yv).float().to(DEVICE),
    }
    print(f"  {name}: {len(envs[name]['X_tr'])} train, {max_ch}ch padded, {GRID}x{GRID}")

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
    def param_count(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        f = sum(b.numel() for b in self.buffers())
        return {'trained': t, 'frozen': f}

model = RescorWFM(max_ch, out_ch, K=32, hid=64, depth=2, seed=SEED).to(DEVICE)
pc = model.param_count()
print(f"\nRescorWFM: trained={pc['trained']:,} frozen={pc['frozen']:,}")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"\nTraining on 2 PDEs for {EPOCHS} epochs...")
t0 = time.time()

for ep in range(EPOCHS):
    model.train()
    for env in envs.values():
        n = len(env['X_tr']); perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            loss = nn.functional.mse_loss(model(env['X_tr'][idx]), env['Y_tr'][idx])
            opt.zero_grad(); loss.backward(); opt.step()

    if ep % 10 == 0 or ep == EPOCHS - 1:
        model.eval(); vals = []
        for name, env in envs.items():
            with torch.no_grad():
                mse = nn.functional.mse_loss(model(env['X_ts'][:256]), env['Y_ts'][:256]).item()
            vals.append(f"{name[:5]}={mse:.2e}")
        elapsed = (time.time() - t0) / 60
        eta = elapsed / (ep+1) * (EPOCHS - ep - 1)
        print(f"ep {ep+1:2d}/{EPOCHS} [{elapsed:.0f}m eta={eta:.0f}m] {' '.join(vals)}")

os.makedirs('experiments/results', exist_ok=True)
torch.save(model.state_dict(), 'experiments/results/wfm_multienv.pt')

# ZERO-SHOT + FINE-TUNE on held-out Heat (different seed, different params)
print(f"\n{'='*50}\nZERO-SHOT + FINE-TUNE\n{'='*50}")
held = generate_heat(grid_size=GRID, n_trajectories=40, n_steps=30, seed=99)
hX = torch.from_numpy(pad_ch(to_np(held.X_test), max_ch)).float()
hY = torch.from_numpy(pad_ch(to_np(held.Y_test), out_ch)).float()

model.eval()
with torch.no_grad():
    zero_mse = nn.functional.mse_loss(model(hX[:256].to(DEVICE)), hY[:256].to(DEVICE)).item()
print(f"Zero-shot: {zero_mse:.2e}")

# Fine-tune 20 epochs
n_ft = int(len(hX) * 0.8)
Xft, Yft = hX[:n_ft].to(DEVICE), hY[:n_ft].to(DEVICE)
Xfv, Yfv = hX[n_ft:n_ft+256].to(DEVICE), hY[n_ft:n_ft+256].to(DEVICE)

model_ft = RescorWFM(max_ch, out_ch, K=32, hid=64, depth=2, seed=SEED).to(DEVICE)
model_ft.load_state_dict(torch.load('experiments/results/wfm_multienv.pt'))
opt_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-4)
for ep in range(20):
    model_ft.train(); perm = torch.randperm(n_ft, device=DEVICE)
    for i in range(0, n_ft, BATCH):
        loss = nn.functional.mse_loss(model_ft(Xft[perm[i:i+BATCH]]), Yft[perm[i:i+BATCH]])
        opt_ft.zero_grad(); loss.backward(); opt_ft.step()
model_ft.eval()
with torch.no_grad():
    ft_mse = nn.functional.mse_loss(model_ft(Xfv), Yfv).item()
print(f"Fine-tuned (20ep): {ft_mse:.2e}")

# From-scratch baseline
model_s = RescorWFM(max_ch, out_ch, K=32, hid=64, depth=2, seed=SEED).to(DEVICE)
opt_s = torch.optim.Adam(model_s.parameters(), lr=1e-3)
for ep in range(50):
    model_s.train(); perm = torch.randperm(n_ft, device=DEVICE)
    for i in range(0, n_ft, BATCH):
        loss = nn.functional.mse_loss(model_s(Xft[perm[i:i+BATCH]]), Yft[perm[i:i+BATCH]])
        opt_s.zero_grad(); loss.backward(); opt_s.step()
model_s.eval()
with torch.no_grad():
    s_mse = nn.functional.mse_loss(model_s(Xfv), Yfv).item()
print(f"From-scratch (50ep): {s_mse:.2e}")

print(f"\n{'='*50}\nRESULTS:")
print(f"  Zero-shot:           {zero_mse:.2e}")
print(f"  Fine-tuned (20ep):   {ft_mse:.2e}")
print(f"  From-scratch (50ep): {s_mse:.2e}")
print(f"  FT vs scratch:       {s_mse/ft_mse:.1f}x {'WIN!' if ft_mse < s_mse else 'LOSS'}")
