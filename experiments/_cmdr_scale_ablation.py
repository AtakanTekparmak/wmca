"""Rescor WFM scale ablation — sweep K and NCA on Gray-Scott 32×32 (spatial PDE)."""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')
from wmca.modules.hybrid import CML2DMultiR
from wmca.benchmarks import generate_gray_scott

DEVICE = torch.device('mps')
EPOCHS = 30; BATCH = 8; SEED = 42

# Generate Gray-Scott at 32×32
data = generate_gray_scott(grid_size=32, n_trajectories=100, n_steps=50)
X_all = data.X_train; Y_all = data.Y_train
N = len(X_all); n_train = int(N * 0.8)
in_ch = X_all.shape[1]; out_ch = Y_all.shape[1]
H, W = X_all.shape[2], X_all.shape[3]
print(f'Gray-Scott: {N} samples, {in_ch}ch, {H}x{W}')

_to_dev = lambda t: (torch.from_numpy(t) if isinstance(t, np.ndarray) else t).float().to(DEVICE)
X_tr = _to_dev(X_all[:n_train])
Y_tr = _to_dev(Y_all[:n_train])
X_val = _to_dev(X_all[n_train:])
Y_val = _to_dev(Y_all[n_train:])

class RescorScaled(nn.Module):
    def __init__(self, in_ch, out_ch, K=32, hid=16, depth=1, seed=42):
        super().__init__()
        self.cml = CML2DMultiR(in_channels=in_ch, K=K, steps=15, seed=seed, gate_mode='uniform')
        layers = []
        for d in range(depth):
            layers += [nn.Conv2d(in_ch*2 if d==0 else hid, hid, 3, 1, 1), nn.ReLU()]
        layers.append(nn.Conv2d(hid if depth>0 else in_ch*2, out_ch, 1))
        self.nca = nn.Sequential(*layers)
        self.use_sigmoid = (out_ch == in_ch)

    def forward(self, x):
        with torch.no_grad(): cml_out = self.cml(x)
        corr = self.nca(torch.cat([x, cml_out], dim=1))
        out = cml_out + corr
        if self.use_sigmoid: return torch.clamp(out, 0, 1)
        return out

    def param_count(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        f = sum(b.numel() for b in self.buffers())
        return {'trained': t, 'frozen': f}

K_values = [1, 4, 8, 16, 32, 64, 128, 256]
depths = [1, 2, 3]
hids = [16, 32, 64]

results = []
total = 0
for K in K_values:
    for depth in depths:
        for hid in hids:
            total += 1
            torch.manual_seed(SEED)
            model = RescorScaled(in_ch, out_ch, K=K, hid=hid, depth=depth, seed=SEED).to(DEVICE)
            pc = model.param_count()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            best_val = float('inf'); t0 = time.time()

            for ep in range(EPOCHS):
                model.train()
                perm = torch.randperm(n_train, device=DEVICE)
                for i in range(0, n_train, BATCH):
                    idx = perm[i:i+BATCH]
                    loss = nn.functional.mse_loss(model(X_tr[idx]), Y_tr[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    val_mse = nn.functional.mse_loss(model(X_val[:64]), Y_val[:64]).item()
                if val_mse < best_val: best_val = val_mse

            elapsed = time.time() - t0
            results.append({'K': K, 'depth': depth, 'hid': hid, 'trained': pc['trained'], 'val_mse': best_val, 'time': elapsed})
            print(f'[{len(results)}/{total}] K={K:3d} d={depth} h={hid:3d} train={pc["trained"]:5d} val={best_val:.2e} [{elapsed:.0f}s]')

results.sort(key=lambda r: r['val_mse'])
print(f'\n{"="*70}')
print(f'TOP 15 CONFIGS (Gray-Scott 32x32, 30 epochs):')
print(f'{"K":>4s} {"d":>2s} {"hid":>4s} {"train":>6s} {"val_mse":>10s}')
for r in results[:15]:
    print(f'{r["K"]:4d} {r["depth"]:2d} {r["hid"]:4d} {r["trained"]:6d} {r["val_mse"]:10.2e}')

print(f'\nBEST PER K:')
for K in K_values:
    best = min([r for r in results if r['K']==K], key=lambda r: r['val_mse'])
    print(f'K={K:3d}: d={best["depth"]} h={best["hid"]:3d} train={best["trained"]:5d} val={best["val_mse"]:.2e}')

print(f'\nSCALING CONCLUSION: best=K={results[0]["K"]} depth={results[0]["depth"]} hid={results[0]["hid"]} params={results[0]["trained"]} val={results[0]["val_mse"]:.2e}')
