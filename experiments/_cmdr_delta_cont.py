
"""Train RescorDeltaContinuous on delta embeddings — Architecture A."""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')
from wmca.modules.hybrid import CML2DMultiR

DEVICE = torch.device('mps')
D = 64
EPOCHS = 50
BATCH = 8
SEEDS = [42, 43, 44]

# Load pre-encoded delta embeddings
deltas = np.load('experiments/crafter_data/delta_embeddings.npy', mmap_mode='r')
actions = np.load('experiments/crafter_data/actions.npy')
N = len(deltas) - 1
n_train = int(N * 0.7); n_val = int(N * 0.15)
print(f'Delta embeddings: {deltas.shape}, {N} training pairs')

class RescorDeltaContinuous(nn.Module):
    def __init__(self, D=64, hid=16, K=32, seed=42):
        super().__init__()
        self.action_embed = nn.Embedding(17, D)  # Crafter 17 actions
        self.input_proj = nn.Conv2d(D * 2, 1, 1)  # delta_emb + action_emb -> 1ch
        self.cml = CML2DMultiR(in_channels=1, K=K, steps=15, seed=seed, gate_mode='uniform')
        self.nca = nn.Sequential(nn.Conv2d(2, hid, 3, 1, 1), nn.ReLU(), nn.Conv2d(hid, D, 1))
        self.cml_proj = nn.Conv2d(1, D, 1)

    def forward(self, delta_emb, action):
        act = self.action_embed(action).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 16, 16)
        combined = torch.cat([delta_emb, act], dim=1)  # (B, 2D, 16, 16)
        drive = torch.sigmoid(self.input_proj(combined))
        cml_out = self.cml(drive)
        corr = self.nca(torch.cat([drive, cml_out], dim=1))
        cml_e = self.cml_proj(cml_out)
        return corr + cml_e

    def param_count(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        f = sum(b.numel() for b in self.buffers())
        return {'trained': t, 'frozen': f}

for seed in SEEDS:
    torch.manual_seed(seed)
    model = RescorDeltaContinuous(D=D, seed=seed).to(DEVICE)
    pc = model.param_count()
    opt = torch.optim.Adam(model.parameters(), lr=1.4e-3)
    best_val = float('inf'); best_state = None; t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0; n_b = 0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i+BATCH]
            d0 = torch.from_numpy(deltas[idx].copy()).float().to(DEVICE)
            d1 = torch.from_numpy(deltas[idx + 1].copy()).float().to(DEVICE)
            act = torch.from_numpy(actions[idx]).long().to(DEVICE)
            pred = model(d0, act)
            loss = nn.functional.mse_loss(pred, d1)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_b += 1

        model.eval()
        with torch.no_grad():
            dv = torch.from_numpy(deltas[n_train:n_train+64].copy()).float().to(DEVICE)
            dv1 = torch.from_numpy(deltas[n_train+1:n_train+65].copy()).float().to(DEVICE)
            av = torch.from_numpy(actions[n_train:n_train+64]).long().to(DEVICE)
            val_mse = nn.functional.mse_loss(model(dv, av), dv1).item()
        if val_mse < best_val:
            best_val = val_mse; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == EPOCHS - 1:
            eta = (time.time() - t0) / (ep + 1) * (EPOCHS - ep - 1)
            print(f'  seed={seed} ep={ep+1:2d}/{EPOCHS} val_mse={val_mse:.6e} eta={eta/60:.0f}m')

    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), f'experiments/crafter_data/rescor_delta_cont_seed{seed}.pt')
    print(f'  seed={seed} DONE val_mse={best_val:.6e} time={(time.time()-t0)/60:.1f}m')

print('DONE')
