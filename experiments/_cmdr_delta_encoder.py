
"""Train DeltaEncoder + DeltaDecoder on Crafter frame pairs."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')

DEVICE = torch.device('mps')
D = 64
EPOCHS = 10
BATCH = 8

# Load frames
frames = np.load('experiments/crafter_data/frames.npy', mmap_mode='r')
N = len(frames)
n_train = int(N * 0.85)
F0 = frames[:n_train-1]
F1 = frames[1:n_train]
print(f'Frames: {N}, train pairs: {len(F0)}')

# DeltaEncoder: siamese conv branches
class FrameBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
        )
    def forward(self, x): return self.net(x)  # (B,3,64,64)->(B,32,16,16)

class DeltaEncoder(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.branch = FrameBranch()
        self.fusion = nn.Conv2d(64, D, 1)  # concat 32+32 -> D
    def forward(self, f0, f1):
        b0 = self.branch(f0); b1 = self.branch(f1)
        return self.fusion(torch.cat([b0, b1], dim=1))

class DeltaDecoder(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.down = nn.AvgPool2d(4)
        self.net = nn.Sequential(
            nn.Conv2d(3 + D, 64, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )
    def forward(self, f0, delta_emb):
        f0_down = self.down(f0)  # (B,3,64,64)->(B,3,16,16)
        return self.net(torch.cat([f0_down, delta_emb], dim=1))

encoder = DeltaEncoder(D).to(DEVICE)
decoder = DeltaDecoder(D).to(DEVICE)
params = sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
print(f'Params: encoder={sum(p.numel() for p in encoder.parameters()):,} decoder={sum(p.numel() for p in decoder.parameters()):,} total={params:,}')

opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
best_psnr = -1

for ep in range(EPOCHS):
    encoder.train(); decoder.train()
    perm = torch.randperm(len(F0))
    total_loss = 0.0; n_b = 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        f0 = torch.from_numpy(F0[idx].copy()).float().to(DEVICE)
        f1 = torch.from_numpy(F1[idx].copy()).float().to(DEVICE)
        delta = encoder(f0, f1)
        recon = decoder(f0, delta)
        loss = nn.functional.mse_loss(recon, f1)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n_b += 1

    # Validation
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        f0v = torch.from_numpy(frames[n_train:n_train+64].copy()).float().to(DEVICE)
        f1v = torch.from_numpy(frames[n_train+1:n_train+65].copy()).float().to(DEVICE)
        recon_v = decoder(f0v, encoder(f0v, f1v))
        mse = nn.functional.mse_loss(recon_v, f1v).item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 99
    if psnr > best_psnr: best_psnr = psnr
    print(f'ep {ep+1:2d}/{EPOCHS} loss={total_loss/max(n_b,1):.4f} val_psnr={psnr:.1f} dB')

torch.save(encoder.state_dict(), 'experiments/crafter_data/delta_encoder.pt')
torch.save(decoder.state_dict(), 'experiments/crafter_data/delta_decoder.pt')
print(f'Saved: best PSNR={best_psnr:.1f} dB')

# Encode all deltas (skip last partial batch)
encoder.eval()
all_deltas = []
for i in range(0, N - 66, 64):
    f0 = torch.from_numpy(frames[i:i+64].copy()).float().to(DEVICE)
    f1 = torch.from_numpy(frames[i+1:i+65].copy()).float().to(DEVICE)
    with torch.no_grad():
        all_deltas.append(encoder(f0, f1).cpu().numpy())
all_deltas = np.concatenate(all_deltas)
np.save('experiments/crafter_data/delta_embeddings.npy', all_deltas)
print(f'Delta embeddings saved: {all_deltas.shape}')
