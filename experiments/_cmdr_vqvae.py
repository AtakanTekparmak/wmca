"""VQ-VAE quick smoke training on Crafter frames. 3 epochs, 5000 samples."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')
from wmca.modules.vqvae import VQVAE

device = torch.device('mps')
frames = np.load('experiments/crafter_data/frames.npy', mmap_mode='r')
N = len(frames)
n_train = int(N * 0.85)
print(f'Frames: {N}, train: {n_train}')

model = VQVAE(num_embeddings=256, embed_dim=64, in_channels=3).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(3):
    model.train()
    total_r, total_v, n_b = 0, 0, 0
    perm = torch.randperm(n_train)
    for i in range(0, min(n_train, 5000), 8):
        idx = perm[i:i+8]
        xb = torch.from_numpy(frames[idx.numpy()].copy()).float().to(device)
        recon, indices, vq_loss = model(xb)
        loss = nn.functional.mse_loss(recon, xb) + vq_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_r += nn.functional.mse_loss(recon, xb).item()
        total_v += vq_loss.item()
        n_b += 1
    print(f'epoch {ep+1}/3 recon={total_r/max(n_b,1):.6f} vq={total_v/max(n_b,1):.6f}')

torch.save(model.state_dict(), 'experiments/crafter_data/vqvae_smoke.pt')
print('VQ-VAE saved to experiments/crafter_data/vqvae_smoke.pt')
