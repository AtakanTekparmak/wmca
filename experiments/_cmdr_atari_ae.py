"""Train grid-native AE on Atari frames. Batch=8 to fit 3GB total cap."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')
DEVICE = torch.device('mps')

class GridAE(nn.Module):
    def __init__(self, in_ch=4, hid=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(in_ch, hid, 3, padding=1), nn.ReLU(), nn.Conv2d(hid, 1, 3, padding=1), nn.Sigmoid())
        self.dec = nn.Sequential(nn.Conv2d(1, hid, 3, padding=1), nn.ReLU(), nn.Conv2d(hid, in_ch, 3, padding=1), nn.Sigmoid())
    def encode(self, x): return self.enc(x)
    def forward(self, x): return self.dec(self.enc(x))

for game in ['pong', 'breakout']:
    frames = np.load(f'experiments/atari_data/{game}_frames.npy')
    n = len(frames)
    n_train = int(n * 0.85)
    X = torch.from_numpy(frames[:n_train]).float().to(DEVICE)
    Xv = torch.from_numpy(frames[n_train:]).float().to(DEVICE)
    model = GridAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_psnr = -1
    for ep in range(10):
        model.train()
        perm = torch.randperm(len(X), device=DEVICE)
        total = 0.0
        for i in range(0, len(perm), 8):
            idx = perm[i:i+8]
            pred = model(X[idx])
            loss = nn.functional.mse_loss(pred, X[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            vp = model(Xv[:64])
            mse = nn.functional.mse_loss(vp, Xv[:64]).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 99
        if psnr > best_psnr: best_psnr = psnr
        if ep % 3 == 0: print(f'{game} ep {ep+1}/10 loss={total/max(len(perm)//8,1):.4f} psnr={psnr:.1f}')
    torch.save(model.state_dict(), f'experiments/atari_data/frame_encoder_{game}.pt')
    print(f'{game} saved, best PSNR={best_psnr:.1f}')

    # Encode latents
    model.eval()
    lats, next_lats = [], []
    all_f = torch.from_numpy(frames).float()
    all_nf = torch.from_numpy(np.load(f'experiments/atari_data/{game}_next_frames.npy')).float()
    with torch.no_grad():
        for i in range(0, len(all_f), 64):
            lats.append(model.encode(all_f[i:i+64].to(DEVICE)).cpu().numpy())
            next_lats.append(model.encode(all_nf[i:i+64].to(DEVICE)).cpu().numpy())
    lats = np.concatenate(lats)
    next_lats = np.concatenate(next_lats)
    np.save(f'experiments/atari_data/{game}_latents.npy', lats)
    np.save(f'experiments/atari_data/{game}_next_latents.npy', next_lats)
    print(f'{game} latents: {lats.shape}')
print('DONE')
