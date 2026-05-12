"""Re-encode all delta embeddings using saved DeltaEncoder (post-bugfix)."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')

DEVICE = torch.device('mps')
D = 64

class FrameBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
        )
    def forward(self, x): return self.net(x)

class DeltaEncoder(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.branch = FrameBranch()
        self.fusion = nn.Conv2d(64, D, 1)
    def forward(self, f0, f1):
        b0 = self.branch(f0); b1 = self.branch(f1)
        return self.fusion(torch.cat([b0, b1], dim=1))

encoder = DeltaEncoder(D).to(DEVICE)
encoder.load_state_dict(torch.load('experiments/crafter_data/delta_encoder.pt', map_location=DEVICE))
encoder.eval()

frames = np.load('experiments/crafter_data/frames.npy', mmap_mode='r')
N = len(frames)
print(f'Encoding deltas for {N} frames...')

all_deltas = []
for i in range(0, N - 1, 64):
    end = min(i + 64, N - 1)
    f0 = torch.from_numpy(frames[i:end].copy()).float().to(DEVICE)
    f1 = torch.from_numpy(frames[i+1:end+1].copy()).float().to(DEVICE)
    with torch.no_grad():
        all_deltas.append(encoder(f0, f1).cpu().numpy())
all_deltas = np.concatenate(all_deltas)
np.save('experiments/crafter_data/delta_embeddings.npy', all_deltas)
print(f'Delta embeddings saved: {all_deltas.shape}')
