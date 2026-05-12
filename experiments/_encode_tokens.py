"""Encode Crafter frames to VQ-VAE tokens using existing checkpoint."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wmca.modules.vqvae import VQVAE

DATA_DIR = Path(__file__).parent / "crafter_data"
CKPT_PATH = Path(__file__).parent / "vqvae_checkpoints" / "vqvae_best.pt"
BATCH_SIZE = 64
device = torch.device("mps")

# Load model with same config as training
model = VQVAE(num_embeddings=512, embed_dim=64, in_channels=3).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=False))
model.eval()
print(f"Model loaded from {CKPT_PATH}")

# Load frames (mmap)
frames = np.load(DATA_DIR / "frames.npy", mmap_mode="r")
N = len(frames)
print(f"Frames: {N}, shape={frames.shape}")

# Encode all frames to token indices
all_indices = []
with torch.no_grad():
    for i in range(0, N, BATCH_SIZE):
        batch = torch.from_numpy(frames[i:i+BATCH_SIZE].copy()).float().to(device)
        _, tok_idx, _ = model(batch)
        all_indices.append(tok_idx.cpu())
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  {i}/{N}")

all_indices = torch.cat(all_indices, dim=0)  # (N, H, W)
print(f"Tokens shape: {all_indices.shape}")

# Compute usage
usage = len(torch.unique(all_indices)) / 512
print(f"Codebook usage: {usage:.1%}")

# Save tokens (shift by 1 for prediction target)
tokens = all_indices[:-1].numpy()
next_tokens = all_indices[1:].numpy()
np.save(DATA_DIR / "tokens.npy", tokens)
np.save(DATA_DIR / "next_tokens.npy", next_tokens)
print(f"Saved: tokens={tokens.shape}, next_tokens={next_tokens.shape}")
print("DONE")
