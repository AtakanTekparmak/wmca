"""Train VQ on delta embeddings — Architecture B prep."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')
from wmca.modules.vqvae import VectorQuantizer

DEVICE = torch.device('mps')
V = 512; D = 64; EPOCHS = 10; BATCH = 8

deltas = np.load('experiments/crafter_data/delta_embeddings.npy', mmap_mode='r')
N = len(deltas); n_train = int(N * 0.85)
print(f'Delta embeddings: {deltas.shape}, train={n_train}')

vq = VectorQuantizer(num_embeddings=V, embedding_dim=D, commitment_cost=0.25).to(DEVICE)
opt = torch.optim.Adam(vq.parameters(), lr=1e-3) if list(vq.parameters()) else None

for ep in range(EPOCHS):
    perm = torch.randperm(n_train)
    total_loss = 0.0; n_b = 0
    for i in range(0, min(n_train, 20000), BATCH):
        idx = perm[i:i+BATCH]
        x = torch.from_numpy(deltas[idx].copy()).float().to(DEVICE)
        z_q, indices, loss = vq(x)
        if opt: opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n_b += 1
    usage = len(indices.unique()) / V
    print(f'ep {ep+1:2d}/{EPOCHS} loss={total_loss/max(n_b,1):.4f} usage={usage:.1%}')

torch.save(vq.state_dict(), 'experiments/crafter_data/delta_vq.pt')
print(f'VQ saved. Usage: {float(len(indices.unique()))/V:.1%}')

# Encode all deltas to tokens
vq.eval()
all_tokens = []
for i in range(0, N, 64):
    x = torch.from_numpy(deltas[i:i+64].copy()).float().to(DEVICE)
    with torch.no_grad():
        _, indices, _ = vq(x)
    all_tokens.append(indices.cpu().numpy())
all_tokens = np.concatenate(all_tokens)
np.save('experiments/crafter_data/delta_tokens.npy', all_tokens.astype(np.int16))
np.save('experiments/crafter_data/delta_next_tokens.npy', all_tokens[1:].astype(np.int16))
print(f'Tokens saved: {all_tokens.shape}, range=[{all_tokens.min()},{all_tokens.max()}]')
