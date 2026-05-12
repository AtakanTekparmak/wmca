"""Rollout probe for Architecture A (continuous) and B (discrete) delta models."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')
from wmca.modules.hybrid import CML2DMultiR
from wmca.modules.vqvae import VectorQuantizer

DEVICE = torch.device('mps')
D = 64; V = 512; HORIZONS = [15, 50, 100]; SEEDS = [42, 43, 44]

# Load shared data
deltas = np.load('experiments/crafter_data/delta_embeddings.npy', mmap_mode='r')
actions = np.load('experiments/crafter_data/actions.npy')
tokens = np.load('experiments/crafter_data/delta_tokens.npy', mmap_mode='r')

N = len(deltas) - 1
test_start = N - 3750
d_test = deltas[test_start:]
a_test = actions[test_start:]
t_test = tokens[test_start:]
print(f'Test: {len(d_test)} deltas, {len(t_test)} tokens')

# Arch A model class (must match training)
class RescorDeltaCont(nn.Module):
    def __init__(self, D=64, hid=16, K=32, seed=42):
        super().__init__()
        self.action_embed = nn.Embedding(17, D)
        self.input_proj = nn.Conv2d(D*2, 1, 1)
        self.cml = CML2DMultiR(in_channels=1, K=K, steps=15, seed=seed, gate_mode='uniform')
        self.nca = nn.Sequential(nn.Conv2d(2, hid, 3, 1, 1), nn.ReLU(), nn.Conv2d(hid, D, 1))
        self.cml_proj = nn.Conv2d(1, D, 1)
    def forward(self, de, act):
        a = self.action_embed(act).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,16,16)
        drive = torch.sigmoid(self.input_proj(torch.cat([de, a], dim=1)))
        cml_out = self.cml(drive)
        return self.cml_proj(cml_out) + self.nca(torch.cat([drive, cml_out], dim=1))

# === ARCHITECTURE A ROLLOUT ===
print('\n=== ARCHITECTURE A — Continuous Delta ===')
for seed in SEEDS:
    ckpt = torch.load(f'experiments/crafter_data/rescor_delta_cont_seed{seed}.pt', map_location='cpu')
    model = RescorDeltaCont(seed=seed).to(DEVICE)
    model.load_state_dict(ckpt); model.eval()

    starts = np.linspace(0, len(d_test) - 150, 10, dtype=int)
    step1_mses = []; h_mses = {h: [] for h in HORIZONS}

    for s in starts:
        delta_t = d_test[s].copy()  # (D, 16, 16)
        mses = []
        with torch.no_grad():
            for t in range(100):
                act = torch.tensor([a_test[s + t]], device=DEVICE, dtype=torch.long)
                de = torch.from_numpy(delta_t).float().unsqueeze(0).to(DEVICE)
                pred = model(de, act).squeeze(0).cpu().numpy()
                gt = d_test[s + t + 1]
                mses.append(float(np.mean((pred - gt) ** 2)))
                delta_t = pred
        if len(mses) >= 100:
            step1_mses.append(mses[0])
            for h in HORIZONS: h_mses[h].append(mses[h-1])

    s1 = np.median(step1_mses)
    print(f'  seed={seed} step1_mse={s1:.2e}')
    for h in HORIZONS:
        m = np.median(h_mses[h]); r = m/s1 if s1>0 else float('nan')
        print(f'    H={h:3d}: MSE={m:.2e} ratio={r:.1f}x')

# === ARCHITECTURE B ROLLOUT ===
print('\n=== ARCHITECTURE B — Discrete Delta ===')
from wmca.modules.discrete_rescor import DiscreteRescor
vq = VectorQuantizer(num_embeddings=V, embedding_dim=D)
vq.load_state_dict(torch.load('experiments/crafter_data/delta_vq.pt', map_location='cpu'))
vq.eval()

for seed in SEEDS:
    ckpt = torch.load(f'experiments/crafter_data/rescor_delta_disc_seed{seed}.pt', map_location='cpu')
    model = DiscreteRescor(vocab_size=V, n_actions=17, embed_dim=D, seed=seed).to(DEVICE)
    model.load_state_dict(ckpt); model.eval()

    starts = np.linspace(0, len(t_test) - 150, 10, dtype=int)
    step1_accs = []; h_accs = {h: [] for h in HORIZONS}

    for s in starts:
        tok_t = t_test[s].copy()
        accs = []
        with torch.no_grad():
            for t in range(100):
                act = torch.tensor([a_test[s + t]], device=DEVICE, dtype=torch.long)
                tk = torch.from_numpy(tok_t).long().unsqueeze(0).to(DEVICE)
                logits = model(tk, act)  # (1, V, 16, 16)
                pred_tok = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                gt_tok = t_test[s + t + 1]
                accs.append(float(np.mean(pred_tok == gt_tok)))
                tok_t = pred_tok
        if len(accs) >= 100:
            step1_accs.append(accs[0])
            for h in HORIZONS: h_accs[h].append(accs[h-1])

    s1 = np.median(step1_accs)
    print(f'  seed={seed} step1_acc={s1:.3f}')
    for h in HORIZONS:
        a = np.median(h_accs[h])
        print(f'    H={h:3d}: acc={a:.3f}')

print('\nDONE')
