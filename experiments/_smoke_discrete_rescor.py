"""Smoke test: DiscreteRescor on synthetic tokens — quick validation."""
import sys, time, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wmca.modules.discrete_rescor import DiscreteRescor

# ── Synthetic data ──
N, V, H, W, n_actions = 2000, 512, 16, 16, 18
rng = torch.Generator().manual_seed(42)
tokens = torch.randint(0, V, (N, H, W), generator=rng)
next_tokens = (tokens + torch.randint(1, 5, (N, H, W), generator=rng)) % V
actions = torch.randint(0, n_actions, (N,), generator=rng)

n_val = int(N * 0.15)
t_tr, t_v = tokens[:N-n_val], tokens[N-n_val:]
n_tr, n_v = next_tokens[:N-n_val], next_tokens[N-n_val:]
a_tr, a_v = actions[:N-n_val], actions[N-n_val:]

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device} | {len(t_tr)} train / {len(t_v)} val | Grid: {H}x{W} V={V} A={n_actions}")

# ── Model ──
model = DiscreteRescor(vocab_size=V, n_actions=n_actions, embed_dim=16, hidden_ch=4,
                        cml_K=8, cml_steps=3, seed=42).to(device).float()
pc = model.param_count()
print(f"Params: {pc['trained']:,} + {pc.get('frozen',0):,} frozen = {pc['trained']+pc.get('frozen',0):,}")

# ── Training loop ──
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
train_losses, val_losses = [], []

t0 = time.time()
for ep in range(10):
    model.train()
    perm = torch.randperm(len(t_tr))
    total = 0.0
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        logits = model(t_tr[idx].to(device), a_tr[idx].to(device))
        loss = criterion(logits, n_tr[idx].to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    train_losses.append(total / max(len(t_tr)//32, 1))

    model.eval()
    with torch.no_grad():
        v_logits = model(t_v[:64].to(device), a_v[:64].to(device))
        v_loss = criterion(v_logits, n_v[:64].to(device))
    val_losses.append(v_loss.item())
    print(f"  ep {ep+1:2d}/10: train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

elapsed = time.time() - t0

# ── Checks ──
loss_ok = train_losses[-1] < train_losses[0] * 0.9
no_nan = not any(np.isnan(train_losses)) and not any(np.isnan(val_losses))

# Accuracy check
model.eval()
with torch.no_grad():
    logits = model(t_v[:64].to(device), a_v[:64].to(device))
    pred = logits.argmax(dim=1)
    acc = (pred.cpu() == n_v[:64]).float().mean().item()

print(f"\nTrain: {train_losses[0]:.4f} → {train_losses[-1]:.4f}  {'✓' if loss_ok else '✗'}")
print(f"Val:   {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
print(f"Acc:   {acc:.4f}  (random baseline: ~{1/V:.5f})")
print(f"Time:  {elapsed:.1f}s")
print(f"\n{'SMOKE PASS ✓' if loss_ok and no_nan else 'SMOKE FAIL ✗'}")
