"""Train DiscreteDeltaRescor seeds 43+44 on Architecture B delta tokens."""
import sys, numpy as np, torch, torch.nn as nn, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wmca.modules.discrete_rescor import DiscreteRescor

DEVICE = torch.device('mps')
VOCAB = 256
EPOCHS = 50
BATCH = 16
SEEDS = [43, 44]
LR = 1e-3

tokens = np.load('experiments/crafter_data/delta_tokens.npy')
next_tokens = np.load('experiments/crafter_data/delta_next_tokens.npy')
actions = np.load('experiments/crafter_data/actions.npy')[:len(tokens)]
N = len(tokens); n_train = int(N * 0.85)
t_tr = torch.from_numpy(tokens[:n_train]).long()
n_tr = torch.from_numpy(next_tokens[:n_train]).long()
a_tr = torch.from_numpy(actions[:n_train]).long()
t_v = torch.from_numpy(tokens[n_train:]).long()
n_v = torch.from_numpy(next_tokens[n_train:]).long()
a_v = torch.from_numpy(actions[n_train:]).long()
n_actions = int(actions.max()) + 1

for seed in SEEDS:
    print(f'\n--- seed={seed} ---')
    torch.manual_seed(seed)
    model = DiscreteRescor(vocab_size=VOCAB, n_actions=n_actions, embed_dim=32,
                            hidden_ch=8, cml_K=32, seed=seed).to(DEVICE).float()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_val = float('inf'); best_state = None; t0 = time.time()
    
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss, nb = 0.0, 0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i+BATCH]
            logits = model(t_tr[idx].to(DEVICE), a_tr[idx].to(DEVICE))
            loss = criterion(logits, n_tr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            v_logits = model(t_v[:512].to(DEVICE), a_v[:512].to(DEVICE))
            v_loss = criterion(v_logits, n_v[:512].to(DEVICE)).item()
            acc = (v_logits.argmax(1).cpu() == n_v[:512]).float().mean().item()
        if v_loss < best_val: best_val = v_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f'  ep {ep+1:2d}/{EPOCHS} loss={total_loss/max(nb,1):.4f} ce={v_loss:.4f} acc={acc:.4f} [{time.time()-t0:.0f}s]')
    
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), f'experiments/crafter_data/rescor_delta_disc_seed{seed}.pt')
    print(f'  DONE best_ce={best_val:.4f} time={time.time()-t0:.0f}s')
print('DONE')
