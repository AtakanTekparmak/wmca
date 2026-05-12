"""Train rescor_mamba_rand on Atari latents. K=4 temporal context. ~43h total."""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')
from wmca.modules.hybrid import ResCorMamba

DEVICE = torch.device('mps')
CRITERION = nn.MSELoss()
EPOCHS = 100
BATCH = 8
SEEDS = [42, 43, 44]
K = 4  # temporal context

def build_k4_sequences(latents_path):
    """(N, 1, H, W) -> X: (N-K+1, K, 1, H, W), Y: (N-K+1, 1, H, W)"""
    latents = np.load(latents_path)
    N = len(latents)
    X_list, Y_list = [], []
    for i in range(N - K):
        X_list.append(latents[i:i+K])  # (K, 1, H, W)
        Y_list.append(latents[i+K])   # (1, H, W)
    X = np.stack(X_list)  # (N-K+1, K, 1, H, W)
    Y = np.stack(Y_list)  # (N-K+1, 1, H, W)
    return X, Y

for game in ['pong', 'breakout']:
    print(f"\n{'='*50}\n{game.upper()} — ResCorMamba K=4\n{'='*50}")
    X_all, Y_all = build_k4_sequences(f'experiments/atari_data/{game}_latents.npy')
    N = len(X_all)
    n_train = int(N * 0.7)
    n_val = int(N * 0.15)
    print(f'{N} K4 sequences, {n_train} train, {n_val} val')
    Hg, Wg = Y_all.shape[2], Y_all.shape[3]
    print(f'Grid: {Hg}x{Wg}')

    X_tr = torch.from_numpy(X_all[:n_train]).float().to(DEVICE)
    Y_tr = torch.from_numpy(Y_all[:n_train]).float().to(DEVICE)
    X_val = torch.from_numpy(X_all[n_train:n_train+n_val]).float().to(DEVICE)
    Y_val = torch.from_numpy(Y_all[n_train:n_train+n_val]).float().to(DEVICE)

    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ResCorMamba(
            in_channels=1, out_channels=1, cml_K=32, context_k=K,
            expand=2, zero_init_out=False, use_sigmoid=True, seed=seed
        ).to(DEVICE)
        pc = model.param_count()
        opt = torch.optim.Adam(model.parameters(), lr=1.4e-3)
        best_val = float('inf')
        best_state = None
        t0 = time.time()

        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(X_tr), device=DEVICE)
            total_loss = 0.0
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i+BATCH]
                pred = model(X_tr[idx])
                loss = CRITERION(pred, Y_tr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_mse = CRITERION(model(X_val[:256]), Y_val[:256]).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if ep % 25 == 0 or ep == EPOCHS - 1:
                elapsed = (time.time() - t0) / 60
                eta = elapsed / (ep + 1) * (EPOCHS - ep - 1)
                print(f'  {game} seed={seed} ep={ep+1:3d}/{EPOCHS} train={total_loss/max(len(perm)//BATCH,1):.6f} val={val_mse:.6e} [{elapsed:.0f}m elapsed, {eta:.0f}m eta]')

        if best_state: model.load_state_dict(best_state)
        torch.save(model.state_dict(), f'experiments/atari_data/mamba_{game}_seed{seed}.pt')
        elapsed = (time.time() - t0) / 60
        print(f'  {game} seed={seed} DONE val_mse={best_val:.6e} time={elapsed:.0f}m')

print('\nALL DONE')
