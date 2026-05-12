"""Train rescor_rens on Atari latents. 3 seeds x 100 epochs x 2 games."""
import sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, 'src')
from wmca.modules.hybrid import ResidualCorrectionWM

DEVICE = torch.device('mps')
CRITERION = nn.MSELoss()
EPOCHS = 100
BATCH = 8
SEEDS = [42, 43, 44]
N_ACTIONS = 3  # Pong/Breakout both 3 actions

def make_action_field(actions, n_actions, H, W):
    """(N,) actions -> (N, 1, H, W) float action field."""
    vals = (actions + 1.0) / n_actions
    return np.tile(vals[:, None, None, None], (1, 1, H, W)).astype(np.float32)

def build_data(latents_path, next_latents_path, actions_path, n_actions):
    latents = np.load(latents_path)
    next_latents = np.load(next_latents_path)
    actions = np.load(actions_path)[:len(latents)]
    H, W = latents.shape[2], latents.shape[3]
    act_field = make_action_field(actions, n_actions, H, W)
    X = np.concatenate([latents, act_field], axis=1)  # (N, 2, H, W)
    Y = next_latents  # (N, 1, H, W)
    return X, Y

for game in ['pong', 'breakout']:
    print(f"\n{'='*50}\n{game.upper()}\n{'='*50}")
    X_all, Y_all = build_data(
        f'experiments/atari_data/{game}_latents.npy',
        f'experiments/atari_data/{game}_next_latents.npy',
        f'experiments/atari_data/{game}_actions.npy',
        N_ACTIONS
    )
    N = len(X_all)
    n_train = int(N * 0.7)
    n_val = int(N * 0.15)
    X_tr = torch.from_numpy(X_all[:n_train]).float().to(DEVICE)
    Y_tr = torch.from_numpy(Y_all[:n_train]).float().to(DEVICE)
    X_val = torch.from_numpy(X_all[n_train:n_train+n_val]).float().to(DEVICE)
    Y_val = torch.from_numpy(Y_all[n_train:n_train+n_val]).float().to(DEVICE)
    print(f'{N} samples, {n_train} train, {n_val} val, grid={Y_all.shape[2]}x{Y_all.shape[3]}')

    best_per_seed = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ResidualCorrectionWM(
            in_channels=2, out_channels=1,
            cml_gate='multi_r_uniform', cml_K=32,
            use_sigmoid=True, seed=seed
        ).to(DEVICE)
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
                val_pred = model(X_val)
                val_mse = CRITERION(val_pred, Y_val).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if ep % 20 == 0 or ep == EPOCHS - 1:
                eta = (time.time() - t0) / (ep + 1) * (EPOCHS - ep - 1)
                print(f'  seed={seed} ep={ep+1:3d}/{EPOCHS} train={total_loss/max(len(perm)//BATCH,1):.6f} val_mse={val_mse:.6e} eta={eta/60:.0f}m')

        if best_state: model.load_state_dict(best_state)
        torch.save(model.state_dict(), f'experiments/atari_data/rescor_{game}_seed{seed}.pt')
        best_per_seed[seed] = best_val
        print(f'  seed={seed} DONE best_val_mse={best_val:.6e} time={(time.time()-t0)/60:.1f}m')

    print(f'\n{game} RESULTS:')
    for s in SEEDS:
        print(f'  seed={s}: {best_per_seed[s]:.6e}')
    print(f'  median: {np.median(list(best_per_seed.values())):.6e}')

print('\nDONE')
