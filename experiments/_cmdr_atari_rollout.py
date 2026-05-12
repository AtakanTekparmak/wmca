"""Atari rollout probe v2 — fixed ratios."""
import sys, numpy as np, torch
sys.path.insert(0, 'src')
from wmca.modules.hybrid import ResidualCorrectionWM

DEVICE = torch.device('mps')
N_ACTIONS = 3
HORIZONS = [15, 50, 100]
SEEDS = [42, 43, 44]

def rollout(model, latents, actions, start, horizon, H, W):
    """Roll from start index for horizon steps. Returns [step1_mse, ..., stepH_mse]"""
    model.eval()
    current = latents[start].copy()
    mses = []
    with torch.no_grad():
        for t in range(horizon):
            act_val = (actions[start + t] + 1.0) / N_ACTIONS
            act_field = np.full((1, H, W), act_val, dtype=np.float32)
            x = np.concatenate([current, act_field], axis=0)
            x_t = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
            pred = model(x_t).squeeze(0).cpu().numpy()
            gt = latents[start + t + 1]
            mses.append(float(np.mean((pred - gt) ** 2)))
            current = np.clip(pred, 0.0, 1.0).astype(np.float32)
    return mses

for game in ['pong', 'breakout']:
    latents = np.load(f'experiments/atari_data/{game}_latents.npy')
    actions = np.load(f'experiments/atari_data/{game}_actions.npy').astype(np.int64)
    n = len(latents)
    test_start = n - 3750
    lt = latents[test_start:]
    at = actions[test_start:]
    H, W = lt.shape[2], lt.shape[3]
    print(f'\n{"="*50}\n{game.upper()} | {len(lt)} test | {H}x{W}\n{"="*50}')

    for seed in SEEDS:
        ckpt = torch.load(f'experiments/atari_data/rescor_{game}_seed{seed}.pt', map_location='cpu')
        model = ResidualCorrectionWM(in_channels=2, out_channels=1, cml_gate='multi_r_uniform', cml_K=32, use_sigmoid=True, seed=seed).to(DEVICE)
        model.load_state_dict(ckpt)

        # 20 rollouts from different starts
        starts = np.linspace(0, len(lt) - 150, 20, dtype=int)
        all_step1 = []
        h_mses = {h: [] for h in HORIZONS}

        for s in starts:
            mses = rollout(model, lt, at, s, 100, H, W)
            if len(mses) >= 100:
                all_step1.append(mses[0])
                for h in HORIZONS:
                    h_mses[h].append(mses[h - 1])

        step1_med = np.median(all_step1)
        print(f'  seed={seed} step1_mse={step1_med:.2e}')
        for h in HORIZONS:
            m = np.median(h_mses[h])
            r = m / step1_med if step1_med > 0 else float('nan')
            print(f'    H={h:3d}: MSE={m:.2e} ratio={r:.1f}x')

    # Median across seeds
    print(f'\n{game} SUMMARY (median across 3 seeds):')
    for h in HORIZONS:
        ms = []
        for seed in SEEDS:
            ckpt = torch.load(f'experiments/atari_data/rescor_{game}_seed{seed}.pt', map_location='cpu')
            model = ResidualCorrectionWM(in_channels=2, out_channels=1, cml_gate='multi_r_uniform', cml_K=32, use_sigmoid=True, seed=seed).to(DEVICE)
            model.load_state_dict(ckpt)
            starts = np.linspace(0, len(lt) - 150, 10, dtype=int)
            mses_all = []
            step1s = []
            for s in starts:
                mses = rollout(model, lt, at, s, 100, H, W)
                if len(mses) >= h:
                    mses_all.append(mses[h - 1])
                    step1s.append(mses[0])
            if mses_all and step1s:
                ms.append(np.median(mses_all) / np.median(step1s) if np.median(step1s) > 0 else float('nan'))
        if ms:
            print(f'  H={h:3d}: median ratio={np.median(ms):.1f}x')

print('\nDONE')
