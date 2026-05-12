"""Mamba rollout probe — K=4 temporal context, autoregressive roll."""
import sys, numpy as np, torch
sys.path.insert(0, 'src')
from wmca.modules.hybrid import ResCorMamba

DEVICE = torch.device('mps')
HORIZONS = [15, 50, 100]
SEEDS = [42, 43, 44]
K = 4

def rollout(model, latents, start, horizon):
    """Roll from start index. K=4 buffer seeded from ground truth, then autoregressive."""
    model.eval()
    # Seed buffer with K ground-truth frames
    buffer = latents[start:start+K].copy()  # (K, 1, H, W)
    mses = []
    with torch.no_grad():
        for t in range(horizon):
            x = torch.from_numpy(buffer).float().unsqueeze(0).to(DEVICE)  # (1, K, 1, H, W)
            pred = model(x).squeeze(0).cpu().numpy()  # (1, H, W)
            gt = latents[start + K + t]
            mses.append(float(np.mean((pred - gt) ** 2)))
            # Shift buffer: drop oldest, append prediction
            buffer = np.concatenate([buffer[1:], pred[np.newaxis]], axis=0)
    return mses

for game in ['pong', 'breakout']:
    latents = np.load(f'experiments/atari_data/{game}_latents.npy')
    n_test = 3750
    test_start = len(latents) - n_test
    lt = latents[test_start:]
    H, W = lt.shape[2], lt.shape[3]
    print(f'\n{"="*50}\n{game.upper()} Mamba Rollout | {len(lt)} test | {H}x{W}\n{"="*50}')

    for seed in SEEDS:
        ckpt = torch.load(f'experiments/atari_data/mamba_{game}_seed{seed}.pt', map_location='cpu')
        model = ResCorMamba(in_channels=1, out_channels=1, cml_K=32, context_k=K, expand=2, zero_init_out=False, use_sigmoid=True, seed=seed).to(DEVICE)
        model.load_state_dict(ckpt)

        starts = np.linspace(0, len(lt) - K - 150, 20, dtype=int)
        all_step1 = []
        h_mses = {h: [] for h in HORIZONS}

        for s in starts:
            mses = rollout(model, lt, s, 100)
            if len(mses) >= 100:
                all_step1.append(mses[0])
                for h in HORIZONS:
                    h_mses[h].append(mses[h - 1])

        step1_med = np.median(all_step1)
        print(f'  seed={seed} step1={step1_med:.2e}')
        for h in HORIZONS:
            m = np.median(h_mses[h])
            r = m / step1_med if step1_med > 0 else float('nan')
            print(f'    H={h:3d}: MSE={m:.2e} ratio={r:.1f}x')

    # Summary across seeds
    print(f'\n{game} SUMMARY:')
    for h in HORIZONS:
        ratios = []
        for seed in SEEDS:
            ckpt = torch.load(f'experiments/atari_data/mamba_{game}_seed{seed}.pt', map_location='cpu')
            model = ResCorMamba(in_channels=1, out_channels=1, cml_K=32, context_k=K, expand=2, zero_init_out=False, use_sigmoid=True, seed=seed).to(DEVICE)
            model.load_state_dict(ckpt)
            starts = np.linspace(0, len(lt) - K - 150, 10, dtype=int)
            step1s = []; h_ms = []
            for s in starts:
                mses = rollout(model, lt, s, h)
                if len(mses) >= h:
                    step1s.append(mses[0]); h_ms.append(mses[h-1])
            if step1s and h_ms:
                ratios.append(np.median(h_ms) / np.median(step1s))
        if ratios:
            print(f'  H={h:3d}: median ratio={np.median(ratios):.1f}x')
print('\nDONE')
