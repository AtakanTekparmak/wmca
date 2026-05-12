"""Probe Crafter frame deltas — can a single embedding capture the support?"""
import numpy as np

frames = np.load('experiments/crafter_data/frames.npy', mmap_mode='r')
N = len(frames) - 1
print(f'Crafter frames: {frames.shape}, {N} consecutive pairs')

# Per-pixel absolute delta
deltas = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
print(f'Delta shape: {deltas.shape}')

# Global stats (per-pixel)
print(f'\nPer-pixel delta stats:')
print(f'  mean={deltas.mean():.6f} std={deltas.std():.6f}')
print(f'  min={deltas.min():.6f} max={deltas.max():.6f}')
for p in [50, 90, 95, 99, 99.9]:
    print(f'  P{p}: {np.percentile(deltas, p):.6f}')

# Frame-level MSE (single scalar per frame pair)
frame_mse = np.mean((deltas) ** 2, axis=(1,2,3))
print(f'\nFrame-level MSE (single scalar):')
print(f'  mean={frame_mse.mean():.6f} std={frame_mse.std():.6f}')
print(f'  min={frame_mse.min():.6f} max={frame_mse.max():.6f}')
for p in [50, 90, 95, 99]:
    print(f'  P{p}: {np.percentile(frame_mse, p):.6f}')

# Sparsity: what fraction of pixels change per frame?
changed = (deltas > 1e-4).mean(axis=(1,2,3))
print(f'\nFraction of pixels changed per frame:')
print(f'  mean={changed.mean():.4f} ({changed.mean()*64*64*3:.0f} of 12288 pixels)')
print(f'  P50={np.percentile(changed, 50):.4f} P90={np.percentile(changed, 90):.4f}')

# Bimodality check: action steps vs no-action steps
# Use the actual actions
actions = np.load('experiments/crafter_data/actions.npy')
noop_mask = actions[:-1] == 0  # action=0 is NOOP
action_mask = actions[:-1] != 0

print(f'\nNOOP frames ({noop_mask.sum()}): MSE mean={frame_mse[noop_mask].mean():.6f} std={frame_mse[noop_mask].std():.6f}')
print(f'Action frames ({action_mask.sum()}): MSE mean={frame_mse[action_mask].mean():.6f} std={frame_mse[action_mask].std():.6f}')

# 16x16 patch-level MSE (matching our spatial grid)
H, W = 16, 16
ph, pw = 64 // H, 64 // W  # 4x4 patches
patch_mse = np.zeros((N, H, W))
for h in range(H):
    for w in range(W):
        patch = deltas[:, :, h*ph:(h+1)*ph, w*pw:(w+1)*pw]
        patch_mse[:, h, w] = np.mean(patch ** 2, axis=(1,2,3))

print(f'\n16x16 patch-level MSE:')
print(f'  mean={patch_mse.mean():.6f} std={patch_mse.std():.6f}')
print(f'  P50={np.percentile(patch_mse, 50):.6f} P95={np.percentile(patch_mse, 95):.6f}')

# Conclusion
print(f'\n=== CONCLUSION ===')
if frame_mse.std() / (frame_mse.mean() + 1e-8) > 2:
    print(f'Delta distribution is WIDE (std/mean={frame_mse.std()/frame_mse.mean():.1f}x) — bimodal, single embedding MAY struggle.')
else:
    print(f'Delta distribution is TIGHT (std/mean={frame_mse.std()/frame_mse.mean():.1f}x) — single embedding likely fine.')
total_pixels = 64 * 64 * 3
print(f'Typical change: {changed.mean()*100:.1f}% of pixels ({changed.mean()*total_pixels:.0f}/{total_pixels})')
print(f'Recommendation: {"16x16 PATCH grid needed (per-patch deltas)" if changed.mean() > 0.05 else "SINGLE global delta token may work"}')
