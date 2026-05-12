"""Generate Atari Pong + Breakout training data. Quick run: 50 trajectories each."""
import sys, os, numpy as np
sys.path.insert(0, 'src')
os.makedirs('experiments/atari_data', exist_ok=True)
from wmca.envs.atari_pong import PongEnv, BreakoutEnv

for name, Env in [('pong', PongEnv), ('breakout', BreakoutEnv)]:
    env = Env(seed=42)
    rng = np.random.default_rng(42)
    frames, next_frames, actions = [], [], []
    for traj in range(50):
        state = env.reset()
        for _ in range(50):
            a = rng.integers(0, 3)
            ns = env.step(a)
            frames.append(state.astype(np.float32))
            next_frames.append(ns.astype(np.float32))
            actions.append(a)
            state = ns
        if traj % 10 == 0:
            print(f'{name}: {traj}/50')
    np.save(f'experiments/atari_data/{name}_frames.npy', np.stack(frames))
    np.save(f'experiments/atari_data/{name}_next_frames.npy', np.stack(next_frames))
    np.save(f'experiments/atari_data/{name}_actions.npy', np.array(actions, dtype=np.int64))
    print(f'{name} done: {len(frames)} transitions, shape={np.stack(frames).shape}')
