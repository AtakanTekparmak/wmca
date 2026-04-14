"""Gray-Scott reaction-diffusion control environment.

A 32x32 grid environment where an agent seeds chemical V to grow patterns
toward a target region. The Gray-Scott PDE dynamics propagate the patterns
naturally -- the agent must predict WHERE to seed so that diffusion carries
V to the target.

This environment is designed to showcase world models with CML reservoirs,
which naturally capture reaction-diffusion dynamics.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d


# ---------------------------------------------------------------------------
# Gray-Scott PDE parameters (spot / mitosis regime)
# Kept consistent with benchmarks.py
# ---------------------------------------------------------------------------
GS_D_U = 0.16
GS_D_V = 0.08
GS_F_FEED = 0.035
GS_K_KILL = 0.065
GS_DT = 1.0
GS_DX = 1.0
GS_N_SUBSTEPS = 4
GS_DT_SUB = GS_DT / GS_N_SUBSTEPS

# 5-point Laplacian stencil
_LAP_KERNEL = np.array([[0.0, 1.0, 0.0],
                         [1.0, -4.0, 1.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)

# Blob parameters for seeding
_BLOB_RADIUS = 2       # half-width of the Gaussian blob
_BLOB_V_AMP = 0.25     # amount of V deposited
_BLOB_U_DIP = 0.5      # U is reduced to this value in the blob center

# Reward parameters
_V_THRESHOLD = 0.15    # V above this counts as "activated"


# ---------------------------------------------------------------------------
# PDE stepping (numpy, no torch dependency)
# ---------------------------------------------------------------------------

def _laplacian(field: np.ndarray) -> np.ndarray:
    """Compute 2D Laplacian with periodic (wrap) boundary conditions."""
    return convolve2d(field, _LAP_KERNEL, mode="same", boundary="wrap")


def gray_scott_step(u: np.ndarray, v: np.ndarray,
                    n_sub: int = GS_N_SUBSTEPS,
                    dt_sub: float = GS_DT_SUB,
                    dx: float = GS_DX,
                    d_u: float = GS_D_U, d_v: float = GS_D_V,
                    f: float = GS_F_FEED, k: float = GS_K_KILL,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """One macro-step of Gray-Scott (n_sub Euler sub-steps).

    u, v: (H, W) float32 in [0, 1]. Returns updated (u, v).
    """
    coeff = dt_sub / (dx * dx)
    for _ in range(n_sub):
        lap_u = _laplacian(u)
        lap_v = _laplacian(v)
        uvv = u * v * v
        du = d_u * coeff * lap_u - uvv + f * (1.0 - u)
        dv = d_v * coeff * lap_v + uvv - (f + k) * v
        u = u + dt_sub * du
        v = v + dt_sub * dv
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
    return u, v


# ---------------------------------------------------------------------------
# Target mask generation
# ---------------------------------------------------------------------------

def _make_target_mask(grid_size: int, rng: np.random.Generator,
                      n_blobs: int | None = None) -> np.ndarray:
    """Generate a binary target mask from 1-3 circular blobs.

    Returns (H, W) float32 with values in {0, 1}.
    """
    h = w = grid_size
    if n_blobs is None:
        n_blobs = rng.integers(1, 4)  # 1 to 3

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(n_blobs):
        cy = rng.uniform(0.15, 0.85) * h
        cx = rng.uniform(0.15, 0.85) * w
        radius = rng.uniform(2.0, 5.0)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask[dist <= radius] = 1.0

    return mask


# ---------------------------------------------------------------------------
# Gaussian blob for seeding
# ---------------------------------------------------------------------------

def _make_blob(grid_size: int, row: int, col: int,
               radius: int = _BLOB_RADIUS) -> np.ndarray:
    """Gaussian blob centered at (row, col), shape (H, W), peak = 1."""
    yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
    sigma = max(radius / 2.0, 0.5)
    blob = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2.0 * sigma ** 2))
    return blob.astype(np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GrayScottControlEnv:
    """Gray-Scott reaction-diffusion control environment.

    State observation: (3, H, W) = [U, V, target_mask]
    Action: int in [0, grid_size^2]  (last = no-op)
        - Actions 0..grid_size^2-1 seed a blob of V at that position
        - Action grid_size^2 is a no-op (let dynamics evolve)

    Reward: IoU(V > threshold, target_mask) at each step.
    Episode length: max_steps (default 100).
    """

    def __init__(self, grid_size: int = 32, max_steps: int = 100,
                 sim_substeps: int = 10,
                 rng: np.random.Generator | None = None,
                 seed: int | None = None):
        """
        Args:
            grid_size: Side length of the square grid.
            max_steps: Maximum steps per episode.
            sim_substeps: Number of GS macro-steps per environment step.
                          More substeps = more visible dynamics per action.
            rng: Optional numpy Generator (overrides seed).
            seed: Random seed (used if rng is None).
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.sim_substeps = sim_substeps
        self.n_actions = grid_size * grid_size + 1  # positions + no-op

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

        # State arrays (set by reset)
        self.u: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.target_mask: np.ndarray | None = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        """Reset environment. Returns observation (3, H, W)."""
        h = w = self.grid_size

        # Start with uniform U=1, V=0 (pristine state)
        self.u = np.ones((h, w), dtype=np.float32)
        self.v = np.zeros((h, w), dtype=np.float32)

        # Small initial seed of V in the center to kickstart dynamics
        ch, cw = h // 2, w // 2
        noise = self.rng.uniform(-0.01, 0.01, (4, 4)).astype(np.float32)
        self.v[ch - 2:ch + 2, cw - 2:cw + 2] = 0.25 + noise
        self.u[ch - 2:ch + 2, cw - 2:cw + 2] = 0.5

        # Let dynamics run briefly so some pattern exists before the agent acts
        for _ in range(5):
            self.u, self.v = gray_scott_step(self.u, self.v)

        # Generate target mask
        self.target_mask = _make_target_mask(self.grid_size, self.rng)

        self.step_count = 0
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take one step.

        Args:
            action: int in [0, n_actions). Last action is no-op.

        Returns:
            obs (3, H, W), reward (float), done (bool), info (dict).
        """
        assert self.u is not None, "Call reset() first"
        assert 0 <= action < self.n_actions, f"Invalid action {action}"

        # Apply action: seed V blob at position
        if action < self.grid_size * self.grid_size:
            row = action // self.grid_size
            col = action % self.grid_size
            blob = _make_blob(self.grid_size, row, col, _BLOB_RADIUS)
            self.v = np.clip(self.v + _BLOB_V_AMP * blob, 0.0, 1.0)
            self.u = np.clip(self.u - (1.0 - _BLOB_U_DIP) * blob, 0.0, 1.0)

        # Run Gray-Scott dynamics for several substeps
        for _ in range(self.sim_substeps):
            self.u, self.v = gray_scott_step(self.u, self.v)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        reward = self._compute_reward()
        info = {
            "iou": reward,
            "v_coverage": float((self.v > _V_THRESHOLD).mean()),
            "target_coverage": float(self.target_mask.mean()),
        }

        return self._obs(), reward, done, info

    def _obs(self) -> np.ndarray:
        """Stack [U, V, target_mask] into (3, H, W)."""
        return np.stack([self.u, self.v, self.target_mask], axis=0)

    def _compute_reward(self) -> float:
        """IoU between activated V cells and the target mask."""
        pred = (self.v > _V_THRESHOLD).astype(np.float32)
        target = self.target_mask

        intersection = (pred * target).sum()
        union = np.clip(pred + target, 0.0, 1.0).sum()

        if union < 1e-8:
            return 0.0
        return float(intersection / union)

    # ------------------------------------------------------------------
    # Convenience for world model I/O
    # ------------------------------------------------------------------

    @staticmethod
    def state_to_model_input(obs: np.ndarray, action: int,
                             grid_size: int = 32) -> np.ndarray:
        """Convert (obs, action) to 4-channel model input.

        Input channels: [U, V, target_mask, action_map]
        Returns (4, H, W) float32.
        """
        action_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        if action < grid_size * grid_size:
            row = action // grid_size
            col = action % grid_size
            action_map[row, col] = 1.0
        return np.concatenate([obs, action_map[np.newaxis]], axis=0)

    @staticmethod
    def model_output_to_state(pred: np.ndarray, target_mask: np.ndarray
                              ) -> np.ndarray:
        """Convert 2-channel model output to 3-channel observation.

        pred: (2, H, W) = [U_pred, V_pred]
        Returns (3, H, W) = [U_pred, V_pred, target_mask].
        """
        return np.concatenate([pred, target_mask[np.newaxis]], axis=0)


# ---------------------------------------------------------------------------
# Data generation for world model training
# ---------------------------------------------------------------------------

def generate_gs_control_data(
    n_trajectories: int = 200,
    n_steps: int = 50,
    grid_size: int = 32,
    sim_substeps: int = 10,
    seed: int = 42,
) -> dict:
    """Generate (state, action, next_state) transition data.

    Uses random actions. Returns dict with numpy arrays:
        states:  (N, 3, H, W)  -- [U, V, target]
        actions: (N,)           -- int action indices
        next_states: (N, 3, H, W)
        model_inputs: (N, 4, H, W) -- [U, V, target, action_map]
        model_targets: (N, 2, H, W) -- [U_next, V_next]
    """
    rng = np.random.default_rng(seed)
    env = GrayScottControlEnv(grid_size=grid_size, sim_substeps=sim_substeps,
                              rng=rng)

    all_states = []
    all_actions = []
    all_next_states = []
    all_model_inputs = []
    all_model_targets = []

    for traj_i in range(n_trajectories):
        obs = env.reset()

        for step_i in range(n_steps):
            # Random action: 70% position actions, 30% no-op
            if rng.random() < 0.7:
                action = int(rng.integers(0, grid_size * grid_size))
            else:
                action = grid_size * grid_size  # no-op

            model_input = GrayScottControlEnv.state_to_model_input(
                obs, action, grid_size)

            next_obs, reward, done, info = env.step(action)

            all_states.append(obs)
            all_actions.append(action)
            all_next_states.append(next_obs)
            all_model_inputs.append(model_input)
            all_model_targets.append(next_obs[:2])  # just U, V

            obs = next_obs
            if done:
                break

    return {
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.int64),
        "next_states": np.array(all_next_states, dtype=np.float32),
        "model_inputs": np.array(all_model_inputs, dtype=np.float32),
        "model_targets": np.array(all_model_targets, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Benchmark integration: generate_gs_control()
# ---------------------------------------------------------------------------

def generate_gs_control(
    grid_size: int = 32,
    n_trajectories: int = 200,
    n_steps: int = 50,
    sim_substeps: int = 10,
    seed: int = 42,
    device: str = "cpu",
):
    """Generate Gray-Scott control benchmark data in BenchmarkData format.

    Compatible with the unified ablation framework in benchmarks.py.

    Returns BenchmarkData namedtuple:
        X_train/val/test: (N, 4, H, W) -- [U, V, target, action_map]
        Y_train/val/test: (N, 2, H, W) -- [U_next, V_next]
        meta: dict
    """
    import torch
    from wmca.benchmarks import BenchmarkData, _to_torch

    data = generate_gs_control_data(
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        grid_size=grid_size,
        sim_substeps=sim_substeps,
        seed=seed,
    )

    X = data["model_inputs"]   # (N, 4, H, W)
    Y = data["model_targets"]  # (N, 2, H, W)

    # Split 70/15/15
    N = len(X)
    n1 = int(0.70 * N)
    n2 = int(0.15 * N)

    X_tr, Y_tr = X[:n1], Y[:n1]
    X_v, Y_v = X[n1:n1 + n2], Y[n1:n1 + n2]
    X_te, Y_te = X[n1 + n2:], Y[n1 + n2:]

    tensors = _to_torch([X_tr, Y_tr, X_v, Y_v, X_te, Y_te], device)

    meta = {
        "name": "gs_control",
        "loss_type": "mse",
        "metric": "mse",
        "in_channels": 4,       # U, V, target, action_map
        "out_channels": 2,      # U_next, V_next
        "grid_size": grid_size,
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "sim_substeps": sim_substeps,
        "d_u": GS_D_U,
        "d_v": GS_D_V,
        "f_feed": GS_F_FEED,
        "k_kill": GS_K_KILL,
        "dt": GS_DT,
        "dx": GS_DX,
        "n_pde_substeps": GS_N_SUBSTEPS,
        "action_conditioned": True,
        "env_class": GrayScottControlEnv,
    }
    return BenchmarkData(*tensors, meta)


# ---------------------------------------------------------------------------
# CEM planning for evaluation
# ---------------------------------------------------------------------------

def run_gs_cem_evaluation(
    model,
    gs_data,
    n_episodes: int = 50,
    horizon: int = 5,
    population: int = 128,
    elite_k: int = 25,
    cem_iters: int = 3,
    max_steps: int = 50,
    device: str = "cpu",
    seed: int = 12345,
) -> dict:
    """CEM planning evaluation for the Gray-Scott control environment.

    Uses the world model to plan action sequences that maximize IoU with
    the target mask.

    Args:
        model: World model mapping (4, H, W) -> (2, H, W).
        gs_data: BenchmarkData with meta containing env_class.
        n_episodes: Number of evaluation episodes.
        horizon: Planning horizon (number of steps to look ahead).
        population: CEM population size.
        elite_k: Number of elite samples per CEM iteration.
        cem_iters: Number of CEM refinement iterations.
        max_steps: Maximum steps per episode.
        device: Torch device.
        seed: Random seed for eval environments.

    Returns:
        dict with avg_iou, avg_final_iou, avg_reward.
    """
    import torch

    dev = torch.device(device) if isinstance(device, str) else device

    meta = getattr(gs_data, "meta", {}) or {}
    env_class = meta.get("env_class", GrayScottControlEnv)
    grid_size = meta.get("grid_size", 32)
    n_actions = grid_size * grid_size + 1

    if hasattr(model, "to"):
        model = model.to(dev)
    if hasattr(model, "eval"):
        model.eval()

    eval_rng = np.random.default_rng(seed)

    total_iou = 0.0
    total_final_iou = 0.0
    total_reward = 0.0

    for ep in range(n_episodes):
        env = env_class(grid_size=grid_size, rng=eval_rng)
        obs = env.reset()  # (3, H, W)

        ep_reward = 0.0
        last_iou = 0.0

        for step in range(max_steps):
            # --- CEM planning ---
            # Action distribution: logits over n_actions for each timestep
            action_logits = torch.zeros(horizon, n_actions, device=dev)

            for _cem_iter in range(cem_iters):
                # Sample action sequences
                probs = torch.softmax(action_logits, dim=-1)
                action_seqs = torch.zeros(population, horizon,
                                          dtype=torch.long, device=dev)
                for t in range(horizon):
                    action_seqs[:, t] = torch.multinomial(
                        probs[t].unsqueeze(0).expand(population, -1),
                        1,
                    ).squeeze(-1)

                # Batched rollout through world model
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
                cur_uv = obs_t[:, :2].expand(population, -1, -1, -1).clone()
                target_ch = obs_t[:, 2:3].expand(population, -1, -1, -1)

                for t in range(horizon):
                    acts = action_seqs[:, t]  # (population,)

                    # Build action maps
                    action_maps = torch.zeros(
                        population, 1, grid_size, grid_size, device=dev)
                    for b in range(population):
                        a = int(acts[b].item())
                        if a < grid_size * grid_size:
                            r = a // grid_size
                            c = a % grid_size
                            action_maps[b, 0, r, c] = 1.0

                    # Model input: [U, V, target, action_map]
                    x = torch.cat([cur_uv, target_ch, action_maps], dim=1)
                    with torch.no_grad():
                        pred_uv = model(x)  # (population, 2, H, W)
                    cur_uv = pred_uv.clamp(0.0, 1.0)

                # Score: IoU of predicted V with target
                v_pred = cur_uv[:, 1]  # (population, H, W)
                activated = (v_pred > _V_THRESHOLD).float()
                target_2d = target_ch[:, 0]  # (population, H, W)

                intersection = (activated * target_2d).sum(dim=(1, 2))
                union = (activated + target_2d).clamp(0.0, 1.0).sum(dim=(1, 2))
                ious = torch.where(union > 1e-8,
                                   intersection / union,
                                   torch.zeros_like(union))

                # Select elites and update distribution
                elite_vals, elite_idx = ious.topk(elite_k)
                elite_actions = action_seqs[elite_idx]  # (elite_k, horizon)

                # Update logits from elite action frequencies
                new_logits = torch.zeros_like(action_logits)
                for t in range(horizon):
                    counts = torch.zeros(n_actions, device=dev)
                    for a in elite_actions[:, t]:
                        counts[a.item()] += 1.0
                    # Smooth to avoid degenerate distributions
                    new_logits[t] = torch.log(counts + 0.1)
                action_logits = new_logits

            # Execute best first action
            probs_final = torch.softmax(action_logits[0], dim=-1)
            best_action = int(probs_final.argmax().item())

            obs, reward, done, info = env.step(best_action)
            ep_reward += reward
            last_iou = info["iou"]

            if done:
                break

        total_reward += ep_reward
        total_iou += ep_reward / max_steps  # average IoU across episode
        total_final_iou += last_iou

    results = {
        "avg_reward": total_reward / n_episodes,
        "avg_iou": total_iou / n_episodes,
        "avg_final_iou": total_final_iou / n_episodes,
        "n_episodes": n_episodes,
        "horizon": horizon,
        "population": population,
        "planning_method": "CEM",
    }
    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Gray-Scott Control Env Smoke Test ===\n")

    # 1. Environment basics
    env = GrayScottControlEnv(grid_size=32, max_steps=20, seed=42)
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    assert obs.shape == (3, 32, 32), f"Expected (3, 32, 32), got {obs.shape}"
    print(f"U range: [{obs[0].min():.4f}, {obs[0].max():.4f}]")
    print(f"V range: [{obs[1].min():.4f}, {obs[1].max():.4f}]")
    print(f"Target coverage: {obs[2].mean():.4f}")
    print(f"N actions: {env.n_actions}")

    # 2. Step through a few actions
    print("\n--- Running 10 steps with random actions ---")
    total_r = 0.0
    for i in range(10):
        action = int(np.random.randint(0, env.n_actions))
        obs, reward, done, info = env.step(action)
        total_r += reward
        if i % 3 == 0:
            print(f"  step {i}: action={action}, reward={reward:.4f}, "
                  f"iou={info['iou']:.4f}, v_cov={info['v_coverage']:.4f}")

    print(f"  Total reward: {total_r:.4f}")
    assert obs.shape == (3, 32, 32)

    # 3. Model I/O helpers
    mi = GrayScottControlEnv.state_to_model_input(obs, 100, 32)
    assert mi.shape == (4, 32, 32), f"Model input shape: {mi.shape}"
    mo = GrayScottControlEnv.model_output_to_state(obs[:2], obs[2])
    assert mo.shape == (3, 32, 32), f"Model output shape: {mo.shape}"
    print(f"\nModel input shape: {mi.shape}")
    print(f"Model output shape: {mo.shape}")

    # 4. Data generation (small)
    print("\n--- Generating training data (5 trajectories, 10 steps) ---")
    data = generate_gs_control_data(
        n_trajectories=5, n_steps=10, grid_size=32, seed=99)
    print(f"  model_inputs:  {data['model_inputs'].shape}")
    print(f"  model_targets: {data['model_targets'].shape}")
    print(f"  actions:       {data['actions'].shape}")
    assert data["model_inputs"].shape[1] == 4
    assert data["model_targets"].shape[1] == 2

    # 5. Benchmark data generation
    print("\n--- Generating benchmark data (5 trajectories) ---")
    bd = generate_gs_control(n_trajectories=5, n_steps=10, grid_size=32)
    print(f"  X_train: {bd.X_train.shape}, Y_train: {bd.Y_train.shape}")
    print(f"  meta: in_ch={bd.meta['in_channels']}, out_ch={bd.meta['out_channels']}")
    assert bd.X_train.shape[1] == 4
    assert bd.Y_train.shape[1] == 2

    # 6. No-op action
    env2 = GrayScottControlEnv(grid_size=32, max_steps=5, seed=0)
    obs2 = env2.reset()
    noop = env2.n_actions - 1
    obs2_next, r2, _, _ = env2.step(noop)
    print(f"\nNo-op step: V changed by {np.abs(obs2_next[1] - obs2[1]).mean():.6f} (mean)")

    print("\n=== All smoke tests passed ===")
