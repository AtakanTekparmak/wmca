"""2D heat control environment for world-model-based planning.

The environment simulates a 2D heat equation on a grid.  An agent toggles
point heat sources to steer the temperature field toward a smooth target
distribution.  The dynamics ARE the discrete heat equation -- exactly the
inductive bias baked into our CML-based world model.

Dependencies: numpy only (no gym/gymnasium required).
"""
from __future__ import annotations

import numpy as np


class HeatControlEnv:
    """2D heat control environment.

    State : temperature field (grid_size x grid_size) in [0, 1]
    Action: integer in [0, grid_size^2) -- toggles heat source at that cell,
            or grid_size^2 for no-op
    Reward: -MSE(temperature, target)
    Dynamics: discrete heat equation with controllable point sources
    """

    def __init__(
        self,
        grid_size: int = 16,
        alpha: float = 0.1,
        source_strength: float = 0.1,
        n_sources_max: int = 8,
        episode_length: int = 50,
        seed: int = 42,
        rng: np.random.Generator | None = None,
    ):
        self.grid_size = grid_size
        self.alpha = alpha  # diffusion coefficient
        self.source_strength = source_strength
        self.n_sources_max = n_sources_max
        self.episode_length = episode_length
        self.rng = rng or np.random.default_rng(seed)

        # toggle at each cell + no-op
        self.n_actions = grid_size * grid_size + 1

        self.temperature: np.ndarray | None = None
        self.sources: np.ndarray | None = None  # binary mask of active heat sources
        self.target: np.ndarray | None = None
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset the environment and return initial state (3, H, W)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.temperature = np.full(
            (self.grid_size, self.grid_size), 0.3, dtype=np.float32
        )
        self.sources = np.zeros(
            (self.grid_size, self.grid_size), dtype=bool
        )
        self.step_count = 0

        # Generate a smooth target temperature field
        self.target = self._generate_target()

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one step: toggle source, apply heat, diffuse.

        Returns (state, reward, done, info).
        """
        self.step_count += 1

        # Toggle heat source (action = cell index, or n_actions-1 for no-op)
        if action < self.grid_size * self.grid_size:
            r, c = divmod(action, self.grid_size)
            self.sources[r, c] = not self.sources[r, c]

        # Apply heat sources (sources emit heat)
        self.temperature = np.where(
            self.sources,
            np.minimum(self.temperature + self.source_strength, 1.0),
            self.temperature,
        )

        # Diffuse
        self.temperature = self._diffuse(self.temperature)
        self.temperature = np.clip(self.temperature, 0.0, 1.0)

        # Reward: negative MSE to target
        reward = -float(np.mean((self.temperature - self.target) ** 2))
        done = self.step_count >= self.episode_length

        info = {"mse": -reward}
        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_target(self) -> np.ndarray:
        """Generate a smooth target temperature field via Gaussian blobs."""
        target = np.full(
            (self.grid_size, self.grid_size), 0.3, dtype=np.float32
        )
        n_blobs = int(self.rng.integers(2, 5))
        for _ in range(n_blobs):
            cx = int(self.rng.integers(2, self.grid_size - 2))
            cy = int(self.rng.integers(2, self.grid_size - 2))
            sigma = float(self.rng.uniform(1.5, 3.0))
            amp = float(self.rng.uniform(0.3, 0.7))
            y, x = np.mgrid[: self.grid_size, : self.grid_size]
            blob = amp * np.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
            )
            target = target + blob.astype(np.float32)
        return np.clip(target, 0.0, 1.0)

    def _diffuse(self, field: np.ndarray) -> np.ndarray:
        """One step of discrete heat equation with periodic (wrap) BC."""
        padded = np.pad(field, 1, mode="wrap")
        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * field
        )
        return field + self.alpha * laplacian

    def _get_state(self) -> np.ndarray:
        """State = [temperature, sources, target] as 3-channel grid.

        Returns shape (3, grid_size, grid_size) float32.
        """
        return np.stack(
            [
                self.temperature,
                self.sources.astype(np.float32),
                self.target,
            ],
            axis=0,
        )

    # ------------------------------------------------------------------
    # Clone for planning
    # ------------------------------------------------------------------

    def clone(self) -> "HeatControlEnv":
        """Create a deep copy (for CEM rollouts in the real env)."""
        env = HeatControlEnv.__new__(HeatControlEnv)
        env.grid_size = self.grid_size
        env.alpha = self.alpha
        env.source_strength = self.source_strength
        env.n_sources_max = self.n_sources_max
        env.episode_length = self.episode_length
        env.n_actions = self.n_actions
        env.rng = np.random.default_rng()
        env.temperature = self.temperature.copy()
        env.sources = self.sources.copy()
        env.target = self.target.copy()
        env.step_count = self.step_count
        return env


# ======================================================================
# Data generation for world model training
# ======================================================================

def generate_heat_control_transitions(
    env: HeatControlEnv,
    n_episodes: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (state+action, next_temperature) transitions.

    Returns:
        states:      (N, 4, H, W) float32  -- [temp, sources, target, action_map]
        next_states: (N, 1, H, W) float32  -- [next_temperature]
    """
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)

        for _t in range(env.episode_length):
            action = int(rng.integers(env.n_actions))

            # Action as a spatial mask: zeros except 1 at the action cell
            action_map = np.zeros(
                (1, env.grid_size, env.grid_size), dtype=np.float32
            )
            if action < env.grid_size * env.grid_size:
                r, c = divmod(action, env.grid_size)
                action_map[0, r, c] = 1.0

            # (4, H, W): [temp, sources, target, action_map]
            state_with_action = np.concatenate([state, action_map], axis=0)

            next_state_raw, _reward, done, _info = env.step(action)

            states.append(state_with_action)
            # Predict only the next temperature channel
            next_states.append(next_state_raw[:1])  # (1, H, W)

            state = next_state_raw
            if done:
                break

    return np.array(states, dtype=np.float32), np.array(next_states, dtype=np.float32)
