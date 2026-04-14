"""Base class for AutumnBench-style grid environments.

All environments share:
- 2D grid of discrete cell types (colors)
- Actions: noop, up, down, left, right, click(x,y)
- Observations: one-hot encoded grid (C, H, W)
- Next-state prediction as the world model task
"""
from __future__ import annotations

import copy

import numpy as np


class AutumnEnv:
    """Base class for AutumnBench-inspired grid environments.

    Subclass and implement:
    - _init_grid(): create initial grid state
    - _step_dynamics(grid): apply one step of dynamics, return new grid
    - CELL_TYPES: list of cell type names
    - GRID_SIZE: default grid size

    Optional overrides:
    - _handle_click(r, c): react to click action at (r, c)
    - _is_passable(r, c): whether agent can move to (r, c)
    """

    CELL_TYPES: list[str] = ["empty"]  # override in subclass
    GRID_SIZE: int = 16
    N_ACTIONS: int = 6  # noop, up, down, left, right, click

    def __init__(self, grid_size: int | None = None, seed: int = 42):
        self.grid_size = grid_size or self.GRID_SIZE
        self.n_cell_types = len(self.CELL_TYPES)
        self.rng = np.random.default_rng(seed)
        self.grid: np.ndarray | None = None  # (H, W) integer array
        self.agent_pos: tuple[int, int] | None = None  # (row, col) or None
        self.step_count = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment and return initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.grid = self._init_grid()
        self.step_count = 0
        return self._get_obs()

    def step(self, action: int) -> np.ndarray:
        """Take one step.

        Actions:
            0 = noop
            1 = up, 2 = down, 3 = left, 4 = right  (agent movement)
            5+ = click at position (action - 5) decoded as (row, col)
        """
        self.step_count += 1

        # Handle agent movement if agent exists
        if self.agent_pos is not None and 1 <= action <= 4:
            self._move_agent(action)

        # Handle click actions
        if action >= 5:
            click_pos = action - 5
            r, c = divmod(click_pos, self.grid_size)
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                self._handle_click(r, c)

        # Apply environment dynamics
        self.grid = self._step_dynamics(self.grid)

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return one-hot encoded grid: (n_cell_types, H, W) float32."""
        obs = np.zeros(
            (self.n_cell_types, self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        for c in range(self.n_cell_types):
            obs[c] = (self.grid == c).astype(np.float32)
        return obs

    def _move_agent(self, action: int) -> None:
        """Move agent in the given direction if passable."""
        r, c = self.agent_pos
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            if self._is_passable(nr, nc):
                self.grid[r, c] = 0  # empty where agent was
                if "agent" in self.CELL_TYPES:
                    self.grid[nr, nc] = self.CELL_TYPES.index("agent")
                self.agent_pos = (nr, nc)

    def _is_passable(self, r: int, c: int) -> bool:
        """Whether agent can move to (r, c). Default: only empty cells."""
        return self.grid[r, c] == 0

    def _handle_click(self, r: int, c: int) -> None:
        """Override for click-based interactions."""
        pass

    def _init_grid(self) -> np.ndarray:
        """Override: return initial (H, W) integer grid."""
        raise NotImplementedError

    def _step_dynamics(self, grid: np.ndarray) -> np.ndarray:
        """Override: apply one step of environment dynamics, return new grid."""
        raise NotImplementedError

    def clone(self) -> AutumnEnv:
        """Create a deep copy for planning rollouts."""
        return copy.deepcopy(self)
