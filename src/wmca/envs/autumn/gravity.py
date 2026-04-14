"""Gravity environment: objects fall and stack on surfaces.

CML relevance: gravity is a directional conv2d shift (downward coupling).
The CML's asymmetric kernel should capture this naturally -- each cell's
next state depends on the cell above (falling into) and cell below
(blocking/supporting).

Cell types: empty(0), wall(1), block(2), agent(3)

Dynamics (per step, processed bottom-to-top):
- Blocks fall 1 cell down if the cell below is empty
- Blocks stack on walls and other blocks
- Agent can move (up/down/left/right)
- Click to spawn a new block at that position
"""
from __future__ import annotations

import numpy as np

from wmca.envs.autumn.base import AutumnEnv


class GravityEnv(AutumnEnv):
    """Objects fall under gravity, stack on surfaces.

    Cell types: empty(0), wall(1), block(2), agent(3)

    Dynamics:
    - Blocks fall 1 cell down per step if the cell below is empty
    - Blocks stack on walls and other blocks
    - Agent can move (up/down/left/right)
    - Click to spawn a new block at that position

    CML relevance: gravity is a directional conv2d shift (downward coupling).
    The CML's asymmetric kernel should capture this naturally.
    """

    CELL_TYPES = ["empty", "wall", "block", "agent"]
    GRID_SIZE = 12

    def __init__(
        self,
        grid_size: int = 12,
        seed: int = 42,
        n_floating_blocks: int = 8,
        n_obstacle_walls: int = 4,
        use_agent: bool = False,
    ):
        super().__init__(grid_size, seed)
        self.n_floating_blocks = n_floating_blocks
        self.n_obstacle_walls = n_obstacle_walls
        self.use_agent = use_agent

    def _init_grid(self) -> np.ndarray:
        H = W = self.grid_size
        grid = np.zeros((H, W), dtype=np.int32)

        # Floor: bottom row is all walls
        grid[H - 1, :] = 1  # wall

        # Obstacle walls: small horizontal platforms scattered around
        for _ in range(self.n_obstacle_walls):
            r = int(self.rng.integers(H // 3, H - 2))  # mid to lower region
            c_start = int(self.rng.integers(0, W - 3))
            length = int(self.rng.integers(2, min(5, W - c_start + 1)))
            for c in range(c_start, min(c_start + length, W)):
                grid[r, c] = 1  # wall

        # Floating blocks: placed in upper half so they have room to fall
        placed = 0
        attempts = 0
        while placed < self.n_floating_blocks and attempts < self.n_floating_blocks * 10:
            attempts += 1
            r = int(self.rng.integers(1, H // 2 + 2))
            c = int(self.rng.integers(0, W))
            if grid[r, c] == 0:  # empty
                grid[r, c] = 2  # block
                placed += 1

        # Optional agent placement
        if self.use_agent:
            # Place agent on the floor row - 1 (standing on the floor)
            for c in range(W):
                if grid[H - 2, c] == 0:
                    grid[H - 2, c] = 3  # agent
                    self.agent_pos = (H - 2, c)
                    break

        return grid

    def _step_dynamics(self, grid: np.ndarray) -> np.ndarray:
        """Apply gravity: blocks fall down 1 cell if space below is empty.

        Process columns bottom-to-top so multiple blocks can fall
        simultaneously without collisions.
        """
        new_grid = grid.copy()
        H, W = grid.shape

        # Process bottom-to-top (skip the very bottom row, nothing can fall further)
        for r in range(H - 2, -1, -1):
            for c in range(W):
                if new_grid[r, c] == 2:  # block
                    if new_grid[r + 1, c] == 0:  # empty below
                        new_grid[r + 1, c] = 2  # block falls
                        new_grid[r, c] = 0       # old position empty

        return new_grid

    def _handle_click(self, r: int, c: int) -> None:
        """Click to spawn a new block at the given position."""
        if self.grid[r, c] == 0:  # only in empty cells
            self.grid[r, c] = 2  # block

    def _is_passable(self, r: int, c: int) -> bool:
        """Agent can move into empty cells only."""
        return self.grid[r, c] == 0
