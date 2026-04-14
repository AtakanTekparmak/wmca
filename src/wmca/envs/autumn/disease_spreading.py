"""SIR disease spreading on a 2D lattice.

This is the ideal CML benchmark: infection spreading IS local spatial
coupling between neighboring cells.

Cell types: empty(0), susceptible(1), infected(2), recovered(3), wall(4)

Dynamics (per step):
- Each infected cell infects each susceptible von Neumann neighbor
  with probability p_infect
- Each infected cell recovers with probability p_recover
- Recovered cells lose immunity with probability p_lose_immunity (SIRS)
- Walls block infection spread
"""
from __future__ import annotations

import numpy as np

from wmca.envs.autumn.base import AutumnEnv


class DiseaseSpreadingEnv(AutumnEnv):
    """SIR disease spreading on a 2D grid.

    Pure dynamics (no agent) by default.
    Click action places quarantine walls.
    """

    CELL_TYPES = ["empty", "susceptible", "infected", "recovered", "wall"]
    GRID_SIZE = 16

    def __init__(
        self,
        grid_size: int = 16,
        seed: int = 42,
        p_infect: float = 0.3,
        p_recover: float = 0.1,
        p_lose_immunity: float = 0.0,
        initial_infected: int = 3,
        wall_density: float = 0.1,
    ):
        super().__init__(grid_size, seed)
        self.p_infect = p_infect
        self.p_recover = p_recover
        self.p_lose_immunity = p_lose_immunity
        self.initial_infected = initial_infected
        self.wall_density = wall_density

    def _init_grid(self) -> np.ndarray:
        grid = np.ones((self.grid_size, self.grid_size), dtype=np.int32)  # susceptible

        # Place walls
        n_walls = int(self.grid_size ** 2 * self.wall_density)
        if n_walls > 0:
            wall_positions = self.rng.choice(
                self.grid_size ** 2, n_walls, replace=False,
            )
            for pos in wall_positions:
                r, c = divmod(pos, self.grid_size)
                grid[r, c] = 4  # wall

        # Place initial infected cells among susceptible
        susceptible = np.argwhere(grid == 1)
        n_infect = min(self.initial_infected, len(susceptible))
        if n_infect > 0:
            infected_idx = self.rng.choice(len(susceptible), n_infect, replace=False)
            for idx in infected_idx:
                r, c = susceptible[idx]
                grid[r, c] = 2  # infected

        return grid

    def _step_dynamics(self, grid: np.ndarray) -> np.ndarray:
        new_grid = grid.copy()
        H, W = grid.shape

        for r in range(H):
            for c in range(W):
                cell = grid[r, c]

                if cell == 1:  # susceptible -> check neighbors for infection
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 2:
                            if self.rng.random() < self.p_infect:
                                new_grid[r, c] = 2  # infected
                                break

                elif cell == 2:  # infected -> may recover
                    if self.rng.random() < self.p_recover:
                        new_grid[r, c] = 3  # recovered

                elif cell == 3:  # recovered -> may lose immunity
                    if self.p_lose_immunity > 0 and self.rng.random() < self.p_lose_immunity:
                        new_grid[r, c] = 1  # susceptible again

        return new_grid

    def _handle_click(self, r: int, c: int) -> None:
        """Click to place quarantine wall."""
        if self.grid[r, c] != 4:  # don't remove existing walls
            self.grid[r, c] = 4
