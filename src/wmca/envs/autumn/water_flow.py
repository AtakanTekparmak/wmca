"""Water flow environment: water falls under gravity and spreads laterally.

CML relevance: water flow = gravity (downward coupling) + lateral diffusion.
This is almost exactly what CML's conv2d coupling computes -- very close to
the heat_control benchmark but discrete, with directional bias.

Cell types: empty(0), wall(1), water(2), source(3)

Dynamics (per step, processed bottom-to-top):
1. Water falls down if the cell below is empty
2. If cell below is blocked (wall/water), water spreads diagonally down
   (down-left or down-right) if available
3. If no downward movement is possible, water spreads laterally
   (left or right, randomly chosen)
4. Source cells continuously spawn water in adjacent empty cells below
"""
from __future__ import annotations

import numpy as np

from wmca.envs.autumn.base import AutumnEnv


class WaterFlowEnv(AutumnEnv):
    """Water flows under gravity and spreads laterally.

    Cell types: empty(0), wall(1), water(2), source(3)

    Dynamics (per step, processed bottom-to-top):
    1. Water falls down if cell below is empty
    2. If cell below is wall/water (blocked), water spreads left or right
    3. Source cells continuously spawn water in adjacent empty cells

    CML relevance: water flow = gravity (downward coupling) + lateral diffusion.
    This is almost exactly what CML's conv2d coupling computes.
    """

    CELL_TYPES = ["empty", "wall", "water", "source"]
    GRID_SIZE = 16

    def __init__(
        self,
        grid_size: int = 16,
        seed: int = 42,
        n_sources: int = 2,
        wall_density: float = 0.15,
        container: bool = True,
    ):
        super().__init__(grid_size, seed)
        self.n_sources = n_sources
        self.wall_density = wall_density
        self.container = container

    def _init_grid(self) -> np.ndarray:
        H = W = self.grid_size
        grid = np.zeros((H, W), dtype=np.int32)

        if self.container:
            # Build a container/basin: floor + side walls
            grid[H - 1, :] = 1  # floor
            grid[:, 0] = 1       # left wall
            grid[:, W - 1] = 1   # right wall

            # Internal walls forming ledges/maze
            n_ledges = max(1, int(H * self.wall_density))
            for _ in range(n_ledges):
                r = int(self.rng.integers(H // 3, H - 2))
                # Leave a gap so water can flow through
                c_start = int(self.rng.integers(1, W // 2))
                c_end = int(self.rng.integers(W // 2, W - 1))
                gap_pos = int(self.rng.integers(c_start, c_end))
                for c in range(c_start, c_end):
                    if c != gap_pos:
                        grid[r, c] = 1
        else:
            # No container, just scattered walls
            n_walls = int(H * W * self.wall_density)
            wall_positions = self.rng.choice(H * W, min(n_walls, H * W), replace=False)
            for pos in wall_positions:
                r, c = divmod(int(pos), W)
                grid[r, c] = 1

        # Place sources near the top
        placed = 0
        attempts = 0
        while placed < self.n_sources and attempts < self.n_sources * 20:
            attempts += 1
            r = int(self.rng.integers(0, max(1, H // 4)))
            c = int(self.rng.integers(1 if self.container else 0,
                                       W - 1 if self.container else W))
            if grid[r, c] == 0:
                grid[r, c] = 3  # source
                placed += 1

        return grid

    def _step_dynamics(self, grid: np.ndarray) -> np.ndarray:
        """Apply water dynamics: gravity + lateral spread.

        Process grid bottom-to-top for proper settling. Each water cell
        tries to move in priority order: down > down-left/down-right > left/right.
        """
        new_grid = grid.copy()
        H, W = grid.shape

        # Track which cells have been moved into this step to avoid conflicts
        moved_from = set()

        # 1. Process existing water: bottom-to-top
        for r in range(H - 2, -1, -1):
            for c in range(W):
                if new_grid[r, c] != 2:  # not water
                    continue
                if (r, c) in moved_from:  # already moved by a cell above
                    continue

                # Try to fall straight down
                if r + 1 < H and new_grid[r + 1, c] == 0:
                    new_grid[r + 1, c] = 2
                    new_grid[r, c] = 0
                    moved_from.add((r + 1, c))
                    continue

                # Try diagonal down (randomize left/right order)
                diag_dirs = [(r + 1, c - 1), (r + 1, c + 1)]
                if self.rng.random() < 0.5:
                    diag_dirs.reverse()

                moved = False
                for nr, nc in diag_dirs:
                    if 0 <= nr < H and 0 <= nc < W and new_grid[nr, nc] == 0:
                        new_grid[nr, nc] = 2
                        new_grid[r, c] = 0
                        moved_from.add((nr, nc))
                        moved = True
                        break
                if moved:
                    continue

                # Try lateral spread (only if blocked below)
                if r + 1 < H and new_grid[r + 1, c] != 0:
                    lat_dirs = [(r, c - 1), (r, c + 1)]
                    if self.rng.random() < 0.5:
                        lat_dirs.reverse()
                    for nr, nc in lat_dirs:
                        if 0 <= nr < H and 0 <= nc < W and new_grid[nr, nc] == 0:
                            new_grid[nr, nc] = 2
                            new_grid[r, c] = 0
                            moved_from.add((nr, nc))
                            break

        # 2. Sources emit water into adjacent empty cells (prefer below)
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] != 3:  # not source
                    continue
                # Try to emit below first, then sides
                emit_targets = []
                if r + 1 < H:
                    emit_targets.append((r + 1, c))
                emit_targets.extend([(r, c - 1), (r, c + 1)])
                if r - 1 >= 0:
                    emit_targets.append((r - 1, c))
                for er, ec in emit_targets:
                    if 0 <= er < H and 0 <= ec < W and new_grid[er, ec] == 0:
                        new_grid[er, ec] = 2  # spawn water
                        break  # emit one water per source per step

        return new_grid

    def _handle_click(self, r: int, c: int) -> None:
        """Click to toggle wall (build/remove barriers)."""
        if self.grid[r, c] == 0:       # empty -> wall
            self.grid[r, c] = 1
        elif self.grid[r, c] == 1:     # wall -> empty
            self.grid[r, c] = 0
