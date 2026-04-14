"""Self-contained Pong and Breakout environments for world model benchmarking.

No external dependencies (no ale-py, no gymnasium).
Integer-position physics, fully deterministic given a seed.
"""
from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  PONG
# ═══════════════════════════════════════════════════════════════════════════════

# Pong cell channels
_PONG_CH_BALL         = 0
_PONG_CH_LEFT_PADDLE  = 1
_PONG_CH_RIGHT_PADDLE = 2
_PONG_CH_WALLS        = 3
_PONG_N_CHANNELS      = 4

# Actions: 0=stay, 1=up, 2=down
_PONG_STAY = 0
_PONG_UP   = 1
_PONG_DOWN = 2
_PONG_N_ACTIONS = 3


class PongEnv:
    """Self-contained Pong for world model benchmarking.

    State: (4, H, W) grid encoding [ball, left_paddle, right_paddle, walls].
    Action: 3 discrete (stay, up, down) for the left paddle.
    Right paddle follows the ball (simple AI).

    Grid size: 16x32 (tall, wide — like a sideways Pong screen).
    Ball moves 1 cell per step in both axes.
    Paddles are 3 cells tall and occupy column 1 (left) / W-2 (right).
    Top/bottom rows are walls.
    """

    def __init__(self, grid_h: int = 16, grid_w: int = 32,
                 paddle_size: int = 3, seed: int = 42):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.paddle_size = paddle_size
        self.rng = np.random.default_rng(seed)

        # Paddle column positions (fixed)
        self.left_col = 1
        self.right_col = grid_w - 2

        # State variables (set in reset)
        self.ball_r = 0
        self.ball_c = 0
        self.ball_dr = 0
        self.ball_dc = 0
        self.paddle_left = 0   # row of paddle center
        self.paddle_right = 0
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to initial state, return observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        h, w = self.grid_h, self.grid_w

        # Ball at center with random direction
        self.ball_r = h // 2
        self.ball_c = w // 2
        self.ball_dr = self.rng.choice([-1, 1])
        self.ball_dc = self.rng.choice([-1, 1])

        # Paddles centered vertically
        self.paddle_left = h // 2
        self.paddle_right = h // 2

        return self._get_obs()

    def step(self, action: int) -> np.ndarray:
        """Execute one step, return next observation.

        1. Move left paddle (agent).
        2. Move right paddle (AI tracks ball).
        3. Move ball.
        4. Bounce off walls / paddles.
        5. Reset ball if it exits left or right.
        """
        h, w = self.grid_h, self.grid_w
        half_p = self.paddle_size // 2

        # --- 1. Left paddle (agent) ---
        if action == _PONG_UP:
            self.paddle_left = max(1 + half_p, self.paddle_left - 1)
        elif action == _PONG_DOWN:
            self.paddle_left = min(h - 2 - half_p, self.paddle_left + 1)

        # --- 2. Right paddle (AI) — move toward ball row ---
        if self.ball_r < self.paddle_right:
            self.paddle_right = max(1 + half_p, self.paddle_right - 1)
        elif self.ball_r > self.paddle_right:
            self.paddle_right = min(h - 2 - half_p, self.paddle_right + 1)

        # --- 3. Move ball ---
        new_r = self.ball_r + self.ball_dr
        new_c = self.ball_c + self.ball_dc

        # --- 4. Wall bounce (top/bottom) ---
        if new_r <= 0:
            new_r = 1
            self.ball_dr = 1
        elif new_r >= h - 1:
            new_r = h - 2
            self.ball_dr = -1

        # --- 5. Paddle bounce ---
        # Left paddle bounce
        if new_c == self.left_col:
            if abs(new_r - self.paddle_left) <= half_p:
                new_c = self.left_col + 1
                self.ball_dc = 1
        # Right paddle bounce
        elif new_c == self.right_col:
            if abs(new_r - self.paddle_right) <= half_p:
                new_c = self.right_col - 1
                self.ball_dc = -1

        # --- 6. Score / reset ball if out of bounds ---
        if new_c <= 0 or new_c >= w - 1:
            # Ball exits — reset to center
            new_r = h // 2
            new_c = w // 2
            self.ball_dr = self.rng.choice([-1, 1])
            self.ball_dc = self.rng.choice([-1, 1])

        self.ball_r = new_r
        self.ball_c = new_c

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return (4, H, W) one-hot observation."""
        h, w = self.grid_h, self.grid_w
        half_p = self.paddle_size // 2
        obs = np.zeros((_PONG_N_CHANNELS, h, w), dtype=np.float32)

        # Ball
        obs[_PONG_CH_BALL, self.ball_r, self.ball_c] = 1.0

        # Left paddle
        for dr in range(-half_p, half_p + 1):
            r = self.paddle_left + dr
            if 0 <= r < h:
                obs[_PONG_CH_LEFT_PADDLE, r, self.left_col] = 1.0

        # Right paddle
        for dr in range(-half_p, half_p + 1):
            r = self.paddle_right + dr
            if 0 <= r < h:
                obs[_PONG_CH_RIGHT_PADDLE, r, self.right_col] = 1.0

        # Walls (top and bottom rows)
        obs[_PONG_CH_WALLS, 0, :] = 1.0
        obs[_PONG_CH_WALLS, h - 1, :] = 1.0

        return obs


# ═══════════════════════════════════════════════════════════════════════════════
#  BREAKOUT
# ═══════════════════════════════════════════════════════════════════════════════

# Breakout cell channels
_BK_CH_BALL    = 0
_BK_CH_PADDLE  = 1
_BK_CH_BRICKS  = 2
_BK_CH_WALLS   = 3
_BK_N_CHANNELS = 4

# Actions: 0=stay, 1=left, 2=right
_BK_STAY  = 0
_BK_LEFT  = 1
_BK_RIGHT = 2
_BK_N_ACTIONS = 3


class BreakoutEnv:
    """Self-contained Breakout for world model benchmarking.

    State: (4, H, W) grid encoding [ball, paddle, bricks, walls].
    Action: 3 discrete (stay, left, right).

    Grid: 20x16.
    Paddle at bottom row-1, width 3.
    Bricks fill rows 2..5.
    Ball bounces off walls, paddle, and bricks.
    Bricks disappear on hit (local state change — good for CML).
    """

    def __init__(self, grid_h: int = 20, grid_w: int = 16,
                 paddle_width: int = 3, brick_rows: int = 4,
                 seed: int = 42):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.paddle_width = paddle_width
        self.brick_rows = brick_rows
        self.rng = np.random.default_rng(seed)

        # State (set in reset)
        self.ball_r = 0
        self.ball_c = 0
        self.ball_dr = 0
        self.ball_dc = 0
        self.paddle_center = 0  # column of paddle center
        self.bricks: np.ndarray | None = None  # (H, W) bool
        self.paddle_row = grid_h - 2
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to initial state, return observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        h, w = self.grid_h, self.grid_w

        # Ball just above paddle with upward trajectory
        self.paddle_center = w // 2
        self.ball_r = self.paddle_row - 1
        self.ball_c = w // 2
        self.ball_dr = -1
        self.ball_dc = self.rng.choice([-1, 1])

        # Bricks: rows 2..(2+brick_rows-1), columns 1..(W-2)
        self.bricks = np.zeros((h, w), dtype=bool)
        for r in range(2, 2 + self.brick_rows):
            for c in range(1, w - 1):
                self.bricks[r, c] = True

        return self._get_obs()

    def step(self, action: int) -> np.ndarray:
        """Execute one step, return next observation."""
        h, w = self.grid_h, self.grid_w
        half_p = self.paddle_width // 2

        # --- 1. Move paddle ---
        if action == _BK_LEFT:
            self.paddle_center = max(1 + half_p, self.paddle_center - 1)
        elif action == _BK_RIGHT:
            self.paddle_center = min(w - 2 - half_p, self.paddle_center + 1)

        # --- 2. Move ball ---
        new_r = self.ball_r + self.ball_dr
        new_c = self.ball_c + self.ball_dc

        # --- 3. Wall bounces ---
        # Top wall
        if new_r <= 0:
            new_r = 1
            self.ball_dr = 1
        # Side walls
        if new_c <= 0:
            new_c = 1
            self.ball_dc = 1
        elif new_c >= w - 1:
            new_c = w - 2
            self.ball_dc = -1

        # --- 4. Paddle bounce ---
        if new_r == self.paddle_row:
            if abs(new_c - self.paddle_center) <= half_p:
                new_r = self.paddle_row - 1
                self.ball_dr = -1

        # --- 5. Brick collision ---
        if 0 <= new_r < h and 0 <= new_c < w and self.bricks[new_r, new_c]:
            self.bricks[new_r, new_c] = False
            # Bounce back vertically
            self.ball_dr = -self.ball_dr
            # Don't move into brick — stay at previous position offset
            new_r = self.ball_r
            # Keep new_c (horizontal position unchanged)

        # --- 6. Ball falls below paddle — reset ---
        if new_r >= h - 1:
            new_r = self.paddle_row - 1
            new_c = w // 2
            self.ball_dr = -1
            self.ball_dc = self.rng.choice([-1, 1])

        self.ball_r = new_r
        self.ball_c = new_c

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return (4, H, W) one-hot observation."""
        h, w = self.grid_h, self.grid_w
        half_p = self.paddle_width // 2
        obs = np.zeros((_BK_N_CHANNELS, h, w), dtype=np.float32)

        # Ball
        obs[_BK_CH_BALL, self.ball_r, self.ball_c] = 1.0

        # Paddle
        for dc in range(-half_p, half_p + 1):
            c = self.paddle_center + dc
            if 0 <= c < w:
                obs[_BK_CH_PADDLE, self.paddle_row, c] = 1.0

        # Bricks
        obs[_BK_CH_BRICKS] = self.bricks.astype(np.float32)

        # Walls: top row, left col, right col
        obs[_BK_CH_WALLS, 0, :] = 1.0
        obs[_BK_CH_WALLS, :, 0] = 1.0
        obs[_BK_CH_WALLS, :, w - 1] = 1.0

        return obs
