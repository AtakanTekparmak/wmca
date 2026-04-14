"""Grid World + CEM Planning: World Model for Navigation.

Trains a CML-based world model on a simple 2D grid environment,
then uses Cross-Entropy Method (CEM) planning over imagined rollouts
to navigate an agent to a goal.

Compares:
  1. CEM + ResidualCorrection (CML) world model
  2. CEM + PureNCA world model
  3. CEM + true environment (oracle upper bound)
  4. Random actions (lower bound)

Usage:
    PYTHONPATH=src python3 -u experiments/grid_world_planning.py --no-wandb
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmca.modules.hybrid import CML2D, PureNCA, ResidualCorrectionWM
from wmca.utils import pick_device

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# ===== Constants ===========================================================
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
SEED = 42
GRID_SIZE = 16
N_CELL_TYPES = 4  # empty=0, wall=1, agent=2, goal=3
N_ACTIONS = 4     # up=0, down=1, left=2, right=3

# Training
N_TRANSITIONS = 5000
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50

# CEM
HORIZON = 10
POP_SIZE = 200
ELITE_K = 40
CEM_ITERS = 3

# Evaluation
N_EVAL_EPISODES = 100
MAX_STEPS = 50

WALL_DENSITY = 0.20


# ===== Grid World ==========================================================

class SimpleGridWorld:
    """16x16 grid world with walls, agent, and goal."""

    def __init__(self, grid_size: int = GRID_SIZE, wall_density: float = WALL_DENSITY,
                 rng: np.random.Generator | None = None):
        self.grid_size = grid_size
        self.wall_density = wall_density
        self.rng = rng or np.random.default_rng(SEED)
        self.reset()

    def reset(self) -> np.ndarray:
        """Generate a random maze and return one-hot state."""
        gs = self.grid_size
        self.grid = np.zeros((gs, gs), dtype=np.int64)

        # Place walls
        wall_mask = self.rng.random((gs, gs)) < self.wall_density
        # Keep borders clear for agent/goal placement flexibility
        wall_mask[0, :] = False
        wall_mask[-1, :] = False
        wall_mask[:, 0] = False
        wall_mask[:, -1] = False
        self.grid[wall_mask] = 1  # wall

        # Place agent on random empty cell
        empty = np.argwhere(self.grid == 0)
        idx = self.rng.integers(len(empty))
        self.agent_pos = tuple(empty[idx])
        self.grid[self.agent_pos] = 2

        # Place goal on different random empty cell
        empty = np.argwhere(self.grid == 0)
        idx = self.rng.integers(len(empty))
        self.goal_pos = tuple(empty[idx])
        self.grid[self.goal_pos] = 3

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return 4-channel one-hot encoding: (4, H, W)."""
        state = np.zeros((N_CELL_TYPES, self.grid_size, self.grid_size), dtype=np.float32)
        for c in range(N_CELL_TYPES):
            state[c] = (self.grid == c).astype(np.float32)
        return state

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Execute action, return (next_state, reward, done)."""
        ar, ac = self.agent_pos
        # Action deltas: up=0, down=1, left=2, right=3
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = ar + dr, ac + dc

        # Check bounds and walls
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self.grid[nr, nc] != 1:
            # Move agent
            self.grid[ar, ac] = 0  # old position becomes empty
            self.agent_pos = (nr, nc)

            # Check goal
            if self.agent_pos == self.goal_pos:
                self.grid[nr, nc] = 2
                return self._get_state(), 1.0, True

            self.grid[nr, nc] = 2
        # else: agent stays in place (blocked)

        # Re-mark goal (it might have been overwritten if agent is on goal,
        # but we handle done above)
        if self.grid[self.goal_pos] == 0:
            self.grid[self.goal_pos] = 3

        return self._get_state(), 0.0, False

    def clone(self) -> "SimpleGridWorld":
        """Create a copy for planning."""
        env = SimpleGridWorld.__new__(SimpleGridWorld)
        env.grid_size = self.grid_size
        env.wall_density = self.wall_density
        env.rng = np.random.default_rng()
        env.grid = self.grid.copy()
        env.agent_pos = self.agent_pos
        env.goal_pos = self.goal_pos
        return env


def state_to_tensor(state: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (4, H, W) numpy state to (1, 4, H, W) tensor."""
    return torch.from_numpy(state).unsqueeze(0).to(device)


def make_action_field(state: np.ndarray, action: int, device: torch.device) -> torch.Tensor:
    """Create 4-channel action field: one-hot at agent position.

    Returns (1, 4, H, W) tensor.
    """
    gs = state.shape[1]
    action_field = np.zeros((N_ACTIONS, gs, gs), dtype=np.float32)
    # Find agent position from state
    agent_ch = state[2]  # channel 2 is agent
    agent_pos = np.argwhere(agent_ch > 0.5)
    if len(agent_pos) > 0:
        ar, ac = agent_pos[0]
        action_field[action, ar, ac] = 1.0
    return torch.from_numpy(action_field).unsqueeze(0).to(device)


def make_action_field_from_pos(agent_pos: tuple[int, int], action: int,
                                gs: int, device: torch.device) -> torch.Tensor:
    """Create action field given agent position directly."""
    action_field = torch.zeros(1, N_ACTIONS, gs, gs, device=device)
    action_field[0, action, agent_pos[0], agent_pos[1]] = 1.0
    return action_field


# ===== Data Generation =====================================================

def generate_transitions(n: int, device: torch.device,
                         rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
    """Generate n random (state, action_field, next_state) transitions."""
    states = []
    action_fields = []
    next_states = []

    env = SimpleGridWorld(rng=rng)
    transitions_collected = 0

    while transitions_collected < n:
        state = env.reset()
        for _ in range(20):  # up to 20 steps per episode
            action = rng.integers(N_ACTIONS)
            af = make_action_field(state, action, torch.device("cpu"))
            next_state, _, done = env.step(action)

            states.append(torch.from_numpy(state))
            action_fields.append(af.squeeze(0))
            next_states.append(torch.from_numpy(next_state))
            transitions_collected += 1

            if transitions_collected >= n or done:
                break
            state = next_state

    states_t = torch.stack(states[:n]).to(device)
    afs_t = torch.stack(action_fields[:n]).to(device)
    next_states_t = torch.stack(next_states[:n]).to(device)
    return states_t, afs_t, next_states_t


# ===== World Models =========================================================

class ActionConditionedResCor(nn.Module):
    """ResidualCorrection world model with action conditioning.

    CML operates on 4 state channels only (physics of state space).
    NCA correction receives state + CML output + action field (12ch -> 4ch).
    This separates physics (CML) from action perturbation (NCA).
    """
    def __init__(self, hidden_ch: int = 32, cml_steps: int = 5):
        super().__init__()
        state_ch = N_CELL_TYPES   # 4
        action_ch = N_ACTIONS     # 4

        # CML operates ONLY on state channels
        self.cml_2d = CML2D(in_channels=state_ch, steps=cml_steps)

        # NCA correction: state(4) + cml_out(4) + action(4) = 12 -> 4
        self.nca = nn.Sequential(
            nn.Conv2d(state_ch + state_ch + action_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, state_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 8, H, W) -> (B, 4, H, W) logits."""
        state = x[:, :N_CELL_TYPES]   # (B, 4, H, W)
        action = x[:, N_CELL_TYPES:]  # (B, 4, H, W)
        cml_out = self.cml_2d(state)
        correction = self.nca(torch.cat([state, cml_out, action], dim=1))
        return cml_out + correction  # residual correction on CML base

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters())
        frozen = sum(b.numel() for b in self.cml_2d.buffers())
        return {"trained": trained, "frozen": frozen}


class ActionConditionedNCA(nn.Module):
    """Pure NCA world model with action conditioning. Baseline."""
    def __init__(self, hidden_ch: int = 32, steps: int = 1):
        super().__init__()
        in_ch = N_CELL_TYPES + N_ACTIONS  # 8
        out_ch = N_CELL_TYPES  # 4
        self.steps = steps

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters()),
            "frozen": 0,
        }


class Conv2DBaseline(nn.Module):
    """Conv2D baseline world model."""
    def __init__(self, hidden_ch: int = 32):
        super().__init__()
        in_ch = N_CELL_TYPES + N_ACTIONS  # 8
        out_ch = N_CELL_TYPES  # 4

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> dict[str, int]:
        return {
            "trained": sum(p.numel() for p in self.parameters()),
            "frozen": 0,
        }


# ===== Training =============================================================

def train_world_model(model: nn.Module, states: torch.Tensor,
                      action_fields: torch.Tensor, next_states: torch.Tensor,
                      epochs: int = EPOCHS, lr: float = LR,
                      batch_size: int = BATCH_SIZE, device: torch.device = None) -> list[float]:
    """Train world model with cross-entropy loss."""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = states.shape[0]
    # Convert next_states from one-hot (B, 4, H, W) to class indices (B, H, W)
    next_labels = next_states.argmax(dim=1)  # (B, H, W) with values in {0,1,2,3}

    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            s = states[idx]
            af = action_fields[idx]
            ns_labels = next_labels[idx]

            x = torch.cat([s, af], dim=1)  # (B, 8, H, W)
            logits = model(x)  # (B, 4, H, W)

            loss = F.cross_entropy(logits, ns_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}")

    model.eval()
    return losses


# ===== CEM Planning =========================================================

def predict_next_state(model: nn.Module, state_t: torch.Tensor,
                       action: int, device: torch.device) -> torch.Tensor:
    """Use world model to predict next state.

    state_t: (1, 4, H, W) one-hot float
    Returns: (1, 4, H, W) one-hot float (argmax of softmax)
    """
    # Find agent position from state
    agent_ch = state_t[0, 2]  # (H, W)
    agent_pos_flat = agent_ch.argmax()
    ar = agent_pos_flat // GRID_SIZE
    ac = agent_pos_flat % GRID_SIZE

    af = torch.zeros(1, N_ACTIONS, GRID_SIZE, GRID_SIZE, device=device)
    af[0, action, ar, ac] = 1.0

    x = torch.cat([state_t, af], dim=1)
    with torch.no_grad():
        logits = model(x)
    # Convert logits to one-hot prediction
    pred_classes = logits.argmax(dim=1)  # (1, H, W)
    pred_onehot = F.one_hot(pred_classes, N_CELL_TYPES).permute(0, 3, 1, 2).float()
    return pred_onehot


def get_agent_pos_from_state(state_t: torch.Tensor) -> tuple[int, int]:
    """Extract agent (row, col) from one-hot state tensor (1, 4, H, W)."""
    agent_ch = state_t[0, 2]
    pos = (agent_ch > 0.5).nonzero(as_tuple=False)
    if len(pos) == 0:
        return (-1, -1)
    return (pos[0, 0].item(), pos[0, 1].item())


def get_goal_pos_from_state(state_t: torch.Tensor) -> tuple[int, int]:
    """Extract goal (row, col) from one-hot state tensor (1, 4, H, W)."""
    goal_ch = state_t[0, 3]
    pos = (goal_ch > 0.5).nonzero(as_tuple=False)
    if len(pos) == 0:
        return (-1, -1)
    return (pos[0, 0].item(), pos[0, 1].item())


def cem_plan(model: nn.Module | None, state_t: torch.Tensor,
             goal_pos: tuple[int, int], device: torch.device,
             env: SimpleGridWorld | None = None,
             use_true_env: bool = False,
             horizon: int = HORIZON, pop_size: int = POP_SIZE,
             elite_k: int = ELITE_K, cem_iters: int = CEM_ITERS) -> int:
    """CEM planning: return best first action.

    If use_true_env=True, use env.clone() for rollouts instead of model.
    """
    gs = GRID_SIZE
    goal_r, goal_c = goal_pos

    # Initialize action distribution: uniform categorical
    # We represent as logits for each timestep
    action_probs = torch.ones(horizon, N_ACTIONS, device=device) / N_ACTIONS

    for cem_iter in range(cem_iters):
        # Sample population of action sequences
        action_seqs = torch.zeros(pop_size, horizon, dtype=torch.long, device=device)
        for t in range(horizon):
            action_seqs[:, t] = torch.multinomial(action_probs[t].unsqueeze(0).expand(pop_size, -1), 1).squeeze(-1)

        # Evaluate each sequence
        rewards = torch.zeros(pop_size, device=device)

        if use_true_env:
            # Use true environment for rollouts
            for i in range(pop_size):
                env_copy = env.clone()
                for t in range(horizon):
                    _, r, done = env_copy.step(action_seqs[i, t].item())
                    if done:
                        rewards[i] = 1.0
                        break
                if rewards[i] < 0.5:
                    # Distance-based reward
                    ar, ac = env_copy.agent_pos
                    dist = abs(ar - goal_r) + abs(ac - goal_c)
                    rewards[i] = -float(dist)
        else:
            # Batched rollout with world model for efficiency
            # Expand state for all pop members: (pop_size, 4, H, W)
            cur_states = state_t.expand(pop_size, -1, -1, -1).clone()

            for t in range(horizon):
                actions_t = action_seqs[:, t]  # (pop_size,)

                # Build action fields for entire batch
                agent_ch = cur_states[:, 2]  # (pop_size, H, W)
                agent_flat = agent_ch.reshape(pop_size, -1).argmax(dim=1)  # (pop_size,)
                agent_r = agent_flat // gs
                agent_c = agent_flat % gs

                af = torch.zeros(pop_size, N_ACTIONS, gs, gs, device=device)
                batch_idx = torch.arange(pop_size, device=device)
                af[batch_idx, actions_t, agent_r, agent_c] = 1.0

                x = torch.cat([cur_states, af], dim=1)  # (pop_size, 8, H, W)
                with torch.no_grad():
                    logits = model(x)
                pred_classes = logits.argmax(dim=1)  # (pop_size, H, W)
                cur_states = F.one_hot(pred_classes, N_CELL_TYPES).permute(0, 3, 1, 2).float()

            # Compute rewards based on final predicted agent position
            agent_ch = cur_states[:, 2]
            agent_flat = agent_ch.reshape(pop_size, -1).argmax(dim=1)
            pred_ar = agent_flat // gs
            pred_ac = agent_flat % gs

            # Check for exact goal reach
            reached = (pred_ar == goal_r) & (pred_ac == goal_c)
            dist = (pred_ar - goal_r).abs() + (pred_ac - goal_c).abs()
            rewards = torch.where(reached, torch.ones_like(dist, dtype=torch.float32),
                                  -dist.float())

        # Select elites
        _, elite_idx = rewards.topk(elite_k)
        elite_actions = action_seqs[elite_idx]  # (elite_k, horizon)

        # Update action distribution from elites
        for t in range(horizon):
            counts = torch.zeros(N_ACTIONS, device=device)
            for a in range(N_ACTIONS):
                counts[a] = (elite_actions[:, t] == a).float().sum()
            action_probs[t] = (counts + 0.1) / (elite_k + 0.1 * N_ACTIONS)  # smoothed

    # Return first action of best sequence
    best_idx = elite_idx[0]
    return action_seqs[best_idx, 0].item()


# ===== Evaluation ============================================================

@dataclass
class EvalResult:
    name: str
    success_rate: float
    avg_steps: float
    avg_reward: float


def evaluate_strategy(strategy_name: str, model: nn.Module | None,
                      device: torch.device, n_episodes: int = N_EVAL_EPISODES,
                      max_steps: int = MAX_STEPS, use_true_env: bool = False,
                      random_policy: bool = False) -> EvalResult:
    """Evaluate a planning strategy over random episodes."""
    rng = np.random.default_rng(SEED + 999)
    successes = 0
    total_steps_success = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        env = SimpleGridWorld(rng=rng)
        state = env.reset()
        goal_pos = env.goal_pos
        episode_reward = 0.0

        for step in range(max_steps):
            state_t = state_to_tensor(state, device)

            if random_policy:
                action = rng.integers(N_ACTIONS)
            elif use_true_env:
                action = cem_plan(None, state_t, goal_pos, device,
                                  env=env, use_true_env=True)
            else:
                action = cem_plan(model, state_t, goal_pos, device)

            state, reward, done = env.step(action)
            episode_reward += reward

            if done:
                successes += 1
                total_steps_success += step + 1
                break

        total_reward += episode_reward

        if (ep + 1) % 20 == 0:
            print(f"  {strategy_name}: episode {ep+1}/{n_episodes}, "
                  f"successes so far: {successes}")

    success_rate = successes / n_episodes
    avg_steps = total_steps_success / max(successes, 1)
    avg_reward = total_reward / n_episodes

    return EvalResult(strategy_name, success_rate, avg_steps, avg_reward)


# ===== Plotting ==============================================================

def plot_results(results: list[EvalResult], losses_dict: dict[str, list[float]],
                 save_path: Path):
    """Create summary plots."""
    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = [r.name for r in results]
    colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#3498db', '#95a5a6']

    # 1. Success rate bar chart
    ax = axes[0]
    success_rates = [r.success_rate * 100 for r in results]
    bars = ax.bar(names, success_rates, color=colors[:len(results)])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Goal Reaching Success Rate")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha='center', va='bottom', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)

    # 2. Average steps bar chart
    ax = axes[1]
    avg_steps = [r.avg_steps for r in results]
    bars = ax.bar(names, avg_steps, color=colors[:len(results)])
    ax.set_ylabel("Avg Steps to Goal")
    ax.set_title("Steps to Goal (successful episodes)")
    for bar, val in zip(bars, avg_steps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha='center', va='bottom', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)

    # 3. Training loss curves
    ax = axes[2]
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("World Model Training Loss")
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.close()


# ===== Main ==================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--n-transitions", type=int, default=N_TRANSITIONS)
    parser.add_argument("--n-eval", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--skip-oracle", action="store_true",
                        help="Skip oracle (true env) evaluation (slow)")
    args = parser.parse_args()

    device = pick_device()
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("Grid World + CEM Planning Experiment")
    print("=" * 70)

    # ----- Data Generation -----
    print(f"\n[1/4] Generating {args.n_transitions} transitions...")
    t0 = time.time()
    states, action_fields, next_states = generate_transitions(
        args.n_transitions, device, rng)
    print(f"  Data shape: states={states.shape}, action_fields={action_fields.shape}, "
          f"next_states={next_states.shape}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ----- Train World Models -----
    print("\n[2/4] Training world models...")
    losses_dict = {}

    # 1. ResidualCorrection (CML) world model
    print("\n--- ActionConditionedResCor (CML) ---")
    rescor_model = ActionConditionedResCor(hidden_ch=32, cml_steps=10)
    pc = rescor_model.param_count()
    print(f"  Params: trained={pc['trained']}, frozen={pc['frozen']}")
    t0 = time.time()
    losses_rescor = train_world_model(rescor_model, states, action_fields,
                                       next_states, epochs=args.epochs, device=device)
    print(f"  Training time: {time.time()-t0:.1f}s")
    losses_dict["ResCor(CML)"] = losses_rescor

    # 2. PureNCA world model
    print("\n--- ActionConditionedNCA (PureNCA) ---")
    nca_model = ActionConditionedNCA(hidden_ch=32)
    pc = nca_model.param_count()
    print(f"  Params: trained={pc['trained']}, frozen={pc['frozen']}")
    t0 = time.time()
    losses_nca = train_world_model(nca_model, states, action_fields,
                                    next_states, epochs=args.epochs, device=device)
    print(f"  Training time: {time.time()-t0:.1f}s")
    losses_dict["PureNCA"] = losses_nca

    # 3. Conv2D baseline
    print("\n--- Conv2D Baseline ---")
    conv_model = Conv2DBaseline(hidden_ch=32)
    pc = conv_model.param_count()
    print(f"  Params: trained={pc['trained']}, frozen={pc['frozen']}")
    t0 = time.time()
    losses_conv = train_world_model(conv_model, states, action_fields,
                                     next_states, epochs=args.epochs, device=device)
    print(f"  Training time: {time.time()-t0:.1f}s")
    losses_dict["Conv2D"] = losses_conv

    # ----- Quick accuracy check on training data -----
    print("\n--- Training Accuracy Check ---")
    for name, model in [("ResCor(CML)", rescor_model),
                         ("PureNCA", nca_model),
                         ("Conv2D", conv_model)]:
        model.eval()
        with torch.no_grad():
            x = torch.cat([states[:500], action_fields[:500]], dim=1)
            logits = model(x)
            preds = logits.argmax(dim=1)
            targets = next_states[:500].argmax(dim=1)
            acc = (preds == targets).float().mean().item()
            print(f"  {name}: pixel accuracy = {acc:.4f}")

    # ----- Evaluate Planning Strategies -----
    print("\n[3/4] Evaluating planning strategies...")
    results = []

    # 1. CEM + ResCor
    print("\n--- CEM + ResCor(CML) ---")
    t0 = time.time()
    r1 = evaluate_strategy("CEM+ResCor", rescor_model, device, n_episodes=args.n_eval)
    print(f"  Time: {time.time()-t0:.1f}s")
    results.append(r1)

    # 2. CEM + PureNCA
    print("\n--- CEM + PureNCA ---")
    t0 = time.time()
    r2 = evaluate_strategy("CEM+NCA", nca_model, device, n_episodes=args.n_eval)
    print(f"  Time: {time.time()-t0:.1f}s")
    results.append(r2)

    # 3. CEM + Conv2D
    print("\n--- CEM + Conv2D ---")
    t0 = time.time()
    r3 = evaluate_strategy("CEM+Conv2D", conv_model, device, n_episodes=args.n_eval)
    print(f"  Time: {time.time()-t0:.1f}s")
    results.append(r3)

    # 4. CEM + True Env (oracle)
    if not args.skip_oracle:
        print("\n--- CEM + True Environment (Oracle) ---")
        t0 = time.time()
        r4 = evaluate_strategy("CEM+Oracle", None, device,
                                n_episodes=args.n_eval, use_true_env=True)
        print(f"  Time: {time.time()-t0:.1f}s")
        results.append(r4)

    # 5. Random actions
    print("\n--- Random Actions ---")
    t0 = time.time()
    r5 = evaluate_strategy("Random", None, device,
                            n_episodes=args.n_eval, random_policy=True)
    print(f"  Time: {time.time()-t0:.1f}s")
    results.append(r5)

    # ----- Results Summary -----
    print("\n" + "=" * 70)
    print("[4/4] RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'Success%':>10} {'AvgSteps':>10} {'AvgReward':>10}")
    print("-" * 52)
    for r in results:
        print(f"{r.name:<20} {r.success_rate*100:>9.1f}% {r.avg_steps:>10.1f} "
              f"{r.avg_reward:>10.3f}")

    # ----- Plot -----
    print("\n[4/4] Generating plots...")
    plot_path = PLOTS_DIR / "grid_world_planning.png"
    plot_results(results, losses_dict, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
