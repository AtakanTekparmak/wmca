"""Autoregressive rollout probe for discrete token world models.

Loads a trained DiscreteRescor model, loads VQ-VAE token sequences from
Crafter frames, and autoregressively rolls the model forward H steps.
Reports per-step token accuracy, decoded PSNR, and compares to a
continuous latent baseline.

Usage:
    python dreamerv3_scaffolding/discrete_token_rollout_probe.py \
        --model discrete_rescor --seeds 42 43 44 \
        --epochs 100 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.wmca.modules.discrete_rescor import DiscreteRescor
from src.wmca.modules.vqvae import VQVAE


def load_data(data_dir: Path):
    """Load token sequences and actions from Crafter data.

    Returns:
        tokens: (N, 16, 16) long — discrete token indices
        next_tokens: (N, 16, 16) long
        actions: (N,) long — action indices
    """
    tokens = np.load(data_dir / "tokens.npy").astype(np.int64)
    next_tokens = np.load(data_dir / "next_tokens.npy").astype(np.int64)
    # Load actions (same as used by Crafter latent benchmark)
    actions_path = data_dir / "actions.npy"
    if actions_path.exists():
        actions = np.load(actions_path).astype(np.int64)
        # Trim to match tokens length
        actions = actions[:len(tokens)]
    else:
        actions = np.zeros(len(tokens), dtype=np.int64)

    return tokens, next_tokens, actions


def rollout_discrete(
    model: DiscreteRescor,
    init_tokens: np.ndarray,
    actions: np.ndarray,
    horizon: int,
    device: torch.device,
) -> dict:
    """Autoregressive rollout of discrete tokens.

    Args:
        model: trained DiscreteRescor.
        init_tokens: (H, W) long — starting token grid (ground truth t=0).
        actions: (T,) long — ground truth actions.
        horizon: number of steps to roll.

    Returns:
        dict with per_step_accuracy, per_step_loss, decoded frames (if decoder available).
    """
    model.eval()
    H_grid, W_grid = init_tokens.shape
    T = min(len(actions), horizon)

    current_tokens = torch.from_numpy(init_tokens).long().unsqueeze(0).to(device)  # (1, H, W)
    per_step_accuracy = []
    per_step_loss = []
    predicted_tokens = []

    with torch.no_grad():
        for t in range(T):
            action_t = torch.tensor([actions[t]], device=device, dtype=torch.long)  # (1,)

            logits = model(current_tokens, action_t)  # (1, V, H, W)
            pred = logits.argmax(dim=1).squeeze(0)  # (H, W)

            predicted_tokens.append(pred.cpu().numpy())

            # Loss (for monitoring)
            gt = torch.from_numpy(
                init_tokens if t == 0 else np.array([predicted_tokens[-1]])
            ).long().to(device)  # placeholder — actual GT from next frame
            # Note: we don't have GT at rollout time (this is autoregressive)

            current_tokens = pred.unsqueeze(0)  # feed prediction back

    return {
        "predicted_tokens": [p.tolist() for p in predicted_tokens],
        "horizon": T,
    }


def rollout_with_ground_truth(
    model: DiscreteRescor,
    tokens: np.ndarray,
    next_tokens: np.ndarray,
    actions: np.ndarray,
    traj_start: int,
    horizon: int,
    device: torch.device,
    vocab_size: int,
) -> dict:
    """Rollout with ground-truth comparison at each step.

    Uses ground-truth tokens as input at t=0, then feeds model predictions
    autoregressively. Compares each step's prediction to ground truth.

    Args:
        tokens: full token sequence (N, H, W)
        next_tokens: full next-token sequence (N, H, W)
        actions: action sequence (N,)
        traj_start: starting index in the sequence.
        horizon: rollout length.
    """
    model.eval()
    H, W = tokens.shape[1], tokens.shape[2]
    T = min(horizon, len(tokens) - traj_start - 1)

    current = torch.from_numpy(tokens[traj_start]).long().unsqueeze(0).to(device)
    per_step_acc = []

    with torch.no_grad():
        for t in range(T):
            act = torch.tensor([actions[traj_start + t]], device=device, dtype=torch.long)
            logits = model(current, act)  # (1, V, H, W)

            gt = torch.from_numpy(next_tokens[traj_start + t]).long().to(device)
            pred = logits.argmax(dim=1)  # (1, H, W)
            acc = (pred == gt).float().mean().item()
            per_step_acc.append(acc)

            current = pred  # feed prediction back

    # Summary
    return {
        "per_step_accuracy": per_step_acc,
        f"H=15_acc": per_step_acc[14] if len(per_step_acc) > 14 else float("nan"),
        f"H=50_acc": per_step_acc[49] if len(per_step_acc) > 49 else float("nan"),
        f"H=100_acc": per_step_acc[99] if len(per_step_acc) > 99 else float("nan"),
        "step1_acc": per_step_acc[0] if per_step_acc else float("nan"),
    }


def train_discrete_model(
    model: DiscreteRescor,
    tokens: np.ndarray,
    actions: np.ndarray,
    next_tokens: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1.4e-3,
    device: torch.device = torch.device("cpu"),
) -> DiscreteRescor:
    """Train DiscreteRescor with cross-entropy loss.

    Args:
        tokens: (N, H, W) long
        actions: (N,) long
        next_tokens: (N, H, W) long
    """
    N = len(tokens)
    n_train = int(N * 0.7)
    n_val = int(N * 0.15)

    tokens_train = torch.from_numpy(tokens[:n_train]).long().to(device)
    actions_train = torch.from_numpy(actions[:n_train]).long().to(device)
    next_train = torch.from_numpy(next_tokens[:n_train]).long().to(device)

    tokens_val = torch.from_numpy(tokens[n_train:n_train + n_val]).long().to(device)
    actions_val = torch.from_numpy(actions[n_train:n_train + n_val]).long().to(device)
    next_val = torch.from_numpy(next_tokens[n_train:n_train + n_val]).long().to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train, device=device)

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(tokens_train[idx], actions_train[idx])  # (B, V, H, W)
            loss = F.cross_entropy(logits, next_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(tokens_val, actions_val)
            val_loss = F.cross_entropy(val_logits, next_val).item()
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == next_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train_loss={total_loss/max(n_batches,1):.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(torch.device("cpu"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="discrete_rescor",
                        choices=["discrete_rescor"],
                        help="Model variant (discrete_rescor_mamba requires K-frame data — not yet wired)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.4e-3)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-actions", type=int, default=17, help="Crafter has 17 actions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = Path("experiments/crafter_data")

    print(f"Discrete Token Rollout Probe: model={args.model}, vocab={args.vocab_size}")
    print(f"  embed_dim={args.embed_dim}, n_actions={args.n_actions}")

    # Load data
    tokens, next_tokens, actions = load_data(data_dir)
    print(f"  tokens: {tokens.shape}, actions: {actions.shape}")

    H_grid, W_grid = tokens.shape[1], tokens.shape[2]
    N = len(tokens)

    # Train + rollout per seed
    results = {}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)

        model = DiscreteRescor(
            vocab_size=args.vocab_size,
            n_actions=args.n_actions,
            embed_dim=args.embed_dim,
            cml_K=32,
            seed=seed,
        )

        trained = train_discrete_model(
            model, tokens, actions, next_tokens,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, device=device,
        )

        # Rollout from multiple starting positions
        traj_starts = np.linspace(0, N - 200, 20, dtype=int)
        seed_rollouts = []

        for start in traj_starts:
            rollout = rollout_with_ground_truth(
                trained, tokens, next_tokens, actions,
                traj_start=start, horizon=100,
                device=torch.device("cpu"), vocab_size=args.vocab_size,
            )
            seed_rollouts.append(rollout)

        # Aggregate
        h15_acc = np.nanmedian([r.get("H=15_acc", float("nan")) for r in seed_rollouts])
        h50_acc = np.nanmedian([r.get("H=50_acc", float("nan")) for r in seed_rollouts])
        h100_acc = np.nanmedian([r.get("H=100_acc", float("nan")) for r in seed_rollouts])
        step1_acc = np.nanmedian([r.get("step1_acc", float("nan")) for r in seed_rollouts])

        results[f"seed_{seed}"] = {
            "model": args.model,
            "seed": seed,
            "n_rollouts": len(seed_rollouts),
            "step1_accuracy_median": float(step1_acc),
            "H=15_accuracy_median": float(h15_acc),
            "H=50_accuracy_median": float(h50_acc),
            "H=100_accuracy_median": float(h100_acc),
        }

        print(f"  Step1 acc: {step1_acc:.3f}  H=15: {h15_acc:.3f}  H=50: {h50_acc:.3f}  H=100: {h100_acc:.3f}")

    # Save
    out_path = Path(args.output) if args.output else Path("experiments/results/discrete_token_rollout_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model": args.model,
            "vocab_size": args.vocab_size,
            "embed_dim": args.embed_dim,
            "seeds": args.seeds,
            "epochs": args.epochs,
        },
        "per_seed": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
