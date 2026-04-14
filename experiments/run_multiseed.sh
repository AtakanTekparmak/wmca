#!/bin/bash
# Run unified ablation for 3 seeds, saving results per seed then averaging.
# Usage: bash experiments/run_multiseed.sh

MODELS="mlp rescor rescor_e3c rescor_mp_gate pure_nca conv2d"
BENCHMARKS="heat gol ks gray_scott rule110 wireworld crafter_lite minigrid autumn_disease autumn_gravity autumn_water atari_pong atari_breakout dmcontrol"

for SEED in 42 43 44; do
    echo "=== Running seed $SEED ==="
    uv run --with scikit-learn,matplotlib,scipy python experiments/unified_ablation.py \
        --models $MODELS \
        --benchmarks $BENCHMARKS \
        --seed $SEED \
        --no-wandb

    # Save per-seed results
    cp experiments/results/unified_ablation.json \
       experiments/results/unified_ablation_seed${SEED}.json
    echo "=== Seed $SEED complete ==="
done

echo "All seeds done. Results saved as unified_ablation_seed{42,43,44}.json"
