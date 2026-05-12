#!/bin/bash
cd /Users/atakantekparmak/Desktop/personal/research/wmca
source .venv/bin/activate
echo "=== STEP 1: Generate 500-trajectory Atari data ==="
PYTHONPATH=src python3 experiments/_cmdr_atari_data_v2.py
echo ""
echo "=== STEP 2: Train AE + encode latents ==="
PYTHONPATH=src python3 experiments/_cmdr_atari_ae_v2.py
echo ""
echo "=== DONE ==="
