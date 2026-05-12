"""Smoke tests for the MSDC drift-conditioned multistep loss patch.

Runs 5 checks (CPU only, GPU is busy with combo A):
  1. Backwards-compat: msdc_alpha=0.0 (default) -> bit-identical training.
  2. Mutex: msdc_alpha + pushforward -> ValueError.
  3. Mutex: msdc_alpha + multistep_horizon=1 -> ValueError.
  4. Mutex: msdc_alpha on a non-gated model -> ValueError.
  5. Patch smoke: train rescor_mamba_gated_rand H=8 K_bptt=4 alpha=0.5 on
     heat × seed=42 × 2 epochs, verify no NaN, gate_scale/gate_bias
     move during training, loss stays finite.

Run:
    PYTHONPATH=src python experiments/_smoke_drift_gated_msdc.py
"""
from __future__ import annotations

import sys
import time
import traceback
from copy import deepcopy

import torch

from wmca.benchmarks import generate_heat
from wmca.model_registry import create_model, train_model


def _make_data(seed=42, grid=8, n_traj=8, n_steps=20, context_k=4, device="cpu"):
    return generate_heat(
        grid_size=grid, seed=seed,
        n_steps=n_steps, n_trajectories=n_traj,
        context_k=context_k, device=device,
    )


def _state_dict_close(a, b, tol=0.0):
    if set(a.keys()) != set(b.keys()):
        return False, f"key sets differ: {set(a.keys()) ^ set(b.keys())}"
    for k in a:
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            return False, f"{k}: shape differs"
        diff = (ta.float() - tb.float()).abs().max().item()
        if diff > tol:
            return False, f"{k}: max abs diff = {diff:.3e} > {tol}"
    return True, "ok"


def smoke_backwards_compat():
    """msdc_alpha=0.0 default => bit-identical state_dict."""
    print("[1/5] backwards-compat: msdc_alpha=0.0 default ...")
    data = _make_data(seed=42)
    meta = data.meta
    torch.manual_seed(42)
    m1 = create_model("rescor_mamba_gated_rand",
                      in_channels=meta["in_channels"],
                      out_channels=meta["out_channels"],
                      grid_size=8, seed=42)
    torch.manual_seed(42)
    m2 = create_model("rescor_mamba_gated_rand",
                      in_channels=meta["in_channels"],
                      out_channels=meta["out_channels"],
                      grid_size=8, seed=42)
    # Same init?
    ok, why = _state_dict_close(m1.state_dict(), m2.state_dict(), tol=0.0)
    assert ok, f"models init-different: {why}"

    # Train m1 with default kwargs (no msdc_alpha)
    torch.manual_seed(0)
    m1 = train_model(
        m1, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=2, batch_size=8, lr=1e-3, device="cpu",
        multistep_horizon=4, multistep_bptt=2, multistep_n_steps=20,
    )
    # Train m2 with msdc_alpha=0.0 explicit (should match)
    torch.manual_seed(0)
    m2 = train_model(
        m2, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=2, batch_size=8, lr=1e-3, device="cpu",
        multistep_horizon=4, multistep_bptt=2, multistep_n_steps=20,
        msdc_alpha=0.0,
    )
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    ok, why = _state_dict_close(sd1, sd2, tol=0.0)
    if not ok:
        # Expected: bit-identical (msdc_alpha=0.0 takes the same code path).
        # Allow tiny FP drift only if it really creeps in.
        ok2, why2 = _state_dict_close(sd1, sd2, tol=1e-7)
        if ok2:
            print(f"  PASS (within 1e-7 fp drift; msdc=0.0 path matched)")
            return True
        print(f"  FAIL: {why}")
        return False
    print("  PASS (bit-identical state_dict)")
    return True


def smoke_mutex_pushforward():
    """msdc_alpha > 0 + pushforward=True -> ValueError."""
    print("[2/5] mutex: msdc_alpha + pushforward ...")
    data = _make_data(seed=42)
    meta = data.meta
    m = create_model("rescor_mamba_gated_rand",
                     in_channels=meta["in_channels"],
                     out_channels=meta["out_channels"],
                     grid_size=8, seed=42)
    try:
        train_model(
            m, data.X_train, data.Y_train,
            X_val=data.X_val, Y_val=data.Y_val,
            loss_type=meta["loss_type"],
            epochs=1, batch_size=8, lr=1e-3, device="cpu",
            multistep_horizon=4, multistep_bptt=2,
            pushforward=True,
            msdc_alpha=0.5,
        )
    except ValueError as e:
        print(f"  PASS (raised ValueError: {e})")
        return True
    print("  FAIL: did not raise")
    return False


def smoke_mutex_h1():
    """msdc_alpha > 0 + multistep_horizon=1 -> ValueError."""
    print("[3/5] mutex: msdc_alpha + multistep_horizon=1 ...")
    data = _make_data(seed=42)
    meta = data.meta
    m = create_model("rescor_mamba_gated_rand",
                     in_channels=meta["in_channels"],
                     out_channels=meta["out_channels"],
                     grid_size=8, seed=42)
    try:
        train_model(
            m, data.X_train, data.Y_train,
            X_val=data.X_val, Y_val=data.Y_val,
            loss_type=meta["loss_type"],
            epochs=1, batch_size=8, lr=1e-3, device="cpu",
            multistep_horizon=1,  # default
            msdc_alpha=0.5,
        )
    except ValueError as e:
        print(f"  PASS (raised ValueError: {e})")
        return True
    print("  FAIL: did not raise")
    return False


def smoke_mutex_nongated():
    """msdc_alpha > 0 on a non-gated model (no compute_gate) -> ValueError."""
    print("[4/5] mutex: msdc_alpha on non-gated model ...")
    data = _make_data(seed=42)
    meta = data.meta
    # rescor_mamba_rand exists and lacks compute_gate.
    try:
        m = create_model("rescor_mamba_rand",
                         in_channels=meta["in_channels"],
                         out_channels=meta["out_channels"],
                         grid_size=8, seed=42)
    except Exception as e:
        # Fall back to rescor_rens (always available).
        print(f"  (rescor_mamba_rand unavailable: {e}; using rescor_rens)")
        m = create_model("rescor_rens",
                         in_channels=meta["in_channels"],
                         out_channels=meta["out_channels"],
                         grid_size=8, seed=42)
    if hasattr(m, "compute_gate"):
        print("  FAIL: chosen non-gated model unexpectedly has compute_gate")
        return False
    try:
        train_model(
            m, data.X_train, data.Y_train,
            X_val=data.X_val, Y_val=data.Y_val,
            loss_type=meta["loss_type"],
            epochs=1, batch_size=8, lr=1e-3, device="cpu",
            multistep_horizon=4, multistep_bptt=2, multistep_n_steps=20,
            msdc_alpha=0.5,
        )
    except ValueError as e:
        print(f"  PASS (raised ValueError: {e})")
        return True
    print("  FAIL: did not raise")
    return False


def smoke_patch_train():
    """Train H=8 K_bptt=4 alpha=0.5 on heat 2 epochs; check finite, gate moves."""
    print("[5/5] patch smoke: H=8 K_bptt=4 alpha=0.5 heat 2ep ...")
    data = _make_data(seed=42, grid=8, n_traj=8, n_steps=20, context_k=4)
    meta = data.meta
    m = create_model("rescor_mamba_gated_rand",
                     in_channels=meta["in_channels"],
                     out_channels=meta["out_channels"],
                     grid_size=8, seed=42)
    with torch.no_grad():
        m.gate_bias.data.fill_(1.0)
    gate_scale_init = float(m.gate_scale.item())
    gate_bias_init = float(m.gate_bias.item())
    t0 = time.time()
    try:
        m = train_model(
            m, data.X_train, data.Y_train,
            X_val=data.X_val, Y_val=data.Y_val,
            loss_type=meta["loss_type"],
            epochs=2, batch_size=8, lr=1e-3, device="cpu",
            multistep_horizon=8, multistep_bptt=4, multistep_n_steps=20,
            msdc_alpha=0.5,
        )
    except Exception as e:
        print(f"  FAIL: training raised {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    elapsed = time.time() - t0
    gate_scale_final = float(m.gate_scale.item())
    gate_bias_final = float(m.gate_bias.item())
    # Forward to verify no NaN.
    m.eval()
    with torch.no_grad():
        out = m(data.X_test[:4])
    if not torch.isfinite(out).all():
        print("  FAIL: forward output non-finite")
        return False
    moved_scale = abs(gate_scale_final - gate_scale_init)
    moved_bias = abs(gate_bias_final - gate_bias_init)
    moved = (moved_scale > 1e-6) or (moved_bias > 1e-6)
    print(f"  elapsed={elapsed:.1f}s  finite=ok  "
          f"gate_scale {gate_scale_init:.4f}->{gate_scale_final:.4f} "
          f"(d={moved_scale:.2e})  "
          f"gate_bias {gate_bias_init:.4f}->{gate_bias_final:.4f} "
          f"(d={moved_bias:.2e})")
    if not moved:
        print("  FAIL: gate params did not move")
        return False
    print("  PASS")
    return True


def main():
    results = []
    results.append(("backwards-compat", smoke_backwards_compat()))
    results.append(("mutex pushforward", smoke_mutex_pushforward()))
    results.append(("mutex H=1", smoke_mutex_h1()))
    results.append(("mutex non-gated", smoke_mutex_nongated()))
    results.append(("patch train", smoke_patch_train()))
    print()
    print("=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")
    print(f"\n{n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
