"""Shape-only dry-run for the scale-up scaffolds.

This script instantiates both `RescorMSPyramid` and `ResCorformer` with the
DEFAULT stubs (no `wmca.modules` imports needed) and verifies:
  1. forward pass returns the correct output shape
  2. values remain finite and in [0, 1]
  3. trained/frozen param counts are reported

It performs no training, no oracle lookup, no benchmark. Run with:
    python scaleup_scaffolding/test_shapes.py

The script is intentionally standalone — it does NOT import anything from
`wmca` so a running experiment cannot be broken by executing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the scaffold dir importable regardless of CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rescor_ms import RescorMSPyramid  # noqa: E402
from rescorformer import ResCorformer  # noqa: E402


def _check_tensor(name: str, t: torch.Tensor, expected_shape: tuple) -> None:
    assert t.shape == expected_shape, f"{name}: got {tuple(t.shape)}, expected {expected_shape}"
    assert torch.isfinite(t).all(), f"{name}: non-finite values detected"
    lo, hi = float(t.min()), float(t.max())
    assert 0.0 - 1e-5 <= lo and hi <= 1.0 + 1e-5, f"{name}: out of [0,1], min={lo} max={hi}"
    print(f"  OK  {name}: shape={tuple(t.shape)} range=[{lo:.4f}, {hi:.4f}]")


def test_rescor_ms() -> None:
    print("\n[1/2] RescorMSPyramid (64x64 U-Net with stubbed CMLs)")
    model = RescorMSPyramid()  # all stubs
    pc = model.param_count()
    print(f"  params trained={pc['trained']}  frozen={pc['frozen']}")

    x = torch.rand(2, 1, 64, 64)
    y = model(x)
    _check_tensor("rescor_ms output", y, (2, 1, 64, 64))

    # Also verify a batch-of-1 and a batch-of-4 both work (no accidental batch hardcoding).
    for b in (1, 4):
        y_b = model(torch.rand(b, 1, 64, 64))
        _check_tensor(f"rescor_ms batch={b}", y_b, (b, 1, 64, 64))


def test_rescorformer() -> None:
    print("\n[2/2] ResCorformer (L=1, 4x4 patches, window=8, stubbed CML)")
    model = ResCorformer()
    pc = model.param_count()
    print(f"  params trained={pc['trained']}  frozen={pc['frozen']}")
    print(f"  n_tokens={model.n_tokens} head_dim={model.head_dim} window={model.window}")

    x = torch.rand(2, 1, 64, 64)
    y = model(x)
    _check_tensor("rescorformer output", y, (2, 1, 64, 64))

    # Intermediate shape sanity: patchify and unpatchify roundtrip.
    patches = model._patchify(x)
    assert patches.shape == (2, model.n_tokens, model.patch_dim), \
        f"patch shape mismatch: {tuple(patches.shape)}"
    round_trip = model._unpatchify(patches)
    assert torch.allclose(round_trip, x), "patchify/unpatchify is not the identity"
    print("  OK  patchify/unpatchify roundtrip is bit-identical")

    for b in (1, 4):
        y_b = model(torch.rand(b, 1, 64, 64))
        _check_tensor(f"rescorformer batch={b}", y_b, (b, 1, 64, 64))


def main() -> None:
    torch.manual_seed(0)
    print("=== scale-up scaffolding shape dry-run ===")
    test_rescor_ms()
    test_rescorformer()
    print("\nAll shape checks passed.")


if __name__ == "__main__":
    main()
