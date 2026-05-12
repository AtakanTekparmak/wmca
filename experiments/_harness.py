"""Standing experiment harness -- multi-seed ablation protocol (post 2026-04-22).

Why this module exists
----------------------
After the 2026-04-22 demotion we retired single-seed reporting. Several earlier
"wins" (notably the S56 epoch-diagnostic) turned out to be seed-luck artefacts:
single-seed MSE numbers looked 5-10x better than the honest 3-seed median. The
standing methodology adopted at that review is:

  1. ALL ablations must run >= 3 seeds. 5 seeds preferred when compute allows.
     ``run_ablation`` asserts this as a hard rail at start.
  2. ALL ablations must PRE-REGISTER a winner criterion BEFORE launching the
     run. The criterion is persisted on disk (``protocol.pre_registered`` in
     the output JSON) by ``pre_register(out_path, ...)`` up-front, which is
     the protocol's defence against post-hoc criterion drift.
  3. ALL ablations must report median + mean + std + min + max + n per cell.
     No single-seed callouts. No cherry-picked seeds in summary prose.
  4. Epochs default 100; hard-warn below 60 per the S56 methodology caveat.
  5. Incremental JSON persistence after every cell (resume-from-crash).

Rather than re-copy this ceremony into every ablation script (and risk drift),
every future script imports this harness. The reference implementation that
this module generalises is ``experiments/phase1_honest_baseline.py``; the
summary-table format matches that script line-for-line so downstream tooling
(plots, win-rate aggregators) keeps working.

Usage
-----
::

    from pathlib import Path
    from _harness import run_ablation, pre_register, print_summary_table
    from wmca.benchmarks import generate_heat, generate_gol

    BENCHMARKS = {"heat": generate_heat, "gol": generate_gol}
    VARIANTS   = ["rescor_rens", "rescor_rens_stat_full"]
    SEEDS      = [42, 43, 44, 45, 46]
    OUT        = Path("experiments/results/my_ablation.json")

    pre_register(OUT, {
        "heat": "median stat_full beats rens by >= 2x (MSE)",
        "gol":  "median stat_full beats rens by >= 1.0pp (accuracy)",
    })

    results = run_ablation(
        variants=VARIANTS, benchmarks=BENCHMARKS, seeds=SEEDS,
        epochs=100, K=32, out_path=OUT,
    )
    print_summary_table(results, VARIANTS, list(BENCHMARKS))

Note
----
This module intentionally has no external network calls and no global state
beyond a process-local ``_PRE_REGISTERED`` cache (purely a convenience so that
``run_ablation`` can re-emit the block if the output JSON has been rewritten).
It is import-safe: ``wmca`` is lazy-imported inside the per-cell runner.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import torch


# --------------------------------------------------------------------------- #
#  Protocol constants -- the rails that enforce post-demotion methodology.    #
# --------------------------------------------------------------------------- #

#: Hard minimum seed count.  Going below this is a methodology violation and
#: ``run_ablation`` will raise before any training starts.
MIN_SEEDS = 3

#: Below this epoch count the harness prints a loud caveat banner referencing
#: the S56 diagnostic (short training regimes invert CML-ablation rankings).
EPOCH_WARNING_THRESHOLD = 60

#: Harness identifier written into every output JSON's protocol block, so old
#: result files can be audited / upgraded later without guessing the format.
HARNESS_VERSION = "2026-04-22-standing"


# --------------------------------------------------------------------------- #
#  Pre-registration -- win criterion must be persisted BEFORE the run starts. #
# --------------------------------------------------------------------------- #

#: Process-local cache of the most recently pre-registered criterion. Used
#: only so ``run_ablation`` can re-inject it when it rewrites the output JSON;
#: the ground truth is always the on-disk ``protocol.pre_registered`` block.
_PRE_REGISTERED: Dict[str, Dict[str, Any]] = {}


def pre_register(out_path: str | Path,
                 criterion_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Persist the pre-registered winner criterion to ``out_path`` up-front.

    Writes (or merges into) a ``protocol.pre_registered`` block keyed by
    benchmark name.  Must be called BEFORE ``run_ablation``.  If the criterion
    is not on disk before the training GPU warms up, the run does not count.

    Arguments
    ---------
    out_path :
        Destination JSON path -- same path you will pass to ``run_ablation``.
    criterion_dict :
        ``{benchmark: rule}`` where ``rule`` is either a free-form string
        (e.g. ``"median gap >= 2x on MSE"``) or a small dict with keys like
        ``{"metric": "mse", "threshold": "..."}``. Stored verbatim.

    Returns
    -------
    The normalised ``{benchmark: {"rule": ...}}`` dict that was written.
    """
    if not isinstance(criterion_dict, Mapping):
        raise TypeError("pre_register expects a dict-like mapping")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise every value into a dict with a "rule" field for consistency.
    normalised: Dict[str, Dict[str, Any]] = {}
    for bn, rule in criterion_dict.items():
        if isinstance(rule, Mapping):
            normalised[bn] = dict(rule)
            normalised[bn].setdefault("rule", rule.get("threshold",
                                                       rule.get("rule", "")))
        else:
            normalised[bn] = {"rule": str(rule)}

    # Merge into any existing protocol block on disk so repeated calls compose.
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    protocol = existing.setdefault("protocol", {})
    pre = protocol.setdefault("pre_registered", {})
    pre.update(normalised)
    protocol.setdefault("harness_version", HARNESS_VERSION)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    # Refresh in-memory cache so run_ablation picks up this criterion when
    # it rewrites the file.
    _PRE_REGISTERED.clear()
    _PRE_REGISTERED.update(normalised)
    return dict(normalised)


# --------------------------------------------------------------------------- #
#  Formatting / stats helpers -- match phase1_honest_baseline.py exactly.     #
# --------------------------------------------------------------------------- #

def fmt_score(score: Optional[float], metric: str) -> str:
    """Format a scalar score. MSE -> sci notation; BCE/accuracy -> percent.

    ``None`` -> literal ``"FAILED"`` so failure cells render consistently in
    tables without special-casing every call site.
    """
    if score is None:
        return "FAILED"
    return f"{score:.4e}" if metric == "mse" else f"{score * 100:.2f}%"


def stats_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    """Return ``{mean, median, std, min, max, n}`` over an iterable of floats.

    Empty / all-None input returns all-None fields (except ``n=0``) so the
    pretty-printer can treat missing cells uniformly.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "median": None, "std": None,
                "min": None, "max": None, "n": 0}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(var)
    return {"mean": mean, "median": median(vals), "std": std,
            "min": min(vals), "max": max(vals), "n": n}


# --------------------------------------------------------------------------- #
#  Internal: single-cell train + eval.                                        #
# --------------------------------------------------------------------------- #

def _evaluate(model, X_test, Y_test, meta) -> float:
    """Benchmark-aware single-step eval -- mirrors phase1_honest_baseline."""
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
    if meta["metric"] == "mse":
        return ((preds - Y_test) ** 2).mean().item()
    if meta.get("loss_type") in ("ce", "cross_entropy"):
        pred_classes = preds.argmax(dim=1)
        if Y_test.dim() == 4 and Y_test.shape[1] > 1:
            true_classes = Y_test.argmax(dim=1)
        else:
            true_classes = Y_test.long().squeeze(1)
        return (pred_classes == true_classes).float().mean().item()
    pred_binary = (preds > 0.5).float()
    return (pred_binary == Y_test).float().mean().item()


def _train_and_eval(variant: str, bench_name: str, gen_fn: Callable,
                    seed: int, epochs: int, K: int,
                    batch_size: int, lr: float,
                    cml_K_fn: Optional[Callable[[str], int]]) -> Dict[str, Any]:
    """Build the model via the registry, train it, evaluate it, return cell."""
    # Lazy import so merely importing the harness does not force a wmca load.
    from wmca.model_registry import create_model, train_model

    data = gen_fn(grid_size=16, seed=seed)
    meta = data.meta
    effective_K = cml_K_fn(variant) if cml_K_fn is not None else K
    m = create_model(
        variant,
        in_channels=meta["in_channels"], out_channels=meta["out_channels"],
        grid_size=16, seed=seed, cml_K=effective_K,
    )
    pc = m.param_count() if hasattr(m, "param_count") else {}
    t0 = time.time()
    m = train_model(
        m, data.X_train, data.Y_train,
        X_val=data.X_val, Y_val=data.Y_val,
        loss_type=meta["loss_type"],
        epochs=epochs, batch_size=batch_size, lr=lr,
    )
    score = _evaluate(m, data.X_test, data.Y_test, meta)
    return {"score": score, "metric": meta["metric"], "params": pc,
            "time": time.time() - t0, "seed": seed, "variant": variant,
            "benchmark": bench_name, "epochs": epochs, "K": effective_K}


# --------------------------------------------------------------------------- #
#  Incremental persistence helpers.                                           #
# --------------------------------------------------------------------------- #

def _load_resume(out_path: Path, variants: List[str],
                 bench_names: Iterable[str]) -> tuple[Dict, int]:
    """Initialise results dict, populating completed cells from prior JSON.

    Handles both the legacy top-level shape ``{variant: {bench: {seed: cell}}}``
    and the canonical ``{protocol, per_seed, summary}`` shape.
    """
    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        v: {bn: {} for bn in bench_names} for v in variants
    }
    if not out_path.exists():
        return results, 0

    try:
        with open(out_path, "r") as f:
            prior = json.load(f)
    except Exception as e:
        print(f"[resume] failed to parse {out_path} ({e}); starting fresh")
        return results, 0

    cell_src = prior.get("per_seed", prior)
    n_loaded = 0
    for v in variants:
        if v not in cell_src or not isinstance(cell_src[v], dict):
            continue
        for bn in bench_names:
            if bn not in cell_src[v]:
                continue
            for seed_key, cell in cell_src[v][bn].items():
                if isinstance(cell, dict) and cell.get("score") is not None:
                    results[v][bn][seed_key] = cell
                    n_loaded += 1
    return results, n_loaded


def _dump_incremental(out_path: Path, results: Dict, variants: List[str],
                      seeds: List[int], epochs: int, K: int,
                      bench_names: List[str], partial: bool = True) -> None:
    """Write partial results preserving any pre-registered criterion on disk."""
    # Preserve an existing ``protocol.pre_registered`` block if the user called
    # ``pre_register`` earlier -- that file now lives on disk and we must not
    # clobber it.
    prior_pre: Dict[str, Any] = dict(_PRE_REGISTERED)
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                prior = json.load(f)
            disk_pre = prior.get("protocol", {}).get("pre_registered", {})
            # Disk wins over in-memory so pre_register calls before the
            # run are authoritative.
            if disk_pre:
                prior_pre = dict(disk_pre)
        except Exception:
            pass

    payload = {
        "protocol": {
            "variants": list(variants), "seeds": list(seeds),
            "epochs": epochs, "K": K, "benchmarks": list(bench_names),
            "pre_registered": prior_pre,
            "min_seeds_rail": MIN_SEEDS,
            "epoch_warning_threshold": EPOCH_WARNING_THRESHOLD,
            "harness_version": HARNESS_VERSION,
            "partial": partial,
        },
        "per_seed": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------- #
#  Main entry point.                                                          #
# --------------------------------------------------------------------------- #

def run_ablation(
    variants: List[str],
    benchmarks: Mapping[str, Callable],
    seeds: List[int],
    epochs: int,
    K: int,
    out_path: str | Path,
    batch_size: int = 64,
    lr: float = 1e-3,
    skip_existing: bool = True,
    cml_K_fn: Optional[Callable[[str], int]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """Run a multi-seed ablation with the standing protocol.

    Enforces the post-2026-04-22 rules:
      * ``len(seeds) >= MIN_SEEDS`` (asserted -- raises before any training).
      * ``epochs < EPOCH_WARNING_THRESHOLD`` -> loud S56 warning banner.
      * Writes a full ``protocol`` block to ``out_path`` on first save,
        preserving any ``pre_registered`` criterion written by ``pre_register``.
      * Incremental JSON save after every benchmark cell (crash-safe).
      * Per-cell exceptions are caught and recorded as ``score=None`` cells;
        the run continues to the next cell.

    Returns nested dict ``{variant: {benchmark: {seed_str: cell}}}``.
    """
    # ---- Hard rails ----
    assert len(seeds) >= MIN_SEEDS, (
        f"Standing protocol requires len(seeds) >= {MIN_SEEDS}; got "
        f"{len(seeds)}. If compute is tight, prefer fewer benchmarks over "
        "fewer seeds -- single-seed callouts are forbidden post-2026-04-22."
    )
    if epochs < EPOCH_WARNING_THRESHOLD:
        print("!" * 78)
        print(f"!! WARNING: epochs={epochs} < {EPOCH_WARNING_THRESHOLD}. "
              "Per the S56 methodology caveat,")
        print("!! short training regimes can invert ablation rankings. "
              "Results from")
        print("!! this run MUST NOT be reported as definitive without a "
              "100-epoch")
        print("!! follow-up on the winning variant.")
        print("!" * 78)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bench_names = list(benchmarks.keys())

    # ---- Resume from prior incremental JSON ----
    if skip_existing:
        results, n_loaded = _load_resume(out_path, variants, bench_names)
        if n_loaded:
            print(f"[resume] loaded {n_loaded} completed cells from {out_path}")
    else:
        results = {v: {bn: {} for bn in bench_names} for v in variants}

    # Write the protocol block immediately. This also ensures that the
    # pre_registered block (if any) survives the very first _dump_incremental
    # call below.
    _dump_incremental(out_path, results, variants, seeds, epochs, K,
                      bench_names, partial=True)

    # ---- Main grid ----
    for variant in variants:
        for seed in seeds:
            print("=" * 78)
            print(f"{variant}   K={K}   seed={seed}   epochs={epochs}")
            print("=" * 78)
            for bn, gf in benchmarks.items():
                existing = results[variant][bn].get(str(seed), {})
                if skip_existing and existing.get("score") is not None:
                    print(f"  {bn:15s}  "
                          f"{fmt_score(existing['score'], existing['metric']):>14s}"
                          f"  [resumed]  params={existing.get('params', '?')}")
                    continue
                try:
                    r = _train_and_eval(variant, bn, gf, seed, epochs, K,
                                        batch_size, lr, cml_K_fn)
                    results[variant][bn][str(seed)] = r
                    print(f"  {bn:15s}  "
                          f"{fmt_score(r['score'], r['metric']):>14s}  "
                          f"[{r['time']:.0f}s]  params={r['params']}")
                except Exception as e:
                    print(f"  {bn:15s}  FAILED: {type(e).__name__}: {e}")
                    results[variant][bn][str(seed)] = {
                        "score": None, "metric": "?",
                        "error": f"{type(e).__name__}: {e}",
                        "seed": seed, "variant": variant, "benchmark": bn,
                    }
                _dump_incremental(out_path, results, variants, seeds, epochs,
                                  K, bench_names, partial=True)
                gc.collect()
            print()

    # ---- Final dump with aggregated summary (partial=False) ----
    summary = {v: {bn: stats_summary([r["score"]
                                      for r in results[v][bn].values()
                                      if r.get("score") is not None])
                   for bn in bench_names}
               for v in variants}
    # Preserve pre-registered block from disk.
    prior_pre: Dict[str, Any] = dict(_PRE_REGISTERED)
    if out_path.exists():
        try:
            with open(out_path, "r") as f:
                disk_pre = json.load(f).get("protocol", {}).get(
                    "pre_registered", {})
            if disk_pre:
                prior_pre = dict(disk_pre)
        except Exception:
            pass

    final = {
        "protocol": {
            "variants": list(variants), "seeds": list(seeds),
            "epochs": epochs, "K": K, "benchmarks": bench_names,
            "pre_registered": prior_pre,
            "min_seeds_rail": MIN_SEEDS,
            "epoch_warning_threshold": EPOCH_WARNING_THRESHOLD,
            "harness_version": HARNESS_VERSION,
            "partial": False,
        },
        "per_seed": results,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# --------------------------------------------------------------------------- #
#  Summary printer -- output shape matches phase1_honest_baseline.             #
# --------------------------------------------------------------------------- #

def _metric_for(bn: str) -> str:
    """Per-benchmark metric convention used across the wmca codebase."""
    return "mse" if bn in ("heat", "gray_scott", "ks") else "bce"


def _verdict(other_med: Optional[float], base_med: Optional[float],
             metric: str) -> str:
    """Pre-registered win rule: 2x for MSE, 1.0pp for accuracy."""
    if other_med is None or base_med is None:
        return "n/a"
    if metric == "mse":
        if other_med <= 0:
            return "n/a"
        ratio = base_med / other_med
        if ratio > 2.0:
            return f"WIN ({ratio:.2f}x better)"
        if ratio < 0.5:
            return f"LOSS ({1/ratio:.2f}x worse)"
        return f"tie ({ratio:.2f}x)"
    diff_pp = (other_med - base_med) * 100
    if diff_pp > 1.0:
        return f"WIN (+{diff_pp:.2f}pp)"
    if diff_pp < -1.0:
        return f"LOSS ({diff_pp:.2f}pp)"
    return f"tie ({diff_pp:+.2f}pp)"


def print_summary_table(
    results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    variants: List[str],
    benchmarks: List[str],
    baselines: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Print per-variant stat block + pre-registered win-check vs ``variants[0]``.

    Format matches ``experiments/phase1_honest_baseline.py`` so downstream
    tooling that scrapes the block (plots, rank aggregators) keeps working.

    ``baselines`` is an optional ``{label: {benchmark: score}}`` dict of
    single-seed reference anchors. Printed AS-IS with a clear "single-seed,
    directional only" caveat (no multi-seed claim is made for them).
    """
    # n_seeds taken from whichever cell actually populated first; safe default 0
    n_seeds = 0
    for v in variants:
        for bn in benchmarks:
            cand = results.get(v, {}).get(bn, {})
            if cand:
                n_seeds = max(n_seeds, len(cand))

    print("=" * 120)
    print(f"SUMMARY -- standing-protocol ablation ({n_seeds} seeds)")
    print("=" * 120)

    summary = {v: {bn: stats_summary([r["score"]
                                      for r in results.get(v, {})
                                      .get(bn, {}).values()
                                      if isinstance(r, dict)
                                      and r.get("score") is not None])
                   for bn in benchmarks}
               for v in variants}

    # ---- Per-variant block ----
    for variant in variants:
        print()
        print(f"-- {variant} " + "-" * max(4, 60 - len(variant)))
        hdr = (f"{'bench':14s}  {'median':>14s}  {'mean':>14s}  "
               f"{'std':>12s}  {'min':>14s}  {'max':>14s}  n")
        print(hdr)
        print("-" * len(hdr))
        for bn in benchmarks:
            s = summary[variant][bn]
            metric = _metric_for(bn)

            def f(v, m=metric):
                return fmt_score(v, m) if v is not None else "n/a"

            if s["std"] is None:
                std_str = "n/a"
            elif metric == "mse":
                std_str = f"{s['std']:.2e}"
            else:
                std_str = f"{s['std']*100:.2f}pp"
            print(f"{bn:14s}  {f(s['median']):>14s}  {f(s['mean']):>14s}  "
                  f"{std_str:>12s}  {f(s['min']):>14s}  {f(s['max']):>14s}  "
                  f"{s['n']}")

    # ---- Pre-registered win check (medians vs variants[0]) ----
    if len(variants) >= 2:
        print()
        print("=" * 120)
        print(f"PRE-REGISTERED WIN CHECK -- medians vs {variants[0]}, "
              "2x for MSE / 1.0pp for accuracy")
        print("=" * 120)
        others = variants[1:]
        hdr = f"{'bench':14s}  " + "  ".join(
            f"{o + ' vs ' + variants[0]:>26s}" for o in others)
        print(hdr)
        print("-" * len(hdr))
        for bn in benchmarks:
            base_m = summary[variants[0]][bn]["median"]
            metric = _metric_for(bn)
            cells = [_verdict(summary[o][bn]["median"], base_m, metric)
                     for o in others]
            print(f"{bn:14s}  " + "  ".join(f"{c:>26s}" for c in cells))

    # ---- Optional: single-seed reference anchors ----
    if baselines:
        print()
        print("-- baselines (single-seed anchors -- DIRECTIONAL ONLY) --")
        for label, row in baselines.items():
            parts = [f"{bn}={fmt_score(row.get(bn), _metric_for(bn))}"
                     for bn in benchmarks if bn in row]
            print(f"  {label}: " + "  ".join(parts))

    return summary
