#!/usr/bin/env python3
"""
benchmarks/benchmark_pipeline.py
==================================
End-to-end performance benchmarks for the SEREP ROM pipeline.

Runs the complete pipeline on three synthetic model sizes and reports:
  - Wall-clock time per stage
  - FLOP counts and speedup
  - Memory usage
  - FRF accuracy vs reference

Run:
    python benchmarks/benchmark_pipeline.py
    python benchmarks/benchmark_pipeline.py --sizes 500 2000 5000
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from typing import List

import numpy as np

from pyserep.models.synthetic import spring_chain, euler_beam
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.selection.mode_selector import select_modes_pipeline
from pyserep.selection.dof_selector import select_dofs_eid
from pyserep.core.rom_builder import build_serep_rom, verify_eigenvalues
from pyserep.frf.direct_frf import compute_frf_direct
from pyserep.frf.modal_frf import compute_frf_modal_reference
from pyserep.analysis.performance import flop_count


# ─────────────────────────────────────────────────────────────────────────────
# Single model benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(N: int, f_max: float = 100.0, n_freq: int = 1000) -> dict:
    """
    Benchmark the SEREP pipeline on a spring chain of size N.

    Returns a dict of timing, FLOP, and accuracy metrics.
    """
    print(f"\n{'='*60}")
    print(f"  BENCHMARK  N = {N:,} DOFs  |  f_max = {f_max} Hz")
    print(f"{'='*60}")

    results = {"N": N}
    tracemalloc.start()

    # ── Model ────────────────────────────────────────────────────────────────
    K, M = spring_chain(N, k=2e5)
    force_dofs  = [N // 2]
    output_dofs = [N // 2]

    # ── Eigensolver ───────────────────────────────────────────────────────────
    n_modes = min(120, N // 2)
    t0 = time.perf_counter()
    freqs, phi = solve_eigenproblem(K, M, n_modes=n_modes, verbose=False)
    results["t_eigensolver"] = time.perf_counter() - t0
    print(f"  Eigensolver   : {results['t_eigensolver']:.3f}s  ({n_modes} modes)")

    # ── Mode selection ────────────────────────────────────────────────────────
    band_set = FrequencyBandSet(
        [FrequencyBand(1.0, f_max)], n_points_per_band=n_freq
    )
    t0 = time.perf_counter()
    selected = select_modes_pipeline(
        phi, freqs, force_dofs, output_dofs, band_set, verbose=False
    )
    results["t_mode_select"] = time.perf_counter() - t0
    results["n_modes"] = len(selected)
    print(f"  Mode select   : {results['t_mode_select']:.3f}s  ({len(selected)} modes)")

    if len(selected) == 0:
        print("  No modes selected — skip")
        return results

    # ── DOF selection ─────────────────────────────────────────────────────────
    required = np.array(force_dofs + output_dofs, dtype=int)
    t0 = time.perf_counter()
    dofs, kappa = select_dofs_eid(phi, selected, required_dofs=required, verbose=False)
    results["t_dof_select"] = time.perf_counter() - t0
    results["kappa"] = kappa
    print(f"  DOF select    : {results['t_dof_select']:.3f}s  κ = {kappa:.4e}")

    # ── ROM build ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    T, Ka, Ma = build_serep_rom(K, M, phi, selected, dofs, verbose=False)
    results["t_rom_build"] = time.perf_counter() - t0
    _, max_err = verify_eigenvalues(Ka, Ma, freqs, selected, verbose=False)
    results["max_freq_err"] = max_err
    print(f"  ROM build     : {results['t_rom_build']:.3f}s  "
          f"eigenval err = {max_err:.8f}%")

    # ── Direct FRF ───────────────────────────────────────────────────────────
    freq_eval = band_set.frequency_grid()
    dof_map   = {int(d): i for i, d in enumerate(dofs)}
    lf = [dof_map[d] for d in force_dofs if d in dof_map]
    lo = [dof_map[d] for d in output_dofs if d in dof_map]

    if not lf:
        print("  force DOF not in master — should not happen with required_dofs")
        return results
    t0 = time.perf_counter()
    _, H_rom = compute_frf_direct(Ka, Ma, lf, lo, freq_eval, zeta=0.01, verbose=False)
    results["t_frf_rom"] = time.perf_counter() - t0

    # ── Reference FRF ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    H_ref = compute_frf_modal_reference(
        phi, freqs, 1.0, force_dofs, output_dofs, band_set, zeta=0.01, verbose=False
    )
    results["t_frf_ref"] = time.perf_counter() - t0

    # ── FRF error ─────────────────────────────────────────────────────────────
    key_r = f"f{lf[0]}_o{lo[0]}"
    key_f = f"f{force_dofs[0]}_o{output_dofs[0]}"
    h_r   = np.abs(H_rom.get(key_r, np.zeros(len(freq_eval))))
    h_f   = np.abs(H_ref.get(key_f, np.ones(len(freq_eval))))
    denom = np.where(h_f > 1e-30, h_f, 1e-30)
    err   = np.abs(h_r - h_f) / denom * 100.0
    results["frf_max_err"] = float(err.max())
    results["frf_rms_err"] = float(np.sqrt(np.mean(err**2)))

    # ── FLOPs ─────────────────────────────────────────────────────────────────
    n_pts = len(freq_eval)
    m     = len(selected)
    n_el  = len(freqs)
    # FLOP counts
    results["flops_rom"] = flop_count(m,    n_pts, 1, "direct")
    results["flops_ref"] = flop_count(n_el, n_pts, 1, "modal")
    # Real speedup = wall-clock time ratio; FLOP ratio shown separately
    t_frf_ref = results.get("t_frf_ref", 1e-9)
    t_frf_rom = results.get("t_frf_rom", 1e-9)
    results["wallclock_speedup"] = t_frf_ref / max(t_frf_rom, 1e-9)
    results["flop_ratio_ref_over_rom"] = results["flops_ref"] / max(results["flops_rom"], 1)

    # ── Memory ────────────────────────────────────────────────────────────────
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["peak_mem_mb"] = peak_mem / 1e6

    print(
        f"  FRF (ROM)     : {results['t_frf_rom']:.3f}s  "
        f"max={results['frf_max_err']:.4f}%\n"
        f"  FRF (ref)     : {results['t_frf_ref']:.3f}s\n"
        f"  Wallclock spd : {results['wallclock_speedup']:.2f}x\n"
        f"  DOF retention : {len(dofs)/N*100:.4f}%\n"
        f"  Peak memory   : {results['peak_mem_mb']:.1f} MB"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-size sweep and summary table
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(sizes: List[int], f_max: float = 100.0, n_freq: int = 1000) -> None:
    all_results = []
    for N in sizes:
        try:
            r = benchmark_one(N, f_max=f_max, n_freq=n_freq)
            all_results.append(r)
        except Exception as exc:
            print(f"  ERROR for N={N}: {exc}")
            continue

    # Summary table
    print(f"\n\n{'='*85}")
    print(f"  BENCHMARK SUMMARY  (f_max={f_max} Hz, {n_freq} freq pts)")
    print(f"{'='*85}")
    print(
        f"  {'N':>8}  {'Modes':>6}  {'κ':>12}  {'EigErr%':>10}  "
        f"{'FRFmax%':>10}  {'Speedup':>8}  {'Mem MB':>8}"
    )
    print("─" * 85)
    for r in all_results:
        if "frf_max_err" not in r:
            continue
        print(
            f"  {r['N']:>8,}  "
            f"{r.get('n_modes', 0):>6}  "
            f"{r.get('kappa', 0):>12.4e}  "
            f"{r.get('max_freq_err', 0):>10.8f}  "
            f"{r.get('frf_max_err', 0):>10.4f}  "
            f"{r.get('wallclock_speedup', 0):>8.2f}×  "
            f"{r.get('peak_mem_mb', 0):>8.1f}"
        )
    print("=" * 85)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEREP ROM Pipeline Benchmark")
    parser.add_argument("--sizes",  nargs="+", type=int,
                        default=[300, 1000, 3000],
                        help="Model sizes (N DOFs) to benchmark")
    parser.add_argument("--f-max",  type=float, default=100.0)
    parser.add_argument("--n-freq", type=int,   default=1000)
    args = parser.parse_args()

    run_benchmark(args.sizes, f_max=args.f_max, n_freq=args.n_freq)
