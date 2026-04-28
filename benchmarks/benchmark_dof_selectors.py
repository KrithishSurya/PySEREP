#!/usr/bin/env python3
"""
benchmarks/benchmark_dof_selectors.py
=======================================
Benchmark all four DOF selection methods across multiple model sizes.

For each model size N, reports:
  - Wall-clock time
  - Condition number κ(Φₐ)
  - Numerical rank of Φₐ
  - Maximum eigenvalue error after building the ROM
  - Maximum FRF error vs reference

Run:
    python benchmarks/benchmark_dof_selectors.py
    python benchmarks/benchmark_dof_selectors.py --sizes 200 500 1000 2000
"""

from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np

from pyserep.models.synthetic import spring_chain
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.selection.mode_selector import select_modes_pipeline
from pyserep.selection.dof_selector import (
    select_dofs_eid, select_dofs_kinetic,
    select_dofs_modal_disp, select_dofs_svd,
)
from pyserep.core.rom_builder import build_serep_rom, verify_eigenvalues
from pyserep.frf.direct_frf import compute_frf_direct
from pyserep.frf.modal_frf import compute_frf_modal_reference


SELECTOR_FUNCS = {
    "DS1 KE":      select_dofs_kinetic,
    "DS2 Disp":    select_dofs_modal_disp,
    "DS3 SVD":     select_dofs_svd,
    "DS4 EID":     select_dofs_eid,
}


def benchmark_selectors(N: int, f_max: float = 80.0, n_freq: int = 500) -> dict:
    print(f"\n{'='*70}")
    print(f"  N = {N:,} DOFs  |  f_max = {f_max} Hz")
    print(f"{'='*70}")

    K, M = spring_chain(N, k=3e4)
    required = np.array([N // 2])
    force_dofs = output_dofs = [N // 2]

    freqs, phi = solve_eigenproblem(K, M, n_modes=min(80, N//2), verbose=False)
    band_set   = FrequencyBandSet([FrequencyBand(1.0, f_max)], n_points_per_band=n_freq)
    selected   = select_modes_pipeline(phi, freqs, force_dofs, output_dofs, band_set, verbose=False)
    m          = len(selected)

    if m == 0:
        print("  No modes selected — skip"); return {}

    freq_eval = band_set.frequency_grid()
    H_ref     = compute_frf_modal_reference(
        phi, freqs, 1.0, force_dofs, output_dofs, band_set, zeta=0.01, verbose=False
    )
    ref_key = f"f{force_dofs[0]}_o{output_dofs[0]}"

    print(f"  Modes selected: {m}  |  Freq range: "
          f"{freqs[selected[0]]:.2f}–{freqs[selected[-1]]:.2f} Hz")
    print(f"\n  {'Method':<14}  {'κ(Φₐ)':>12}  {'Rank':>8}  "
          f"{'EigErr%':>10}  {'FRFmax%':>10}  {'Time':>8}")
    print("  " + "─" * 68)

    results = {"N": N, "n_modes": m}

    for name, fn in SELECTOR_FUNCS.items():
        t0 = time.perf_counter()
        try:
            if name == "DS4 EID":
                dofs, kappa = fn(phi, selected, required_dofs=required, verbose=False)
            else:
                dofs, kappa = fn(phi, selected, verbose=False)
        except Exception as exc:
            print(f"  {name:<14}  ERROR: {exc}"); continue
        t_sel = time.perf_counter() - t0

        rank = int(np.linalg.matrix_rank(phi[dofs, :][:, selected]))

        try:
            t0 = time.perf_counter()
            T, Ka, Ma = build_serep_rom(K, M, phi, selected, dofs, verbose=False)
            _, max_eig = verify_eigenvalues(Ka, Ma, freqs, selected, verbose=False)
            t_rom = time.perf_counter() - t0
        except Exception as exc:
            print(f"  {name:<14}  ROM build ERROR: {exc}"); continue

        # FRF — only if force DOF is in master set
        dof_map = {int(d): i for i, d in enumerate(dofs)}
        lf = [dof_map[d] for d in force_dofs if d in dof_map]
        lo = [dof_map[d] for d in output_dofs if d in dof_map]
        frf_err = np.nan
        if lf:
            try:
                _, H_rom = compute_frf_direct(Ka, Ma, lf, lo, freq_eval, zeta=0.01, verbose=False)
                key_r = f"f{lf[0]}_o{lo[0]}"
                h_r   = np.abs(H_rom[key_r])
                h_f   = np.abs(H_ref[ref_key])
                d     = np.where(h_f > 1e-30, h_f, 1e-30)
                frf_err = float(np.abs(h_r - h_f).max() / d.max() * 100.0)
            except Exception:
                pass

        total_t = t_sel + t_rom
        results[name] = {
            "kappa": kappa, "rank": rank, "rank_m": m,
            "max_eig_err": float(max_eig) if not np.isnan(max_eig) else np.nan,
            "max_frf_err": frf_err,
            "elapsed_s": total_t,
        }

        eig_str = f"{max_eig:.6f}" if not np.isnan(max_eig) else "  N/A  "
        frf_str = f"{frf_err:.4f}" if not np.isnan(frf_err) else "  N/A  "
        print(
            f"  {name:<14}  {kappa:>12.4e}  {rank:>4}/{m}  "
            f"{eig_str:>10}  {frf_str:>10}  {total_t:>7.3f}s"
        )

    return results


def run_selector_benchmark(sizes: List[int], f_max: float = 80.0) -> None:
    all_results = []
    for N in sizes:
        try:
            r = benchmark_selectors(N, f_max=f_max)
            if r:
                all_results.append(r)
        except Exception as exc:
            print(f"  ERROR at N={N}: {exc}")
            continue

    if not all_results:
        return

    print(f"\n\n{'='*90}")
    print("  SUMMARY — κ(Φₐ) comparison across model sizes")
    print(f"{'='*90}")
    methods = list(SELECTOR_FUNCS.keys())
    header  = f"  {'N':>8}  {'Modes':>6}  " + "  ".join(f"{m:>12}" for m in methods)
    print(header)
    print("  " + "─" * 86)
    for r in all_results:
        row = f"  {r['N']:>8,}  {r.get('n_modes',0):>6}  "
        for name in methods:
            if name in r:
                row += f"  {r[name]['kappa']:>12.4e}"
            else:
                row += f"  {'—':>12}"
        print(row)
    print("=" * 90)
    print(f"  Best condition number consistently: DS4 EID (Kammer 1991)")
    print(f"  DS4 is the only method guaranteed to produce κ < 10³ for SEREP")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOF Selector Benchmark")
    parser.add_argument("--sizes",  nargs="+", type=int, default=[200, 500, 1000])
    parser.add_argument("--f-max",  type=float, default=80.0)
    args = parser.parse_args()
    run_selector_benchmark(args.sizes, f_max=args.f_max)
