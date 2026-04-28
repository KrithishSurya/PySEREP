#!/usr/bin/env python3
"""
Example 06 — HyAPA Parametric Material Study
=============================================
Demonstrates running SEREP ROM on two material configurations
in parallel — the standard aluminium-equivalent and HyAPA —
and comparing their ROM performance metrics.

This mirrors the thesis workflow for the Garteur SM-AG19 benchmark.
Adjust paths and DOF indices to match your actual Ansys-exported matrices.

Run:
    python examples/06_hyapa_material.py
"""

import os
import numpy as np

from pyserep import SereпPipeline, ROMConfig, FrequencyBand

# ── Material configurations ───────────────────────────────────────────────────
CONFIGS = {
    "Standard": dict(
        stiffness_file = "StiffMatrixmm_standard.mtx",
        mass_file      = "MassMatrixmm_standard.mtx",
        save_prefix    = "GARTEUR_STD",
        export_folder  = "results_standard",
    ),
    "HyAPA": dict(
        stiffness_file = "StiffMatrixmm_hyapa.mtx",
        mass_file      = "MassMatrixmm_hyapa.mtx",
        save_prefix    = "GARTEUR_HYAPA",
        export_folder  = "results_hyapa",
    ),
}

# ── Common pipeline settings ──────────────────────────────────────────────────
COMMON = dict(
    force_dofs        = [3000],
    output_dofs       = [3000],
    bands             = [
        FrequencyBand(0.1, 100.0, label="LowBand"),
        FrequencyBand(400.0, 500.0, label="HighBand"),
    ],
    frf_method        = "direct",
    damping_type      = "modal",
    zeta              = 0.005,
    num_modes_eigsh   = 120,
    n_points_per_band = 2000,
    ms1_alpha         = 1.5,
    ms2_threshold     = 1.0,
    ms3_threshold     = 5.0,
    mac_threshold     = 0.90,
    dof_method        = "eid",
    plot              = True,
    verbose           = True,
)

# ── Run both configurations ───────────────────────────────────────────────────
results_all = {}

for material, spec in CONFIGS.items():
    # Skip if matrix files don't exist (demo mode)
    k_path = spec["stiffness_file"]
    m_path = spec["mass_file"]
    if not (os.path.exists(k_path) and os.path.exists(m_path)):
        print(f"\n[{material}] Matrix files not found — skipping.")
        print(f"  Expected: {k_path}, {m_path}")
        continue

    print(f"\n{'#'*60}")
    print(f"  Running: {material}")
    print(f"{'#'*60}")

    cfg     = ROMConfig(**{**COMMON, **spec})
    results = SereпPipeline(cfg).run()
    results_all[material] = results

# ── Parallel comparison table ────────────────────────────────────────────────
if len(results_all) >= 2:
    print(f"\n{'='*65}")
    print("  MATERIAL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Metric':<30}  {'Standard':>14}  {'HyAPA':>14}")
    print("─" * 65)

    rows = {}
    for mat, r in results_all.items():
        rows[mat] = {
            "Full DOFs"    : r.phi.shape[0],
            "Modes selected": len(r.selected_modes),
            "Master DOFs"  : len(r.master_dofs),
            "DOF retention": f"{len(r.master_dofs)/r.phi.shape[0]*100:.4f}%",
            "κ(Φₐ)"       : f"{r.kappa:.4e}",
            "Max freq err%": f"{r.max_freq_err:.8f}",
            "FRF max err%" : f"{max(e['max_pct'] for e in r.frf.errors.values()):.6f}",
            "FRF rms err%" : f"{max(e['rms_pct'] for e in r.frf.errors.values()):.6f}",
        }

    mats = list(results_all.keys())
    for metric in list(rows[mats[0]].keys()):
        print(
            f"  {metric:<30}  "
            + "  ".join(f"{rows[m][metric]:>14}" for m in mats)
        )
    print("=" * 65)
elif len(results_all) == 1:
    mat = list(results_all.keys())[0]
    print(f"\nRan {mat} only. Provide both matrix files for comparison.")
    print(results_all[mat].summary())
else:
    print("\nNo matrix files found. To run this example:")
    print("  1. Export K.mtx and M.mtx from Ansys for each material configuration")
    print("  2. Update CONFIGS above with the correct file paths")
    print("  3. Re-run this script")
