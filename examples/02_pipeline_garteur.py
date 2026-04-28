#!/usr/bin/env python3
"""
Example 02 — Pipeline API: Garteur SM-AG19 Benchmark
======================================================
Runs the complete SEREP ROM pipeline on the Garteur SM-AG19 benchmark
structure using the high-level SereпPipeline interface.

Assumes you have K.mtx and M.mtx in the current directory.
Adjust paths as needed.

Run:
    python examples/02_pipeline_garteur.py
"""

import os
from pyserep import SereпPipeline, ROMConfig, FrequencyBand

# ── Configuration ─────────────────────────────────────────────────────────────
cfg = ROMConfig(
    stiffness_file    = "StiffMatrixmm.mtx",
    mass_file         = "MassMatrixmm.mtx",

    # Force and response DOFs (Ansys convention: (node-1)*3 + direction)
    force_dofs        = [3000],    # Node 1001, UX
    output_dofs       = [3000],

    # Selective bands — ignore the 100–400 Hz gap
    bands = [
        FrequencyBand(0.1, 100.0,  label="LowBand"),
        FrequencyBand(400.0, 500.0, label="HighBand"),
    ],

    # Eigensolver
    num_modes_eigsh   = 120,

    # FRF — direct method (uses Kₐ, Mₐ physical matrices)
    frf_method        = "direct",
    damping_type      = "modal",
    zeta              = 0.005,
    n_points_per_band = 2000,

    # Mode selection
    ms1_alpha         = 1.5,
    ms2_threshold     = 1.0,
    ms3_threshold     = 5.0,
    mac_threshold     = 0.90,
    rb_hz             = 1.0,

    # DOF selection — EID (DS4)
    dof_method        = "eid",
    ke_prescreen_frac = 0.5,

    # Output
    export_folder     = "serep_output_garteur",
    save_prefix       = "GARTEUR",
    save_matrices     = True,
    plot              = True,
    verbose           = True,
)

# ── Run pipeline ──────────────────────────────────────────────────────────────
results = SereпPipeline(cfg).run()

# ── Print key results ─────────────────────────────────────────────────────────
print(results.summary())
if results.performance:
    print(results.performance.summary())
if results.validation:
    print(results.validation.summary())
