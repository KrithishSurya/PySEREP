#!/usr/bin/env python3
"""
Example 01 — Quick Start: Functional API
=========================================
Demonstrates using the low-level functional API without the pipeline.
Builds a synthetic 500-DOF spring-mass chain and runs the full SEREP flow.

Run:
    python examples/01_quick_start.py
"""

import numpy as np
import scipy.sparse as sp

# ── Build a synthetic 500-DOF spring-mass chain ───────────────────────────────
N   = 500
k   = 2e5   # stiffness N/m
m   = 1.0   # mass kg

K = sp.diags([[-k]*(N-1), [2*k]*N, [-k]*(N-1)], [-1,0,1], format="csc").astype(float)
K[0,0] = k          # fixed left end
M = sp.eye(N, format="csc") * m

print(f"Model: N = {N} DOFs  |  K nnz = {K.nnz}")

# ── Step 1: Eigenproblem ──────────────────────────────────────────────────────
from pyserep import solve_eigenproblem

freqs, phi = solve_eigenproblem(K, M, n_modes=80)
print(f"\nComputed {len(freqs)} modes.  Range: {freqs[0]:.2f} – {freqs[-1]:.2f} Hz")

# ── Step 2: Mode selection ────────────────────────────────────────────────────
from pyserep import select_modes, FrequencyBand, FrequencyBandSet

band_set = FrequencyBandSet(
    [FrequencyBand(0.1, 60.0, label="Band1"),
     FrequencyBand(100.0, 130.0, label="Band2")],
    n_points_per_band=500,
)
force_dofs  = [N // 2]       # mid-span
output_dofs = [N // 2]

selected = select_modes(
    phi, freqs,
    force_dofs=force_dofs,
    output_dofs=output_dofs,
    band_set=band_set,
)
print(f"\nSelected {len(selected)} modes")

# ── Step 3: DOF selection (DS4 EID) ──────────────────────────────────────────
from pyserep import select_dofs_eid, compare_dof_selectors

# Optional: compare all four selectors
comparison = compare_dof_selectors(phi, selected)

master_dofs, kappa = select_dofs_eid(phi, selected)
print(f"\nSelected {len(master_dofs)} master DOFs  |  κ(Φₐ) = {kappa:.4e}")

# ── Step 4: Build SEREP ROM ───────────────────────────────────────────────────
from pyserep import build_serep_rom, verify_eigenvalues

T, Ka, Ma = build_serep_rom(K, M, phi, selected, master_dofs)
errors, max_err = verify_eigenvalues(Ka, Ma, freqs, selected)
print(f"\nMax eigenvalue error: {max_err:.8f}%")

# ── Step 5: Direct FRF computation ───────────────────────────────────────────
from pyserep import compute_frf_direct, compute_frf_modal

freq_eval = band_set.frequency_grid()

# Map global DOFs → local indices in master set
dof_map     = {int(d): i for i, d in enumerate(master_dofs)}
local_force = [dof_map[d] for d in force_dofs if d in dof_map]
local_out   = [dof_map[d] for d in output_dofs if d in dof_map]

freqs_eval, H_rom = compute_frf_direct(
    Ka, Ma,
    force_dof_indices  = local_force,
    output_dof_indices = local_out,
    freq_eval          = freq_eval,
    zeta               = 0.01,
    damping_type       = "modal",
)

# Reference: modal FRF with all elastic modes
_, H_ref = compute_frf_modal(
    phi, freqs, np.where(freqs > 1.0)[0],
    force_dofs, output_dofs, band_set, zeta=0.01,
)

# ── Step 6: Compute FRF error ────────────────────────────────────────────────
key = f"f{local_force[0]}_o{local_out[0]}"
ref_key = f"f{force_dofs[0]}_o{output_dofs[0]}"

h_r = np.abs(H_rom[key])
h_f = np.abs(H_ref[ref_key])
denom  = np.where(h_f > 1e-30, h_f, 1e-30)
errors = np.abs(h_r - h_f) / denom * 100.0

print(f"\nFRF error:  max = {errors.max():.4f}%  |  rms = {np.sqrt(np.mean(errors**2)):.4f}%")

# ── Step 7: Validation suite ─────────────────────────────────────────────────
from pyserep import validate_serep

report = validate_serep(K, M, phi, freqs, selected, master_dofs, T, Ka, Ma)
print(f"\nValidation passed: {report.passed()}")

# ── Step 8: Performance summary ──────────────────────────────────────────────
from pyserep import reduction_metrics

metrics = reduction_metrics(N, len(master_dofs), len(freqs), len(selected))
print(f"\nDOF retention: {metrics['dof_retention_pct']:.4f}%")
print("Done.")
