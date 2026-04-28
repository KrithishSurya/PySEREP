#!/usr/bin/env python3
"""
Example 07 — Euler-Bernoulli Beam + Sensitivity Analysis
==========================================================
Demonstrates:
  1. Running SEREP ROM on the built-in Euler beam model
  2. Eigenvalue sensitivity to Young's modulus perturbation
  3. Monte Carlo FRF uncertainty quantification (±3% E, ±2% ρ)

Run:
    python examples/07_euler_beam.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pyserep.models.synthetic import euler_beam, model_info
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.selection.mode_selector import select_modes_pipeline
from pyserep.selection.dof_selector import select_dofs_eid
from pyserep.core.rom_builder import build_serep_rom, verify_eigenvalues
from pyserep.frf.direct_frf import compute_frf_direct
from pyserep.frf.modal_frf import compute_frf_modal_reference
from pyserep.analysis.sensitivity import (
    eigenvalue_sensitivity, monte_carlo_frf,
)
from pyserep.utils.sparse_ops import matrix_stats, build_dof_map

# ── Build Euler-Bernoulli beam ────────────────────────────────────────────────
n_elem  = 60
EI      = 2.1e11 * (0.01**4 / 12)   # Steel I-section: E=210 GPa, b=h=10 cm
rho_A   = 7850 * 0.01 ** 2           # Steel density × cross-section area
length  = 2.0                         # 2 m cantilevered beam

K, M = euler_beam(n_elements=n_elem, length=length, EI=EI, rho_A=rho_A)
N    = K.shape[0]
print(f"\nModel: {model_info(K, M, 'Euler beam')}")
print(f"  n_elements={n_elem}  EI={EI:.3e}  ρA={rho_A:.3f}  L={length}m")

# Force at mid-span (node 31, UY direction = DOF index)
force_node  = 31
force_dof   = (force_node - 1) * 2 + 0    # transverse DOF (0 = w, 1 = θ)
force_dofs  = [force_dof]
output_dofs = [force_dof]

# ── Eigenproblem ──────────────────────────────────────────────────────────────
freqs, phi = solve_eigenproblem(K, M, n_modes=30, verbose=True)
print(f"\nFirst 5 natural frequencies (Hz): {freqs[:5].round(3)}")

# ── Mode selection ────────────────────────────────────────────────────────────
f_max    = freqs[10] * 1.2     # capture first 10 elastic modes
band_set = FrequencyBandSet([FrequencyBand(0.1, f_max)], n_points_per_band=800)
required = np.array(force_dofs + output_dofs)

selected = select_modes_pipeline(
    phi, freqs, force_dofs, output_dofs, band_set, rb_hz=0.5
)
print(f"\nSelected {len(selected)} modes  f_range=[{freqs[selected[0]]:.2f}, {freqs[selected[-1]]:.2f}] Hz")

# ── DOF selection (EID, required DOFs pinned) ─────────────────────────────────
dofs, kappa = select_dofs_eid(phi, selected, required_dofs=required)
print(f"\nMaster DOFs: {len(dofs)}  κ(Φₐ) = {kappa:.4e}")

# ── Build SEREP ROM ───────────────────────────────────────────────────────────
T, Ka, Ma = build_serep_rom(K, M, phi, selected, dofs)
_, max_err = verify_eigenvalues(Ka, Ma, freqs, selected)
print(f"\nMax eigenvalue error: {max_err:.8f}%")

# ── Local DOF mapping ─────────────────────────────────────────────────────────
local_force, local_output = build_dof_map(dofs, force_dofs, output_dofs)

freq_eval = band_set.frequency_grid()
_, H_rom  = compute_frf_direct(Ka, Ma, local_force, local_output, freq_eval, zeta=0.02)
H_ref     = compute_frf_modal_reference(phi, freqs, 0.5, force_dofs, output_dofs, band_set, zeta=0.02)

key_r = f"f{local_force[0]}_o{local_output[0]}"
key_f = f"f{force_dofs[0]}_o{output_dofs[0]}"
h_r   = np.abs(H_rom[key_r]);  h_f = np.abs(H_ref[key_f])
err   = np.abs(h_r - h_f) / np.where(h_f > 1e-30, h_f, 1e-30) * 100.0
print(f"\nFRF error: max={err.max():.6f}%  rms={np.sqrt(np.mean(err**2)):.6f}%")

# ── Eigenvalue sensitivity to E ───────────────────────────────────────────────
print("\n--- Eigenvalue sensitivity to Young's modulus (1% change) ---")
import scipy.sparse as sp
dK_dp = 0.01 * K                                # ∂K/∂E = K/E, scaled to 1% change
dM_dp = sp.csc_matrix(K.shape, dtype=float)    # ∂M/∂E = 0
dlam  = eigenvalue_sensitivity(K, M, phi, freqs, selected, dK_dp, dM_dp)
dfreq = dlam / (2.0 * np.pi * freqs[selected] + 1e-10) / (2.0 * np.pi)
print(f"  ∂f/∂(1%E): max = {np.abs(dfreq).max():.4f} Hz  "
      f"— largest mode: Mode {selected[np.argmax(np.abs(dfreq))]}")

# ── Monte Carlo UQ ────────────────────────────────────────────────────────────
print("\n--- Monte Carlo FRF UQ (50 samples, σE=3%, σρ=2%) ---")
mc = monte_carlo_frf(
    Ka, Ma, local_force, local_output, freq_eval,
    E_cov_pct=3.0, rho_cov_pct=2.0,
    n_samples=50, zeta=0.02, verbose=True,
)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"Euler Beam SEREP ROM — {len(selected)} modes  κ={kappa:.2e}", fontweight="bold")

ax = axes[0, 0]
ax.semilogy(freq_eval, h_f, "k-",  lw=1.5, alpha=0.6, label="Reference")
ax.semilogy(freq_eval, h_r, "b--", lw=1.8, label="ROM (direct)")
ax.set(xlabel="Frequency (Hz)", ylabel="|H| (m/N)", title="FRF — ROM vs reference")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
ax.semilogy(freq_eval, np.maximum(err, 1e-10), "r-", lw=1.2)
ax.axhline(0.01, color="navy", ls="--", lw=0.8, label="0.01% threshold")
ax.set(xlabel="Frequency (Hz)", ylabel="Error (%)", title="FRF relative error")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 0]
ax.bar(range(len(dfreq)), np.abs(dfreq),
       color="steelblue", edgecolor="white", lw=0.5)
ax.set(xlabel="Mode index (selected)", ylabel="|∂f/∂(1%E)| (Hz)",
       title="Eigenvalue sensitivity to Young's modulus")
ax.grid(True, axis="y", alpha=0.3)

ax = axes[1, 1]
ax.semilogy(freq_eval, mc["H_mean"], "b-", lw=1.8, label="Mean")
ax.fill_between(freq_eval, mc["H_p5"], mc["H_p95"],
                alpha=0.25, color="blue", label="5–95 percentile band")
ax.set(xlabel="Frequency (Hz)", ylabel="|H| (m/N)",
       title="Monte Carlo UQ  (σE=3%, σρ=2%, 50 samples)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("euler_beam_results.png", dpi=150)
print("\nPlot saved: euler_beam_results.png")
