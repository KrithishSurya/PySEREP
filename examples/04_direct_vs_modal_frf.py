#!/usr/bin/env python3
"""
Example 04 — Direct FRF vs Modal Superposition FRF
====================================================
Demonstrates the difference between the direct (impedance) FRF method
and the modal superposition method for the ROM.

Both should produce very accurate results vs the full-model reference,
but the direct method is exact within the retained subspace.

Run:
    python examples/04_direct_vs_modal_frf.py
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from pyserep import (
    solve_eigenproblem, select_modes, select_dofs_eid,
    build_serep_rom, compute_frf_direct, compute_frf_modal,
    FrequencyBandSet, FrequencyBand,
)

# ── Build model ───────────────────────────────────────────────────────────────
N = 200
k = 3e4
K = sp.diags([[-k]*(N-1), [2*k]*N, [-k]*(N-1)], [-1,0,1], format="csc").astype(float)
K[0,0] = k
M = sp.eye(N, format="csc")

freqs, phi = solve_eigenproblem(K, M, n_modes=50, verbose=False)
band_set   = FrequencyBandSet([FrequencyBand(1.0, 80.0)], n_points_per_band=1000)
force_dofs = output_dofs = [N // 3]

selected = select_modes(phi, freqs, force_dofs=force_dofs,
                         output_dofs=output_dofs, band_set=band_set, verbose=False)
dofs, _   = select_dofs_eid(phi, selected, verbose=False)
T, Ka, Ma = build_serep_rom(K, M, phi, selected, dofs, verbose=False)

# Local DOF index
dof_map    = {int(d): i for i, d in enumerate(dofs)}
lf = [dof_map[d] for d in force_dofs if d in dof_map]
lo = [dof_map[d] for d in output_dofs if d in dof_map]
freq_eval  = band_set.frequency_grid()

# ── Direct FRF ────────────────────────────────────────────────────────────────
_, H_direct = compute_frf_direct(
    Ka, Ma, lf, lo, freq_eval, zeta=0.01, damping_type="modal", verbose=False
)

# ── Modal FRF (ROM — selected modes only) ─────────────────────────────────────
from pyserep.frf.modal_frf import compute_frf_modal as _modal
_, H_modal_rom = _modal(phi, freqs, selected, force_dofs, output_dofs, band_set,
                         zeta=0.01, verbose=False)

# ── Reference (all elastic modes) ─────────────────────────────────────────────
elastic = np.where(freqs > 1.0)[0]
_, H_ref = _modal(phi, freqs, elastic, force_dofs, output_dofs, band_set,
                   zeta=0.01, verbose=False)

# ── Error computation ─────────────────────────────────────────────────────────
def frf_error(H, key, H_ref, ref_key):
    h_r = np.abs(H[key])
    h_f = np.abs(H_ref[ref_key])
    d   = np.where(h_f > 1e-30, h_f, 1e-30)
    return np.abs(h_r - h_f) / d * 100.0

dk   = f"f{lf[0]}_o{lo[0]}"
mk   = f"f{force_dofs[0]}_o{output_dofs[0]}"
rk   = mk

e_direct = frf_error(H_direct, dk, H_ref, rk)
e_modal  = frf_error(H_modal_rom, mk, H_ref, rk)

print(f"Direct FRF  : max = {e_direct.max():.6f}%  rms = {np.sqrt(np.mean(e_direct**2)):.6f}%")
print(f"Modal FRF   : max = {e_modal.max():.6f}%  rms = {np.sqrt(np.mean(e_modal**2)):.6f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.semilogy(freq_eval, np.abs(H_ref[rk]),       "k-",  lw=1.5, alpha=0.6, label="Reference")
ax1.semilogy(freq_eval, np.abs(H_direct[dk]),    "b--", lw=1.8, label="ROM — Direct")
ax1.semilogy(freq_eval, np.abs(H_modal_rom[mk]), "r:",  lw=1.5, label="ROM — Modal")
ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("|H| (m/N)")
ax1.set_title("FRF Comparison: Direct vs Modal Methods")
ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

ax2.semilogy(freq_eval, np.maximum(e_direct, 1e-10), "b--", lw=1.5, label="Direct error")
ax2.semilogy(freq_eval, np.maximum(e_modal,  1e-10), "r:",  lw=1.5, label="Modal error")
ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Relative error (%)")
ax2.set_title("FRF Error vs Reference")
ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("direct_vs_modal_frf.png", dpi=150)
print("\nPlot saved: direct_vs_modal_frf.png")
