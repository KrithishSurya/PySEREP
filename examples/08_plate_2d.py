#!/usr/bin/env python3
"""
Example 08 — 2D Kirchhoff Plate SEREP ROM
==========================================
Demonstrates running pyserep on the built-in 2D thin plate model.

Features shown:
  1. Built-in 2D Kirchhoff plate model (simply supported, FD discretisation)
  2. Mode shape visualisation for the plate
  3. DOF selector comparison (DS1–DS4) on a 2D geometry
  4. FRF comparison across two spatial locations

Run:
    python examples/08_plate_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pyserep import (
    plate_2d, model_info, solve_eigenproblem,
    select_modes, select_dofs_eid, compare_dof_selectors,
    build_serep_rom, verify_eigenvalues,
    compute_frf_direct, compute_frf_modal,
    FrequencyBand, FrequencyBandSet,
    validate_serep,
)
from pyserep.utils.sparse_ops import build_dof_map

# ── Build plate model ─────────────────────────────────────────────────────────
nx, ny = 12, 10
lx, ly = 1.0, 0.8         # plate dimensions (m)
E   = 70e9                 # aluminium (Pa)
nu  = 0.33
h   = 0.003                # thickness (m)
rho = 2700.0               # density (kg/m³)

D     = E * h**3 / (12 * (1 - nu**2))   # flexural rigidity
rho_h = rho * h

K, M = plate_2d(nx=nx, ny=ny, lx=lx, ly=ly, D=D, rho_h=rho_h)
N    = K.shape[0]

print(f"\nModel: {model_info(K, M, '2D Kirchhoff Plate')}")
print(f"  Dimensions: {lx}m × {ly}m  |  nx={nx}  ny={ny}  h={h*1e3:.0f}mm")

# Internal DOF grid: (nx-1) × (ny-1) interior points
n_ix = nx - 1   # 11 interior columns
n_iy = ny - 1   # 9 interior rows

def dof_idx(i, j):
    """(i,j) → DOF index, 0-based. i=col, j=row."""
    return i * n_iy + j

# Force at near-centre, output at two locations
fi = n_ix // 2;  fj = n_iy // 2
oi1 = n_ix // 3; oj1 = n_iy // 3
oi2 = n_ix * 2 // 3; oj2 = n_iy * 2 // 3

force_dof  = dof_idx(fi, fj)
output_dof1 = dof_idx(oi1, oj1)
output_dof2 = dof_idx(oi2, oj2)

force_dofs   = [force_dof]
output_dofs  = [output_dof1]
force_dofs2  = [force_dof]
output_dofs2 = [output_dof2]

print(f"\nForce DOF : ({fi},{fj}) → DOF {force_dof}")
print(f"Output 1  : ({oi1},{oj1}) → DOF {output_dof1}")
print(f"Output 2  : ({oi2},{oj2}) → DOF {output_dof2}")

# ── Eigenproblem ──────────────────────────────────────────────────────────────
freqs, phi = solve_eigenproblem(K, M, n_modes=40, verbose=True)
print(f"\nFirst 6 natural frequencies (Hz): {freqs[:6].round(2)}")

# ── Mode selection ────────────────────────────────────────────────────────────
f_max    = freqs[20]
band_set = FrequencyBandSet([FrequencyBand(0.1, f_max)], n_points_per_band=600)
required = np.array([force_dof, output_dof1, output_dof2])

selected = select_modes(
    phi, freqs,
    force_dofs  = force_dofs,
    output_dofs = output_dofs,
    band_set    = band_set,
)
print(f"\nSelected {len(selected)} modes  "
      f"[{freqs[selected[0]]:.2f} – {freqs[selected[-1]]:.2f} Hz]")

# ── DOF selector comparison ───────────────────────────────────────────────────
print("\n--- DOF selector comparison ---")
comparison = compare_dof_selectors(phi, selected)

# ── Build ROM with DS4 ────────────────────────────────────────────────────────
dofs, kappa = select_dofs_eid(phi, selected, required_dofs=required)
T, Ka, Ma   = build_serep_rom(K, M, phi, selected, dofs)
_, max_err  = verify_eigenvalues(Ka, Ma, freqs, selected)
print(f"\nMaster DOFs: {len(dofs)}   κ = {kappa:.3e}   Max freq err = {max_err:.8f}%")

# ── Validation ────────────────────────────────────────────────────────────────
report = validate_serep(K, M, phi, freqs, selected, dofs, T, Ka, Ma)
print(f"Validation passed: {report.passed()}")

# ── Direct FRF — two DOF pairs ────────────────────────────────────────────────
freq_eval = band_set.frequency_grid()
lf1, lo1  = build_dof_map(dofs, force_dofs,  output_dofs)
lf2, lo2  = build_dof_map(dofs, force_dofs2, output_dofs2)

_, H_rom1 = compute_frf_direct(Ka, Ma, lf1, lo1, freq_eval, zeta=0.02)
_, H_rom2 = compute_frf_direct(Ka, Ma, lf2, lo2, freq_eval, zeta=0.02)

# Reference FRFs
elastic   = np.where(freqs > 0.5)[0]
_, H_ref1 = compute_frf_modal(phi, freqs, elastic, force_dofs,  output_dofs,  band_set, zeta=0.02)
_, H_ref2 = compute_frf_modal(phi, freqs, elastic, force_dofs2, output_dofs2, band_set, zeta=0.02)

k1  = f"f{lf1[0]}_o{lo1[0]}";  r1 = f"f{force_dof}_o{output_dof1}"
k2  = f"f{lf2[0]}_o{lo2[0]}";  r2 = f"f{force_dof}_o{output_dof2}"

err1 = np.abs(np.abs(H_rom1[k1]) - np.abs(H_ref1[r1])) / np.maximum(np.abs(H_ref1[r1]), 1e-30) * 100
err2 = np.abs(np.abs(H_rom2[k2]) - np.abs(H_ref2[r2])) / np.maximum(np.abs(H_ref2[r2]), 1e-30) * 100
print(f"\nPair 1 FRF: max={err1.max():.4f}%  rms={np.sqrt(np.mean(err1**2)):.4f}%")
print(f"Pair 2 FRF: max={err2.max():.4f}%  rms={np.sqrt(np.mean(err2**2)):.4f}%")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"2D Plate SEREP ROM — {len(selected)} modes  κ={kappa:.2e}", fontweight="bold")

# Mode shape 1 (reshaped to plate grid)
ax = axes[0, 0]
mode1 = phi[:, selected[0]].reshape(n_ix, n_iy)
im = ax.imshow(mode1.T, origin="lower", aspect="auto",
               extent=[0, lx, 0, ly], cmap="RdBu_r")
plt.colorbar(im, ax=ax)
ax.set(title=f"Mode shape: {freqs[selected[0]]:.2f} Hz",
       xlabel="x (m)", ylabel="y (m)")
ax.plot(fi / n_ix * lx, fj / n_iy * ly, "k*", ms=10, label="Force DOF")
ax.plot(oi1 / n_ix * lx, oj1 / n_iy * ly, "g^", ms=8, label="Output 1")
ax.plot(oi2 / n_ix * lx, oj2 / n_iy * ly, "bs", ms=8, label="Output 2")
ax.legend(fontsize=8)

# DOF selector κ comparison
ax = axes[0, 1]
methods = list(comparison.keys())
kappas  = [comparison[m]["kappa"] for m in methods]
colours = ["#e74c3c", "#e67e22", "#f1c40f", "#27ae60"]
ax.bar(methods, kappas, color=colours, edgecolor="black", lw=0.6)
ax.set_yscale("log")
ax.set_ylabel("κ(Φₐ)"); ax.set_title("DOF selector comparison")
ax.grid(True, axis="y", alpha=0.3)
for i, (m, k) in enumerate(zip(methods, kappas)):
    ax.text(i, k * 1.4, f"{k:.1e}", ha="center", fontsize=8)

# FRF pair 1
ax = axes[1, 0]
ax.semilogy(freq_eval, np.abs(H_ref1[r1]), "k-",  lw=1.5, alpha=0.6, label="Reference")
ax.semilogy(freq_eval, np.abs(H_rom1[k1]), "b--", lw=1.8, label="ROM")
ax.set(xlabel="Frequency (Hz)", ylabel="|H| (m/N)",
       title=f"FRF pair 1  [{oi1},{oj1}] → [{fi},{fj}]")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

# FRF pair 2
ax = axes[1, 1]
ax.semilogy(freq_eval, np.abs(H_ref2[r2]), "k-",  lw=1.5, alpha=0.6, label="Reference")
ax.semilogy(freq_eval, np.abs(H_rom2[k2]), "r--", lw=1.8, label="ROM")
ax.set(xlabel="Frequency (Hz)", ylabel="|H| (m/N)",
       title=f"FRF pair 2  [{oi2},{oj2}] → [{fi},{fj}]")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("plate_2d_results.png", dpi=150)
print("\nPlot saved: plate_2d_results.png")
