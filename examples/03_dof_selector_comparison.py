#!/usr/bin/env python3
"""
Example 03 — DOF Selector Comparison (DS1–DS4)
================================================
Compares all four DOF selection methods on a synthetic model and
generates a bar chart of condition numbers.

Run:
    python examples/03_dof_selector_comparison.py
"""

import numpy as np
import scipy.sparse as sp

from pyserep import solve_eigenproblem, select_modes
from pyserep.selection.dof_selector import compare_dof_selectors
from pyserep.visualization.mode_plots import plot_dof_selector_comparison

# ── Synthetic model ───────────────────────────────────────────────────────────
N, k = 400, 1e5
K = sp.diags([[-k]*(N-1), [2*k]*N, [-k]*(N-1)], [-1,0,1], format="csc").astype(float)
K[0,0] = k
M = sp.eye(N, format="csc")

freqs, phi = solve_eigenproblem(K, M, n_modes=60, verbose=False)

selected = select_modes(
    phi, freqs,
    force_dofs=[N//2], output_dofs=[N//2],
    f_max=150.0, verbose=False,
)
print(f"Selected {len(selected)} modes")

# ── Compare all four methods ──────────────────────────────────────────────────
comparison = compare_dof_selectors(phi, selected, verbose=True)

# ── Table summary ─────────────────────────────────────────────────────────────
print("\n{'─'*50}")
print(f"{'Method':<8}  {'κ(Φₐ)':<16}  {'rank':<10}  {'time':>8}")
for name, res in comparison.items():
    m = len(selected)
    print(
        f"  {name:<6}  {res['kappa']:<16.4e}  "
        f"{res['rank']:>4}/{m}  {res['elapsed_s']:>7.3f}s"
    )

best = min(comparison, key=lambda k: comparison[k]["kappa"])
print(f"\nRecommendation: {best} (lowest κ = {comparison[best]['kappa']:.4e})")

# ── Plot ──────────────────────────────────────────────────────────────────────
plot_dof_selector_comparison(comparison, save_path="dof_comparison.png", show=False)
print("\nPlot saved: dof_comparison.png")
