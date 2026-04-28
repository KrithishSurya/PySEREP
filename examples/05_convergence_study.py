#!/usr/bin/env python3
"""
Example 05 — Mode Count Convergence Study
==========================================
Demonstrates how FRF accuracy and condition number improve as the
retained frequency band is widened (more modes retained).

Run:
    python examples/05_convergence_study.py
"""

import numpy as np

from pyserep.models.synthetic import spring_chain
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.analysis.convergence import mode_count_study

# ── Build model ───────────────────────────────────────────────────────────────
N = 400
K, M = spring_chain(N, k=2e5)
freqs, phi = solve_eigenproblem(K, M, n_modes=80, verbose=False)
print(f"Model: N={N} DOFs  |  {len(freqs)} modes  |  "
      f"range: {freqs[1]:.2f}–{freqs[-1]:.2f} Hz\n")

# ── Convergence sweep ─────────────────────────────────────────────────────────
f_max_ref    = 120.0
f_max_sweep  = [20, 35, 50, 65, 80, 95, 110, f_max_ref]
force_dofs   = [N // 2]
output_dofs  = [N // 2]

study = mode_count_study(
    K, M, phi, freqs,
    force_dofs   = force_dofs,
    output_dofs  = output_dofs,
    f_max        = f_max_ref,
    f_max_values = f_max_sweep,
    zeta         = 0.01,
    n_freq       = 500,
    verbose      = True,
)

# ── Results ───────────────────────────────────────────────────────────────────
print(study.table())

# ── Plot ──────────────────────────────────────────────────────────────────────
study.plot(save_path="convergence_study.png", show=False)
print("\nPlot saved: convergence_study.png")
