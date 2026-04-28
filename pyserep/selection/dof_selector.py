"""
pyserep.selection.dof_selector
=================================
Four DOF selection strategies for SEREP master-DOF identification.

+---------+-----------------------------------+----------------+----------------+
| Method  | Name                              | Condition κ    | Speed          |
+=========+===================================+================+================+
| DS1     | Kinetic Energy                    | ~10¹⁰–10¹⁵   | Very fast O(N) |
+---------+-----------------------------------+----------------+----------------+
| DS2     | Peak Modal Displacement           | ~10⁸–10¹²    | Fast O(N·m)    |
+---------+-----------------------------------+----------------+----------------+
| DS3     | SVD-based (QR with column pivot)  | ~10³–10⁶     | Fast O(N·m²)   |
+---------+-----------------------------------+----------------+----------------+
| DS4     | Effective Independence (Kammer)   | ~10¹–10²     | O(N·m² per it.)|
+---------+-----------------------------------+----------------+----------------+

Recommendation
--------------
**DS4** (Effective Independence) is strongly recommended for SEREP.  It is
the only method that consistently produces κ(Φₐ) < 100, which is the
prerequisite for accurate eigenvalue preservation in the ROM.

Reference: Kammer, D.C. (1991). "Sensor placement for on-orbit modal
identification and correlation of large space structures." Journal of
Guidance, Control, and Dynamics, 14(2), 251–259.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import scipy.linalg as la

# ─────────────────────────────────────────────────────────────────────────────
# DS4 — Effective Independence (recommended)
# ─────────────────────────────────────────────────────────────────────────────

def select_dofs_eid(
    phi: np.ndarray,
    selected_modes: np.ndarray,
    n_master: Optional[int] = None,
    candidate_dofs: Optional[np.ndarray] = None,
    ke_prescreen_frac: float = 0.5,
    required_dofs: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    DS4: Select master DOFs using the Effective Independence (EID) method.

    The algorithm iteratively removes the DOF with the smallest diagonal
    entry of the EI matrix:

        **E = Φₛ (ΦₛᵀΦₛ)⁻¹ Φₛᵀ**

    which corresponds to the DOF contributing the *least* to the linear
    independence of the mode shape partition.  The Fisher Information
    Matrix det(ΦₛᵀΦₛ) is thereby maximised.

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_all_modes)
        Full modal matrix (mass-normalised).
    selected_modes : np.ndarray of int
        Mode indices from the mode selection pipeline.
    n_master : int, optional
        Number of master DOFs.  Default: ``len(selected_modes)``
        (enforces the exact-SEREP rule a = m).
    candidate_dofs : np.ndarray of int, optional
        Pre-screened candidate DOF set.  If None, a kinetic-energy
        pre-screen retains the top *ke_prescreen_frac* fraction.
    ke_prescreen_frac : float
        Fraction to retain in the KE pre-screen (0 < f ≤ 1).
    verbose : bool

    Returns
    -------
    master_dofs : np.ndarray of int
        Selected master DOF indices.
    kappa : float
        Condition number κ(Φₐ) of the final master partition.

    Examples
    --------
    >>> dofs, kappa = select_dofs_eid(phi, selected_modes)
    >>> kappa < 100
    True
    """
    m = len(selected_modes)
    n_master = n_master if n_master is not None else m
    phi_sel = phi[:, selected_modes]     # (N, m)

    if verbose:
        print(
            f"\n[DS4 — Effective Independence]\n"
            f"  Target: a = {n_master} master DOFs  (m = {m} modes)\n"
            f"  Full model: N = {phi.shape[0]:,} DOFs"
        )

    # ── KE pre-screen ─────────────────────────────────────────────────────────
    if candidate_dofs is None:
        candidate_dofs = _ke_prescreen(phi_sel, ke_prescreen_frac, verbose)

    # ── Ensure required DOFs are always in the candidate set ─────────────────
    if required_dofs is not None:
        required = np.asarray(required_dofs, dtype=int)
        candidate_dofs = np.unique(np.concatenate([candidate_dofs, required]))
        # Expand n_master if required_dofs would exceed it
        n_master = max(n_master, len(required))

    if len(candidate_dofs) < n_master:
        raise ValueError(
            f"Candidate set ({len(candidate_dofs)}) < n_master ({n_master}).  "
            "Increase ke_prescreen_frac."
        )

    # ── Iterative EI deletion ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    current = np.array(candidate_dofs, dtype=int)
    n_del   = len(current) - n_master

    if verbose:
        print(f"  Starting EI deletion: {len(current):,} → {n_master} DOFs …")

    for step in range(n_del):
        phi_c = phi_sel[current, :]          # (s, m)
        A = phi_c.T @ phi_c                  # FIM: (m, m)

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)

        # E_d = diag(Φₛ A⁻¹ Φₛᵀ) — row-wise
        E_d = np.einsum("ij,ij->i", phi_c @ A_inv, phi_c)  # (s,)
        # Protect required DOFs — never remove them
        if required_dofs is not None:
            req_set = set(required_dofs.tolist())
            removable = np.array([current[i] not in req_set
                                   for i in range(len(E_d))])
            if removable.any():
                masked = np.where(removable, E_d, np.inf)
                remove_idx = int(np.argmin(masked))
            else:
                break   # all remaining candidates are required — stop early
        else:
            remove_idx = int(np.argmin(E_d))
        current = np.delete(current, remove_idx)

        if verbose and step % max(1, n_del // 20) == 0:
            pct = (step + 1) / n_del * 100
            print(f"  [{pct:5.1f}%] {len(current):,} DOFs remaining", end="\r")

    if verbose:
        print(" " * 60, end="\r")

    phi_a = phi_sel[current, :]
    kappa = float(np.linalg.cond(phi_a))
    rank  = int(np.linalg.matrix_rank(phi_a))
    elapsed = time.perf_counter() - t0

    if verbose:
        label = ("EXCELLENT" if kappa < 1e2 else
                 "GOOD"      if kappa < 1e3 else
                 "MARGINAL"  if kappa < 1e6 else "POOR")
        print(
            f"  Completed in {elapsed:.2f}s\n"
            f"  Master DOFs : {len(current)}\n"
            f"  κ(Φₐ)       : {kappa:.4e}  [{label}]\n"
            f"  rank(Φₐ)    : {rank}/{m}"
        )

    return current, kappa


# ─────────────────────────────────────────────────────────────────────────────
# DS1 — Kinetic Energy
# ─────────────────────────────────────────────────────────────────────────────

def select_dofs_kinetic(
    phi: np.ndarray,
    selected_modes: np.ndarray,
    n_master: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    DS1: Select master DOFs by total kinetic energy across selected modes.

    Score for DOF j:  KE_j = Σᵢ |φᵢ(j)|²

    Selects the top-*n_master* DOFs by descending KE score.
    Fast (O(N·m)) but typically produces κ ~ 10¹⁰–10¹⁵, which is
    unsuitable for SEREP unless used as a pre-screen for DS4.

    Parameters
    ----------
    phi, selected_modes : see :func:`select_dofs_eid`
    n_master : int, optional

    Returns
    -------
    master_dofs, kappa
    """
    m = len(selected_modes)
    n_master = n_master if n_master is not None else m
    phi_sel = phi[:, selected_modes]

    ke = np.sum(phi_sel ** 2, axis=1)      # (N,)
    top = np.argsort(ke)[::-1][:n_master]
    master_dofs = np.sort(top)

    kappa = float(np.linalg.cond(phi_sel[master_dofs, :]))
    if verbose:
        print(f"[DS1 — Kinetic Energy]  κ(Φₐ) = {kappa:.4e}")
    return master_dofs, kappa


# ─────────────────────────────────────────────────────────────────────────────
# DS2 — Peak Modal Displacement
# ─────────────────────────────────────────────────────────────────────────────

def select_dofs_modal_disp(
    phi: np.ndarray,
    selected_modes: np.ndarray,
    n_master: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    DS2: Select master DOFs by maximum modal displacement magnitude.

    Score for DOF j:  S_j = max_i |φᵢ(j)|

    Retains DOFs where at least one mode has large amplitude.

    Parameters
    ----------
    phi, selected_modes : see :func:`select_dofs_eid`
    n_master : int, optional

    Returns
    -------
    master_dofs, kappa
    """
    m = len(selected_modes)
    n_master = n_master if n_master is not None else m
    phi_sel = phi[:, selected_modes]

    score = np.max(np.abs(phi_sel), axis=1)   # (N,)
    top = np.argsort(score)[::-1][:n_master]
    master_dofs = np.sort(top)

    kappa = float(np.linalg.cond(phi_sel[master_dofs, :]))
    if verbose:
        print(f"[DS2 — Modal Displacement]  κ(Φₐ) = {kappa:.4e}")
    return master_dofs, kappa


# ─────────────────────────────────────────────────────────────────────────────
# DS3 — SVD / QR with column pivoting
# ─────────────────────────────────────────────────────────────────────────────

def select_dofs_svd(
    phi: np.ndarray,
    selected_modes: np.ndarray,
    n_master: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    DS3: Select master DOFs via QR factorisation with column pivoting of Φᵀ.

    The column permutation from ``scipy.linalg.qr(Φᵀ, pivoting=True)``
    directly gives the DOF ordering from most to least informative.

    This is equivalent to maximising the diagonal entries of R and
    gives κ values typically in the range 10³–10⁶ — better than DS1/DS2
    but still inferior to DS4 for SEREP.

    Parameters
    ----------
    phi, selected_modes : see :func:`select_dofs_eid`
    n_master : int, optional

    Returns
    -------
    master_dofs, kappa
    """
    m = len(selected_modes)
    n_master = n_master if n_master is not None else m
    phi_sel = phi[:, selected_modes]

    # QR with column pivoting on Φᵀ  →  P selects DOFs
    _, _, perm = la.qr(phi_sel.T, pivoting=True)  # perm[0] = most informative
    master_dofs = np.sort(perm[:n_master])

    kappa = float(np.linalg.cond(phi_sel[master_dofs, :]))
    if verbose:
        print(f"[DS3 — SVD/QR pivot]  κ(Φₐ) = {kappa:.4e}")
    return master_dofs, kappa


# ─────────────────────────────────────────────────────────────────────────────
# Comparison utility
# ─────────────────────────────────────────────────────────────────────────────

def compare_dof_selectors(
    phi: np.ndarray,
    selected_modes: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Run all four DOF selectors and return a comparison table.

    Parameters
    ----------
    phi : np.ndarray
    selected_modes : np.ndarray of int
    verbose : bool

    Returns
    -------
    dict
        Keys: "DS1", "DS2", "DS3", "DS4".
        Values: dict with "master_dofs", "kappa", "rank", "elapsed_s".
    """
    results = {}

    selectors = [
        ("DS1", select_dofs_kinetic),
        ("DS2", select_dofs_modal_disp),
        ("DS3", select_dofs_svd),
        ("DS4", select_dofs_eid),
    ]
    phi_sel = phi[:, selected_modes]
    m = len(selected_modes)

    if verbose:
        print("\n" + "─" * 55)
        print("  DOF SELECTOR COMPARISON")
        print("─" * 55)
        print(f"  {'Method':<8}  {'κ(Φₐ)':<14}  {'rank':>6}  {'Time':>8}")
        print("─" * 55)

    for name, fn in selectors:
        t0 = time.perf_counter()
        dofs, kappa = fn(phi, selected_modes, verbose=False)
        elapsed = time.perf_counter() - t0
        rank = int(np.linalg.matrix_rank(phi_sel[dofs, :]))
        results[name] = {
            "master_dofs": dofs,
            "kappa": kappa,
            "rank": rank,
            "elapsed_s": elapsed,
        }
        if verbose:
            print(f"  {name:<8}  {kappa:<14.4e}  {rank:>6}/{m}  {elapsed:>7.2f}s")

    if verbose:
        print("─" * 55)
        best = min(results, key=lambda k: results[k]["kappa"])
        print(f"  Best κ: {best}  ({results[best]['kappa']:.4e})")
        print("─" * 55 + "\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _ke_prescreen(phi_sel: np.ndarray, fraction: float, verbose: bool) -> np.ndarray:
    """Keep top fraction of DOFs ranked by max modal kinetic energy."""
    N = phi_sel.shape[0]
    ke = np.max(phi_sel ** 2, axis=1)
    n_keep = max(1, int(N * fraction))
    top = np.sort(np.argsort(ke)[::-1][:n_keep])
    if verbose:
        print(f"  KE pre-screen: {N:,} → {n_keep:,} candidate DOFs  "
              f"(top {fraction*100:.0f}%)")
    return top
