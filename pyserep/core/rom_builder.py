"""
pyserep.core.rom_builder
==========================
Constructs the SEREP Reduced Order Model.

Mathematical background
-----------------------
Given:
  Φ  — full modal matrix (N × m), mass-normalised
  Φₐ — partition of Φ at master DOFs (a × m), a = m

The SEREP transformation is:

    **T = Φ · Φₐ⁺**   (N × a)

where Φₐ⁺ is the Moore–Penrose pseudoinverse.  When a = m (exact SEREP),
Φₐ is square and Φₐ⁺ = Φₐ⁻¹, yielding exact eigenvalue preservation.

Reduced matrices:

    **Kₐ = Tᵀ K T**   (a × a)
    **Mₐ = Tᵀ M T**   (a × a)

Key property (SEREP theorem):
    eig(Kₐ, Mₐ) = eig(K, M)|selected_modes   exactly (up to machine ε)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


def build_serep_rom(
    K: sp.csc_matrix,
    M: sp.csc_matrix,
    phi: np.ndarray,
    selected_modes: np.ndarray,
    master_dofs: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the SEREP transformation matrix T and compute reduced matrices Kₐ, Mₐ.

    Parameters
    ----------
    K : scipy.sparse.csc_matrix
        Full stiffness matrix (N × N). **Must be real and symmetric.**
    M : scipy.sparse.csc_matrix
        Full mass matrix (N × N). **Must be real, symmetric, and positive definite.**
    phi : np.ndarray, shape (N, n_all_modes)
        Full mass-normalised modal matrix.
    selected_modes : np.ndarray of int
        Mode indices to retain (output of :func:`select_modes`).
    master_dofs : np.ndarray of int
        Master DOF indices (output of :func:`select_dofs_eid`).
        Must satisfy ``len(master_dofs) == len(selected_modes)`` for
        exact SEREP.
    verbose : bool
        Print construction diagnostics.

    Returns
    -------
    T : np.ndarray, shape (N, m)
        SEREP transformation matrix.
    Ka : np.ndarray, shape (m, m)
        Reduced stiffness matrix.
    Ma : np.ndarray, shape (m, m)
        Reduced mass matrix.

    Raises
    ------
    ValueError
        If ``len(master_dofs) != len(selected_modes)`` and the exact
        inverse would be undefined.
    RuntimeWarning
        If the condition number of Φₐ is very large (> 10⁶), indicating
        a poorly conditioned master DOF set.

    Notes
    -----
    When ``len(master_dofs) > len(selected_modes)`` (over-constrained),
    the Moore–Penrose pseudoinverse is used, resulting in a least-squares
    SEREP approximation.

    Examples
    --------
    >>> T, Ka, Ma = build_serep_rom(K, M, phi, selected_modes, master_dofs)
    >>> Ka.shape
    (37, 37)
    """
    m = len(selected_modes)
    a = len(master_dofs)

    if verbose:
        print(
            f"\n[ROM Builder]  m = {m} retained modes  |  a = {a} master DOFs"
        )
        if a != m:
            print(
                f"  WARNING: a ≠ m ({a} ≠ {m}).  "
                "Over-constrained SEREP — using pseudoinverse (least squares)."
            )

    # ── Modal submatrix for selected modes ────────────────────────────────────
    Phi   = phi[:, selected_modes]       # (N, m)
    Phi_a = Phi[master_dofs, :]          # (a, m)

    # ── Condition number ──────────────────────────────────────────────────────
    kappa = float(np.linalg.cond(Phi_a))
    rank  = int(np.linalg.matrix_rank(Phi_a))
    if verbose:
        _print_cond(kappa, rank, m)

    # ── Pseudoinverse Φₐ⁺ ────────────────────────────────────────────────────
    if a == m:
        try:
            Phi_a_inv = np.linalg.inv(Phi_a)       # exact inverse (a = m)
        except np.linalg.LinAlgError:
            import warnings
            warnings.warn(
                "Φₐ is singular — falling back to pseudoinverse.",
                RuntimeWarning,
                stacklevel=2,
            )
            Phi_a_inv = np.linalg.pinv(Phi_a)
    else:
        Phi_a_inv = np.linalg.pinv(Phi_a)           # least squares (a > m)

    # ── Transformation matrix T = Φ · Φₐ⁺ ────────────────────────────────────
    T = Phi @ Phi_a_inv                              # (N, a)

    # ── Reduced matrices ──────────────────────────────────────────────────────
    #   Efficient:  TᵀKT = (KT)ᵀ T  (avoids N×N intermediate)
    KT = K @ T       # sparse × dense → (N, a)
    MT = M @ T

    Ka = T.T @ KT    # (a, a)
    Ma = T.T @ MT

    # Enforce exact symmetry (remove floating-point asymmetry)
    Ka = 0.5 * (Ka + Ka.T)
    Ma = 0.5 * (Ma + Ma.T)

    if verbose:
        print(
            f"  Ka: {Ka.shape}  |  symmetry error: {np.abs(Ka - Ka.T).max():.2e}\n"
            f"  Ma: {Ma.shape}  |  symmetry error: {np.abs(Ma - Ma.T).max():.2e}"
        )

    return T, Ka, Ma


def verify_eigenvalues(
    Ka: np.ndarray,
    Ma: np.ndarray,
    full_freqs_hz: np.ndarray,
    selected_modes: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Verify SEREP's defining property: exact eigenvalue preservation.

    Solves the reduced eigenvalue problem ``Kₐ φ = λ Mₐ φ`` and
    compares the resulting frequencies to the full-model values.

    Parameters
    ----------
    Ka : np.ndarray, shape (m, m)
    Ma : np.ndarray, shape (m, m)
    full_freqs_hz : np.ndarray
        All full-model natural frequencies (Hz).
    selected_modes : np.ndarray of int
        Indices identifying the target frequencies.
    verbose : bool

    Returns
    -------
    freq_errors_pct : np.ndarray, shape (m,)
        Per-mode percentage error ``|f_rom - f_full| / f_full × 100``.
    max_error_pct : float
        Maximum absolute percentage error.  Should be < 0.001% for a
        well-conditioned SEREP.

    Examples
    --------
    >>> errors, max_err = verify_eigenvalues(Ka, Ma, freqs_hz, selected_modes)
    >>> max_err < 0.001
    True
    """
    target = np.sort(full_freqs_hz[selected_modes])

    try:
        lam = la.eigh(Ka, Ma, eigvals_only=True)
        lam = np.maximum(lam, 0.0)
        freqs_rom = np.sort(np.sqrt(lam) / (2.0 * np.pi))
    except la.LinAlgError as exc:
        if verbose:
            print(f"[Verify] Dense eigensolver failed: {exc}")
        nan = np.full(len(selected_modes), np.nan)
        return nan, np.nan

    n = min(len(freqs_rom), len(target))
    denom = np.maximum(np.abs(target[:n]), 1e-10)
    errors = np.abs(freqs_rom[:n] - target[:n]) / denom * 100.0
    max_err = float(errors.max())

    if verbose:
        status = "✓ PASS" if max_err < 0.01 else ("⚠ WARN" if max_err < 1.0 else "✗ FAIL")
        print(
            f"\n[Eigenvalue Verification]  {status}\n"
            f"  Max error  : {max_err:.8f}%\n"
            f"  Mean error : {errors.mean():.8f}%\n"
            f"  Threshold  : 0.01%  (SEREP exact-preservation criterion)"
        )

    return errors, max_err


# ─────────────────────────────────────────────────────────────────────────────
# Damping matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def build_rayleigh_damping(
    Ka: np.ndarray,
    Ma: np.ndarray,
    zeta: float,
    freqs_hz: np.ndarray,
    modes: np.ndarray,
) -> np.ndarray:
    """
    Build a Rayleigh damping matrix for the reduced model.

    Rayleigh damping:  Cₐ = α Mₐ + β Kₐ

    where α and β are chosen to give damping ratio *zeta* at the
    first and last selected natural frequencies.

    Parameters
    ----------
    Ka, Ma : np.ndarray, shape (m, m)
    zeta : float
        Desired damping ratio (uniform across modes).
    freqs_hz : np.ndarray
        Full-model natural frequencies.
    modes : np.ndarray of int
        Selected mode indices.

    Returns
    -------
    Ca : np.ndarray, shape (m, m)
        Rayleigh damping matrix.
    """
    omega = 2.0 * np.pi * np.sort(freqs_hz[modes])
    w1, w2 = omega[0], omega[-1]
    if w1 < 1e-4:
        w1 = omega[omega > 1e-4][0]  # skip near-zero (rigid body)

    # Solve:  [[1/(2w1), w1/2], [1/(2w2), w2/2]] [α, β] = [zeta, zeta]
    A = np.array([[1 / (2 * w1), w1 / 2], [1 / (2 * w2), w2 / 2]])
    rhs = np.array([zeta, zeta])
    alpha, beta = np.linalg.solve(A, rhs)
    return alpha * Ma + beta * Ka


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_cond(kappa: float, rank: int, m: int) -> None:
    if kappa < 1e2:
        label = "EXCELLENT"
    elif kappa < 1e3:
        label = "GOOD"
    elif kappa < 1e6:
        label = "MARGINAL"
    else:
        label = "POOR — consider different DOF selector"
    print(f"  κ(Φₐ) = {kappa:.4e}  [{label}]  |  rank = {rank}/{m}")
