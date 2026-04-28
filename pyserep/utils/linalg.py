"""
pyserep.utils.linalg
========================
Shared linear algebra utilities used throughout the library.

Functions
---------
condition_number_estimate   — fast Krylov-based κ estimate (avoids full SVD)
rank_revealing_qr           — rank-revealing QR with column pivoting
safe_pinv                   — pseudoinverse with automatic rank truncation
mass_normalise              — enforce Φᵀ M Φ = I
symmetrise                  — enforce A ← 0.5(A + Aᵀ) for sparse or dense
force_positive_definite     — perturb diagonal until Cholesky succeeds
modal_strain_energy         — Φᵀ K Φ diagonal (modal stiffness / ωᵢ²)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────────────────────
# Condition number
# ─────────────────────────────────────────────────────────────────────────────

def condition_number_estimate(
    A: np.ndarray,
    method: str = "exact",
) -> float:
    """
    Estimate the 2-norm condition number of a dense matrix A.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    method : str
        ``"exact"``  — full SVD  (always accurate, O(mn²))
        ``"fast"``   — power iteration on AᵀA  (cheaper for large m)

    Returns
    -------
    float — κ(A) = σ_max / σ_min
    """
    if method == "exact":
        sv = la.svdvals(A)
        sv = sv[sv > 0]
        return float(sv[0] / sv[-1]) if len(sv) > 0 else np.inf

    elif method == "fast":
        # Estimate largest and smallest singular values via power iteration
        m, n = A.shape
        rng = np.random.default_rng(0)
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        # Power iteration for σ_max
        for _ in range(30):
            u = A @ v
            u /= np.linalg.norm(u)
            v = A.T @ u
            s_max = np.linalg.norm(v)
            v /= s_max
        # Inverse iteration for σ_min (via least-squares solve)
        try:
            AtA = A.T @ A
            w = rng.standard_normal(n)
            w /= np.linalg.norm(w)
            for _ in range(20):
                w = la.solve(AtA, w, assume_a="sym")
                s_min_inv = np.linalg.norm(w)
                w /= s_min_inv
            s_min = 1.0 / s_min_inv
        except la.LinAlgError:
            s_min = 0.0
        return float(s_max / max(s_min, 1e-300))
    else:
        raise ValueError(f"Unknown method '{method}'; choose 'exact' or 'fast'.")


# ─────────────────────────────────────────────────────────────────────────────
# Rank-revealing QR
# ─────────────────────────────────────────────────────────────────────────────

def rank_revealing_qr(
    A: np.ndarray,
    tol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Rank-revealing QR factorisation with column pivoting.

    Computes:  A P = Q R   where P is a permutation.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    tol : float, optional
        Rank threshold relative to |R[0,0]|.  Default: machine epsilon × max(m,n).

    Returns
    -------
    Q : np.ndarray, shape (m, m)
    R : np.ndarray, shape (m, n)
    perm : np.ndarray of int, shape (n,)  — column permutation
    rank : int — estimated numerical rank

    Examples
    --------
    >>> Q, R, perm, r = rank_revealing_qr(phi_a)
    >>> perm[:r]   # top-r most informative DOFs
    """
    Q, R, perm = la.qr(A, pivoting=True)
    if tol is None:
        tol = np.finfo(float).eps * max(A.shape) * abs(R[0, 0])
    rank = int(np.sum(np.abs(np.diag(R)) > tol))
    return Q, R, perm, rank


# ─────────────────────────────────────────────────────────────────────────────
# Safe pseudoinverse
# ─────────────────────────────────────────────────────────────────────────────

def safe_pinv(
    A: np.ndarray,
    rcond: Optional[float] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute the Moore–Penrose pseudoinverse with automatic rank truncation.

    Drops singular values below ``rcond * σ_max`` to avoid numerical blow-up
    from near-zero singular values.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
    rcond : float, optional
        Relative condition threshold.  Default: 100 × machine epsilon.
    verbose : bool

    Returns
    -------
    np.ndarray, shape (n, m)
    """
    if rcond is None:
        rcond = 100.0 * np.finfo(float).eps
    A_pinv = np.linalg.pinv(A, rcond=rcond)
    if verbose:
        kappa = condition_number_estimate(A)
        print(f"[safe_pinv] κ = {kappa:.4e}  rcond = {rcond:.2e}")
    return A_pinv


# ─────────────────────────────────────────────────────────────────────────────
# Mass normalisation
# ─────────────────────────────────────────────────────────────────────────────

def mass_normalise(
    phi: np.ndarray,
    M: sp.spmatrix,
) -> np.ndarray:
    """
    Normalise mode shapes so that Φᵢᵀ M Φᵢ = 1 for every column i.

    Parameters
    ----------
    phi : np.ndarray, shape (N, m)
    M : scipy.sparse matrix, shape (N, N)

    Returns
    -------
    np.ndarray, shape (N, m) — mass-normalised phi
    """
    MPhi  = M @ phi                                     # (N, m)
    norms = np.einsum("ij,ij->j", phi, MPhi)            # diag(ΦᵀMΦ)
    norms = np.where(norms > 1e-30, norms, 1.0)
    return phi / np.sqrt(norms)[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# Symmetrise
# ─────────────────────────────────────────────────────────────────────────────

def symmetrise(A):
    """
    Enforce symmetry:  A ← 0.5 (A + Aᵀ).

    Works for both dense ``np.ndarray`` and ``scipy.sparse`` matrices.
    """
    if sp.issparse(A):
        return (0.5 * (A + A.T)).tocsc()
    return 0.5 * (A + A.T)


# ─────────────────────────────────────────────────────────────────────────────
# Force positive definite
# ─────────────────────────────────────────────────────────────────────────────

def force_positive_definite(
    A: np.ndarray,
    shift_factor: float = 1e-8,
    max_iter: int = 50,
) -> Tuple[np.ndarray, float]:
    """
    Make A symmetric positive definite by adding a small multiple of the
    identity until Cholesky succeeds.

    Uses the modified Cholesky approach of Gill, Murray & Wright (1981).

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
    shift_factor : float
        Initial shift as a fraction of |diag(A)|_max.
    max_iter : int

    Returns
    -------
    A_pd : np.ndarray — positive-definite version of A
    shift : float — total shift applied (0 if A was already PD)

    Examples
    --------
    >>> Ka_pd, shift = force_positive_definite(Ka)
    >>> np.linalg.cholesky(Ka_pd)   # succeeds
    """
    A = symmetrise(A)
    shift = 0.0
    diag_max = np.abs(np.diag(A)).max()
    delta = shift_factor * diag_max

    for _ in range(max_iter):
        try:
            np.linalg.cholesky(A + shift * np.eye(len(A)))
            return A + shift * np.eye(len(A)), shift
        except np.linalg.LinAlgError:
            shift = max(2 * shift, delta)

    warnings.warn(
        f"force_positive_definite did not converge after {max_iter} iterations.  "
        f"Applied shift = {shift:.2e}.  Matrix may be highly ill-conditioned.",
        RuntimeWarning,
        stacklevel=2,
    )
    return A + shift * np.eye(len(A)), shift


# ─────────────────────────────────────────────────────────────────────────────
# Modal strain energy
# ─────────────────────────────────────────────────────────────────────────────

def modal_strain_energy(
    phi: np.ndarray,
    K: sp.spmatrix,
    selected_modes: np.ndarray,
) -> np.ndarray:
    """
    Compute the modal strain energy (MSE) for each selected mode.

    MSE_i = φᵢᵀ K φᵢ  (= ωᵢ² for mass-normalised modes)

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_all_modes)
    K : sparse stiffness matrix
    selected_modes : np.ndarray of int

    Returns
    -------
    np.ndarray, shape (len(selected_modes),)  — modal strain energies
    """
    Phi = phi[:, selected_modes]          # (N, m)
    KPhi = K @ Phi                        # (N, m)
    return np.einsum("ij,ij->j", Phi, KPhi)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse submatrix extraction
# ─────────────────────────────────────────────────────────────────────────────

def sparse_submatrix(
    A: sp.spmatrix,
    row_idx: np.ndarray,
    col_idx: Optional[np.ndarray] = None,
) -> sp.csc_matrix:
    """
    Extract a submatrix A[row_idx, :][:, col_idx] efficiently.

    Parameters
    ----------
    A : scipy.sparse matrix
    row_idx : np.ndarray of int
    col_idx : np.ndarray of int, optional — defaults to row_idx (square sub-block)

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    if col_idx is None:
        col_idx = row_idx
    return A[row_idx, :][:, col_idx].tocsc()


# ─────────────────────────────────────────────────────────────────────────────
# FRF residue computation
# ─────────────────────────────────────────────────────────────────────────────

def modal_residues(
    phi: np.ndarray,
    force_dofs: list,
    output_dofs: list,
    selected_modes: np.ndarray,
) -> np.ndarray:
    """
    Compute modal residues Rᵢ = φᵢ(f) · φᵢ(o) for each mode and DOF pair.

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_all_modes)
    force_dofs, output_dofs : list of int
    selected_modes : np.ndarray of int

    Returns
    -------
    np.ndarray, shape (m, n_pairs)
    """
    m = len(selected_modes)
    n_pairs = len(force_dofs)
    R = np.zeros((m, n_pairs))
    for k, (fi, oi) in enumerate(zip(force_dofs, output_dofs)):
        R[:, k] = phi[fi, selected_modes] * phi[oi, selected_modes]
    return R
