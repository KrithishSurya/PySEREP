"""
pyserep.core.eigensolver
==========================
Generalised sparse eigenproblem solver for structural FE models.

Uses ``scipy.sparse.linalg.eigsh`` with shift-invert mode, which is the
industry-standard approach for extracting the lowest *k* eigenpairs from
large positive-definite (K, M) pencils without forming dense matrices.

Theory
------
We solve:

    K φᵢ = λᵢ M φᵢ,    λᵢ = ωᵢ²

The shift-invert transformation converts the sparse problem to:

    (K − σM)⁻¹ M φ = μ φ,    μ = 1 / (λ − σ)

where σ (shift) is chosen slightly positive to capture near-rigid-body
modes.  ARPACK's ``eigsh`` then finds the largest |μ| (corresponding to
the smallest λ), making convergence fast.

Mass normalisation
------------------
After extraction we enforce ``φᵢᵀ M φᵢ = 1`` so that the modal matrix
satisfies:  Φᵀ M Φ = I  and  Φᵀ K Φ = diag(ωᵢ²).
"""

from __future__ import annotations

import time
import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_eigenproblem(
    K: sp.csc_matrix,
    M: sp.csc_matrix,
    n_modes: int,
    sigma: float = 0.01,
    tol: float = 1e-10,
    maxiter: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalised eigenvalue problem ``K φ = λ M φ`` for the
    *n_modes* lowest eigenpairs.

    Parameters
    ----------
    K : scipy.sparse.csc_matrix
        Stiffness matrix (N × N), **real symmetric positive semi-definite**.
        Asymmetric K will produce complex eigenvalues and incorrect results.
    M : scipy.sparse.csc_matrix
        Mass matrix (N × N), **real symmetric positive definite**.
        M must be non-singular; singular M causes ARPACK to fail.
    n_modes : int
        Number of eigenpairs to extract.  Must be < N − 1.
    sigma : float
        Shift parameter (rad²/s²) for shift-invert.  A small positive
        value (e.g. 0.01) ensures rigid-body modes (λ ≈ 0) are captured.
    tol : float
        ARPACK convergence tolerance.  1e-10 is appropriate for structural
        analysis; decrease to 1e-12 for very high accuracy requirements.
    maxiter : int, optional
        Maximum ARPACK iterations.  Default: ``10 * N``.
    verbose : bool
        Print eigenvalue summary.

    Returns
    -------
    freqs_hz : np.ndarray, shape (n_modes,)
        Natural frequencies in Hz, sorted ascending.
    phi : np.ndarray, shape (N, n_modes)
        Mass-normalised mode shapes.  ``phi[:, i]`` corresponds to
        ``freqs_hz[i]``.

    Raises
    ------
    ValueError
        If ``n_modes >= N``.
    RuntimeWarning
        If ARPACK does not fully converge (partial results returned).

    Examples
    --------
    >>> freqs, phi = solve_eigenproblem(K, M, n_modes=100)
    >>> freqs.shape
    (100,)
    >>> phi.shape
    (66525, 100)
    """
    N = K.shape[0]
    if n_modes >= N - 1:
        raise ValueError(
            f"n_modes ({n_modes}) must be < N-1 = {N-1}.  "
            f"Requested {n_modes} eigenpairs from a {N}×{N} system."
        )
    n_req = min(n_modes, N - 2)
    maxiter = maxiter or 10 * N

    if verbose:
        print(
            f"\n[Eigensolver] Solving  K φ = λ M φ\n"
            f"  N = {N:,} DOFs  |  Requesting {n_req} eigenpairs\n"
            f"  Method: ARPACK shift-invert  σ = {sigma}  tol = {tol}",
            flush=True,
        )

    t0 = time.perf_counter()

    try:
        eigenvalues, eigenvectors = spla.eigsh(
            K,
            k=n_req,
            M=M,
            sigma=sigma,
            which="LM",       # largest |μ| → smallest eigenvalues
            tol=tol,
            maxiter=maxiter,
            mode="buckling",  # equivalent to shift-invert for definite (K,M)
        )
    except Exception:
        # Fall back to normal mode on failure
        try:
            eigenvalues, eigenvectors = spla.eigsh(
                K, k=n_req, M=M, sigma=sigma, which="LM",
                tol=tol, maxiter=maxiter,
            )
        except spla.ArpackNoConvergence as exc:
            eigenvalues  = exc.eigenvalues
            eigenvectors = exc.eigenvectors
            warnings.warn(
                f"ARPACK did not fully converge.  Got {len(eigenvalues)} of "
                f"{n_req} modes.  Results may be incomplete.",
                RuntimeWarning,
                stacklevel=2,
            )

    elapsed = time.perf_counter() - t0

    # ── Sort by ascending eigenvalue ──────────────────────────────────────────
    idx = np.argsort(eigenvalues)
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # ── Clamp near-zero negatives (numerical noise) ───────────────────────────
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # ── Convert to Hz ─────────────────────────────────────────────────────────
    freqs_hz = np.sqrt(eigenvalues) / (2.0 * np.pi)

    # ── Mass-normalise: φᵢᵀ M φᵢ = 1 ─────────────────────────────────────────
    eigenvectors = _mass_normalise(eigenvectors, M)

    if verbose:
        n_rb = int(np.sum(freqs_hz < 1.0))
        print(
            f"  Solved in {elapsed:.2f}s\n"
            f"  Rigid-body modes (f < 1 Hz): {n_rb}\n"
            f"  Frequency range: {freqs_hz[0]:.4f} – {freqs_hz[-1]:.2f} Hz\n"
            f"  Orthogonality error |ΦᵀMΦ - I|_max: "
            f"{_ortho_error(eigenvectors, M):.2e}"
        )

    return freqs_hz, eigenvectors


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mass_normalise(phi: np.ndarray, M: sp.csc_matrix) -> np.ndarray:
    """Normalise each mode so that φᵢᵀ M φᵢ = 1."""
    MPhi = M @ phi               # (N, m)
    norms = np.einsum("ij,ij->j", phi, MPhi)  # (m,) = diag(ΦᵀMΦ)
    norms = np.where(norms > 1e-30, norms, 1.0)
    return phi / np.sqrt(norms)[None, :]


def _ortho_error(phi: np.ndarray, M: sp.csc_matrix) -> float:
    """||ΦᵀMΦ - I||_max — orthogonality check."""
    orth = phi.T @ (M @ phi)
    return float(np.abs(orth - np.eye(orth.shape[0])).max())
