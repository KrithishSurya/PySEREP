"""
pyserep.utils.sparse_ops
============================
Utility operations on sparse matrices used throughout the pipeline.

Functions
---------
memory_mb               — report sparse matrix memory usage in MB
sparsity                — fraction of structural zeros
diagonal_scaling        — scale K and M for better eigensolver conditioning
generalized_diagonal_M  — check if M is diagonal (lumped mass)
apply_bcs               — apply fixed-DOF boundary conditions via penalty
reorder_rcm             — Reverse Cuthill-McKee reordering to reduce bandwidth
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as scsg

# ─────────────────────────────────────────────────────────────────────────────
# Inspection
# ─────────────────────────────────────────────────────────────────────────────

def memory_mb(mat: sp.spmatrix) -> float:
    """Return memory usage of a sparse matrix in MB (data + indices + indptr)."""
    if sp.issparse(mat):
        m = mat.tocsc()
        return (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e6
    return mat.nbytes / 1e6


def sparsity(mat: sp.spmatrix) -> float:
    """Return the fraction of structural zeros (higher = sparser)."""
    N = mat.shape[0] * mat.shape[1]
    return 1.0 - mat.nnz / N


def is_diagonal(mat: sp.spmatrix, tol: float = 1e-12) -> bool:
    """Return True if *mat* is effectively diagonal (lumped mass)."""
    off = mat - sp.diags(mat.diagonal())
    return float(abs(off).max()) < tol


def bandwidth(mat: sp.spmatrix) -> int:
    """Return the half-bandwidth of a sparse matrix."""
    coo = mat.tocoo()
    if coo.nnz == 0:
        return 0
    return int(np.max(np.abs(coo.row - coo.col)))


def matrix_stats(K: sp.spmatrix, M: sp.spmatrix, label: str = "") -> str:
    """Return a formatted statistics string for a (K, M) pair."""
    N = K.shape[0]
    tag = f"[{label}] " if label else ""
    return (
        f"{tag}N={N:,}  "
        f"K: nnz={K.nnz:,} mem={memory_mb(K):.1f}MB bw={bandwidth(K)}  |  "
        f"M: nnz={M.nnz:,} mem={memory_mb(M):.1f}MB "
        f"{'(diagonal)' if is_diagonal(M) else '(consistent)'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagonal scaling (improves ARPACK convergence)
# ─────────────────────────────────────────────────────────────────────────────

def diagonal_scaling(
    K: sp.spmatrix,
    M: sp.spmatrix,
) -> Tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """
    Scale (K, M) by the diagonal of M to improve eigensolver conditioning.

    Transformation:  K̃ = D⁻½ K D⁻½,  M̃ = D⁻½ M D⁻½

    where D = diag(M).  This converts the generalised problem to a
    standard one when M is diagonal (lumped mass), and improves
    conditioning for consistent mass matrices.

    Parameters
    ----------
    K, M : scipy.sparse matrices

    Returns
    -------
    K_scaled, M_scaled : scipy.sparse.csc_matrix
    D_inv_sqrt : np.ndarray, shape (N,) — scaling vector for back-transformation
    """
    d = np.array(M.diagonal())
    d = np.where(d > 0, d, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D = sp.diags(d_inv_sqrt, format="csc")
    K_sc = (D @ K @ D).tocsc()
    M_sc = (D @ M @ D).tocsc()
    K_sc = 0.5 * (K_sc + K_sc.T)
    M_sc = 0.5 * (M_sc + M_sc.T)
    return K_sc, M_sc, d_inv_sqrt


def unscale_modes(phi_scaled: np.ndarray, d_inv_sqrt: np.ndarray) -> np.ndarray:
    """Undo diagonal scaling: φ = D⁻½ φ̃  then mass-renormalise."""
    return phi_scaled * d_inv_sqrt[:, None]


# ─────────────────────────────────────────────────────────────────────────────
# Boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def apply_bcs(
    K: sp.spmatrix,
    M: sp.spmatrix,
    fixed_dofs: List[int],
    penalty: float = 1e15,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Apply fixed-DOF boundary conditions via the penalty method.

    Sets K[dof, dof] += penalty and M[dof, dof] = 1 for each fixed DOF.
    This is numerically equivalent to removing the rows/columns but
    preserves the matrix dimensions (convenient for DOF indexing).

    Parameters
    ----------
    K, M : sparse matrices
    fixed_dofs : list of int
    penalty : float

    Returns
    -------
    K_bc, M_bc : scipy.sparse.csc_matrix
    """
    K_lil = K.tolil()
    M_lil = M.tolil()
    for dof in fixed_dofs:
        K_lil[dof, dof] += penalty
        M_lil[dof, dof] = max(float(M_lil[dof, dof]), 1.0)
    return K_lil.tocsc(), M_lil.tocsc()


# ─────────────────────────────────────────────────────────────────────────────
# Reverse Cuthill-McKee reordering
# ─────────────────────────────────────────────────────────────────────────────

def reorder_rcm(
    K: sp.spmatrix,
    M: sp.spmatrix,
) -> Tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """
    Apply Reverse Cuthill-McKee (RCM) reordering to reduce bandwidth.

    Reduces the bandwidth of K (and correspondingly M), which can speed
    up sparse direct solvers used during FRF computation.

    Parameters
    ----------
    K, M : sparse matrices (square, symmetric)

    Returns
    -------
    K_rcm, M_rcm : scipy.sparse.csc_matrix — reordered matrices
    perm : np.ndarray of int — permutation vector (K_rcm = K[perm, :][:, perm])

    Notes
    -----
    The same permutation must be applied to DOF index arrays (master_dofs,
    force_dofs, etc.) when using the reordered matrices.  Use
    ``perm_inv = np.argsort(perm)`` to convert global DOF indices back.
    """
    perm = scsg.reverse_cuthill_mckee(K.tocsr(), symmetric_mode=True)
    K_rcm = K[perm, :][:, perm].tocsc()
    M_rcm = M[perm, :][:, perm].tocsc()
    return K_rcm, M_rcm, perm


# ─────────────────────────────────────────────────────────────────────────────
# DOF index utilities
# ─────────────────────────────────────────────────────────────────────────────

def ansys_dof(node: int, direction: int) -> int:
    """
    Convert Ansys node number + direction to a 0-based DOF index.

    DOF_index = (node_number − 1) × 3 + direction

    Parameters
    ----------
    node : int — 1-based Ansys node number
    direction : int — 0=UX, 1=UY, 2=UZ

    Returns
    -------
    int — 0-based DOF index
    """
    if direction not in (0, 1, 2):
        raise ValueError(f"direction must be 0, 1, or 2; got {direction}")
    return (node - 1) * 3 + direction


def dof_to_ansys(dof: int) -> Tuple[int, int]:
    """
    Convert a 0-based DOF index to (node_number, direction).

    Inverse of :func:`ansys_dof`.

    Returns
    -------
    (node_number, direction) — 1-based node, direction in {0,1,2}
    """
    return dof // 3 + 1, dof % 3


def build_dof_map(
    master_dofs: np.ndarray,
    force_dofs: List[int],
    output_dofs: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Map global force/output DOF indices → local indices within master_dofs.

    Parameters
    ----------
    master_dofs : np.ndarray of int — global master DOF indices
    force_dofs, output_dofs : list of int — global DOF indices

    Returns
    -------
    local_force, local_output : list of int

    Raises
    ------
    KeyError
        If any force/output DOF is not present in master_dofs.
    """
    dof_map = {int(d): i for i, d in enumerate(master_dofs)}
    missing = [d for d in force_dofs + output_dofs if d not in dof_map]
    if missing:
        raise KeyError(
            f"DOFs {missing} not found in master_dofs.  "
            "Pass these as required_dofs to select_dofs_eid()."
        )
    local_force  = [dof_map[d] for d in force_dofs]
    local_output = [dof_map[d] for d in output_dofs]
    return local_force, local_output
