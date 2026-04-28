"""
pyserep.models.synthetic
===========================
Built-in synthetic finite element model generators for testing,
benchmarking, and example scripts.

Models
------
spring_chain          — 1D spring-mass chain (N DOFs, 1 direction)
euler_beam            — 1D Euler-Bernoulli beam (N elements, 2 DOFs/node)
plate_2d              — 2D rectangular thin plate (Kirchhoff, FD discretisation)
random_symmetric      — Dense random symmetric positive-definite (K, M) pair

All functions return (K, M) as ``scipy.sparse.csc_matrix`` pairs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────────────────────
# 1D Spring-mass chain
# ─────────────────────────────────────────────────────────────────────────────

def spring_chain(
    n: int = 300,
    k: float = 1e5,
    m: float = 1.0,
    fixed_left: bool = True,
    fixed_right: bool = False,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    1D spring-mass chain.

    Physical model:  o--k--o--k--o ... --k--o
                     m     m     m         m

    Parameters
    ----------
    n : int
        Number of mass nodes (= DOFs).
    k : float
        Spring stiffness (N/m).
    m : float
        Mass of each node (kg).
    fixed_left : bool
        If True, node 0 is fixed (Kₐ[0,0] = k, not 2k).
    fixed_right : bool
        If True, node n-1 is also fixed.

    Returns
    -------
    K, M : scipy.sparse.csc_matrix

    Examples
    --------
    >>> K, M = spring_chain(n=500, k=2e5)
    >>> K.shape
    (500, 500)
    """
    K = sp.diags(
        [[-k] * (n - 1), [2 * k] * n, [-k] * (n - 1)],
        [-1, 0, 1],
        format="csc",
    ).astype(float)
    M = sp.eye(n, format="csc", dtype=float) * m

    if fixed_left:
        K[0, 0] = k
    if fixed_right:
        K[n - 1, n - 1] = k

    return K, M


# ─────────────────────────────────────────────────────────────────────────────
# 1D Euler-Bernoulli beam
# ─────────────────────────────────────────────────────────────────────────────

def euler_beam(
    n_elements: int = 50,
    length: float = 1.0,
    EI: float = 1e4,
    rho_A: float = 1.0,
    fixed_left: bool = True,
    fixed_right: bool = False,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    1D Euler-Bernoulli beam (consistent mass matrix).

    DOFs per node: transverse displacement w and rotation θ.
    Total DOFs: 2 × (n_elements + 1).
    Fixed BCs applied by row/column zeroing with large diagonal penalty.

    Parameters
    ----------
    n_elements : int
    length : float
        Total beam length (m).
    EI : float
        Bending stiffness (N·m²).
    rho_A : float
        Mass per unit length (kg/m).
    fixed_left, fixed_right : bool
        Clamped boundary conditions at the respective ends.

    Returns
    -------
    K, M : scipy.sparse.csc_matrix, shape (2*(n_elements+1), ...)
    """
    n_nodes = n_elements + 1
    n_dof   = 2 * n_nodes
    L_e     = length / n_elements   # element length

    K_d = np.zeros((n_dof, n_dof))
    M_d = np.zeros((n_dof, n_dof))

    # Element stiffness and consistent mass matrices (Hermite shape functions)
    ke = EI / L_e ** 3 * np.array([
        [ 12,  6*L_e,   -12,  6*L_e],
        [ 6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12, -6*L_e,    12, -6*L_e],
        [ 6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2],
    ])
    me = rho_A * L_e / 420.0 * np.array([
        [156,  22*L_e,   54, -13*L_e],
        [22*L_e, 4*L_e**2, 13*L_e, -3*L_e**2],
        [54,  13*L_e,  156, -22*L_e],
        [-13*L_e, -3*L_e**2, -22*L_e, 4*L_e**2],
    ])

    for e in range(n_elements):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K_d[gi, gj] += ke[i, j]
                M_d[gi, gj] += me[i, j]

    # Apply clamped BCs by penalty method
    penalty = 1e15
    if fixed_left:
        K_d[0, 0]  += penalty
        K_d[1, 1]  += penalty
        M_d[0, 0]  += 1.0   # small mass to avoid rank deficiency
        M_d[1, 1]  += 1.0
    if fixed_right:
        K_d[-2, -2] += penalty
        K_d[-1, -1] += penalty
        M_d[-2, -2] += 1.0
        M_d[-1, -1] += 1.0

    return sp.csc_matrix(K_d), sp.csc_matrix(M_d)


# ─────────────────────────────────────────────────────────────────────────────
# 2D thin plate (Kirchhoff, finite difference)
# ─────────────────────────────────────────────────────────────────────────────

def plate_2d(
    nx: int = 10,
    ny: int = 10,
    lx: float = 1.0,
    ly: float = 1.0,
    D: float = 1e3,
    rho_h: float = 1.0,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    2D rectangular Kirchhoff plate using finite differences.

    Simply supported on all four edges.
    DOFs: transverse displacement w at interior grid points only.
    Total DOFs: (nx-1) × (ny-1).

    Parameters
    ----------
    nx, ny : int
        Number of intervals in x and y (interior nodes = nx-1, ny-1).
    lx, ly : float
        Plate dimensions (m).
    D : float
        Flexural rigidity D = E h³ / (12(1-ν²))  (N·m).
    rho_h : float
        Mass per unit area ρ·h  (kg/m²).

    Returns
    -------
    K, M : scipy.sparse.csc_matrix

    Notes
    -----
    Uses the 13-point biharmonic finite difference stencil for ∇⁴w.
    """
    dx = lx / nx
    dy = ly / ny
    n_int_x = nx - 1
    n_int_y = ny - 1
    N = n_int_x * n_int_y   # total interior DOFs

    def idx(i, j):
        """(i, j) → DOF index, 0 ≤ i < n_int_x, 0 ≤ j < n_int_y."""
        return i * n_int_y + j

    rows, cols, vals = [], [], []

    def _add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # FD coefficients for ∇⁴w = (1/dx⁴)·δₓₓₓₓ + 2/(dx²dy²)·δₓₓyy + (1/dy⁴)·δyyyy
    a  = D / dx**4
    b  = D / dy**4
    c  = 2 * D / (dx**2 * dy**2)

    for i in range(n_int_x):
        for j in range(n_int_y):
            r = idx(i, j)

            # Central point coefficient
            _add(r, r, 2*a + 2*b + 4*c + 2*c)

            # x-direction (zero at boundary → simply supported)
            for di in [-1, 1]:
                ni = i + di
                if 0 <= ni < n_int_x:
                    _add(r, idx(ni, j), -(4*a + 2*c) / 2)
                # Simply supported: w=0 at boundary, no contribution from ghost nodes

            for di in [-2, 2]:
                ni = i + di
                if 0 <= ni < n_int_x:
                    _add(r, idx(ni, j), a)

            # y-direction
            for dj in [-1, 1]:
                nj = j + dj
                if 0 <= nj < n_int_y:
                    _add(r, idx(i, nj), -(4*b + 2*c) / 2)

            for dj in [-2, 2]:
                nj = j + dj
                if 0 <= nj < n_int_y:
                    _add(r, idx(i, nj), b)

            # Mixed term
            for di in [-1, 1]:
                for dj in [-1, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_int_x and 0 <= nj < n_int_y:
                        _add(r, idx(ni, nj), c)

    K = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
    M = sp.eye(N, format="csc", dtype=float) * (rho_h * dx * dy)

    # Symmetrise K (small asymmetry from truncated stencil at boundaries)
    K = 0.5 * (K + K.T)
    return K, M


# ─────────────────────────────────────────────────────────────────────────────
# Random dense positive-definite pair
# ─────────────────────────────────────────────────────────────────────────────

def random_symmetric_pd(
    n: int = 50,
    kappa_K: float = 1e4,
    seed: int = 42,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Random symmetric positive-definite (K, M) pair.

    Useful for unit testing and algorithm benchmarks where the
    structural interpretation is irrelevant.

    Parameters
    ----------
    n : int
        Matrix size.
    kappa_K : float
        Target condition number for K.
    seed : int
        Random seed.

    Returns
    -------
    K, M : scipy.sparse.csc_matrix (dense matrices wrapped in sparse)
    """
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    lam_K = np.logspace(0, np.log10(kappa_K), n)
    K_d = Q @ np.diag(lam_K) @ Q.T
    K_d = 0.5 * (K_d + K_d.T)

    M_d = Q @ np.diag(rng.uniform(0.5, 1.5, n)) @ Q.T
    M_d = 0.5 * (M_d + M_d.T)

    return sp.csc_matrix(K_d), sp.csc_matrix(M_d)


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────

def model_info(K: sp.spmatrix, M: sp.spmatrix, label: str = "") -> str:
    """Return a one-line description of a (K, M) pair."""
    N       = K.shape[0]
    nnz_K   = K.nnz
    sparsity = 1 - nnz_K / N**2
    tag      = f"[{label}] " if label else ""
    return (
        f"{tag}N = {N:,} DOFs  |  "
        f"K nnz = {nnz_K:,}  ({sparsity*100:.2f}% sparse)  |  "
        f"M nnz = {M.nnz:,}"
    )
