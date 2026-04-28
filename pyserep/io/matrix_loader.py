"""
pyserep.io.matrix_loader
==========================
Universal matrix loader with automatic format detection and validation.

Supported formats
-----------------
* Matrix Market (.mtx, .mm)  — via scipy.io.mmread
* Harwell-Boeing (.rua, .rb) — via scipy.io.hb_read
* NumPy dense (.npy)         — loaded and wrapped in CSC
* NumPy sparse (.npz)        — via scipy.sparse.load_npz
* HDF5 (.h5, .hdf5)          — optional, requires h5py
* CSV / text (.csv, .txt)    — dense, via numpy.loadtxt (small matrices only)

Notes
-----
All returned matrices are in CSC format for efficient column operations.
Symmetry is verified (warning issued if asymmetry > tolerance).
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────────────────────
# Single-matrix loader
# ─────────────────────────────────────────────────────────────────────────────

def load_matrix(path: str, symmetry_tol: float = 1e-8) -> sp.csc_matrix:
    """
    Load a single sparse or dense matrix from disk, returning a CSC matrix.

    Parameters
    ----------
    path : str
        Absolute or relative path to the matrix file.
    symmetry_tol : float
        If the relative asymmetry ``|A - Aᵀ|_F / |A|_F`` exceeds this
        threshold, a ``UserWarning`` is issued (does not abort).

    Returns
    -------
    scipy.sparse.csc_matrix
        The loaded matrix in CSC format.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format is unrecognised or the file cannot be parsed.

    Examples
    --------
    >>> K = load_matrix("StiffMatrixmm.mtx")
    >>> K.shape
    (66525, 66525)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Matrix file not found: '{path}'")

    ext = os.path.splitext(path)[1].lower()
    mat = _dispatch_load(path, ext)

    # Convert to CSC
    if sp.issparse(mat):
        mat = mat.tocsc()
    else:
        mat = sp.csc_matrix(mat)

    _check_symmetry(mat, path, symmetry_tol)
    return mat


def _dispatch_load(path: str, ext: str) -> sp.spmatrix | np.ndarray:
    """Dispatch to format-specific loader."""
    if ext in (".mtx", ".mm", ""):
        try:
            return sio.mmread(path)
        except Exception as exc:
            raise ValueError(
                f"Failed to read Matrix Market file '{path}': {exc}"
            ) from exc

    if ext in (".rua", ".rb", ".rsa"):
        try:
            data = sio.hb_read(path)
            return data
        except Exception as exc:
            raise ValueError(
                f"Failed to read Harwell-Boeing file '{path}': {exc}"
            ) from exc

    if ext == ".npz":
        try:
            return sp.load_npz(path)
        except Exception as exc:
            raise ValueError(
                f"Failed to load sparse NumPy archive '{path}': {exc}"
            ) from exc

    if ext == ".npy":
        return np.load(path)

    if ext in (".h5", ".hdf5"):
        return _load_hdf5(path)

    if ext in (".csv", ".txt"):
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim != 2:
            raise ValueError(f"CSV/TXT '{path}' did not yield a 2D array.")
        return arr

    # Last resort: try Matrix Market
    try:
        return sio.mmread(path)
    except Exception:
        pass

    raise ValueError(
        f"Unrecognised file extension '{ext}' for path '{path}'.  "
        "Supported: .mtx, .mm, .rua, .rb, .npz, .npy, .h5, .hdf5, .csv"
    )


def _load_hdf5(path: str) -> sp.csc_matrix:
    """Load a sparse matrix stored in an HDF5 file (requires h5py)."""
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "h5py is required to load HDF5 files.  "
            "Install it with: pip install h5py"
        ) from exc

    import h5py
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if all(k in keys for k in ("data", "indices", "indptr", "shape")):
            data    = f["data"][:]
            indices = f["indices"][:]
            indptr  = f["indptr"][:]
            shape   = tuple(f["shape"][:])
            return sp.csc_matrix((data, indices, indptr), shape=shape)
        # Fallback: look for a dense dataset
        for key in keys:
            arr = f[key][:]
            if arr.ndim == 2:
                return sp.csc_matrix(arr)
    raise ValueError(
        f"Could not extract a matrix from HDF5 file '{path}'.  "
        "Expected datasets: data, indices, indptr, shape  (or a dense 2D array)."
    )


def _check_symmetry(mat: sp.csc_matrix, path: str, tol: float) -> None:
    """
    Check matrix symmetry.

    Issues a UserWarning if the relative asymmetry exceeds *tol*.
    Raises ValueError if asymmetry exceeds 0.01 (1%), which indicates
    a structurally wrong matrix (e.g. a non-symmetric FE export) that would
    corrupt SEREP results silently.

    pyserep requires **real symmetric** K and M.  Non-symmetric matrices
    are not supported and will produce incorrect eigenvalues and FRFs.
    """
    diff = mat - mat.T
    norm_diff = sp.linalg.norm(diff)
    norm_mat  = sp.linalg.norm(mat)
    rel = norm_diff / (norm_mat + 1e-300)

    if rel > 0.01:
        raise ValueError(
            f"Matrix '{os.path.basename(path)}' has severe asymmetry "
            f"(relative error {rel:.2e} > 1%). "
            "pyserep requires symmetric matrices. "
            "Use enforce_symmetry(mat) to fix numerical asymmetry."
        )
    if rel > tol:
        warnings.warn(
            f"Matrix '{os.path.basename(path)}' has relative asymmetry "
            f"{rel:.2e} (tolerance {tol:.0e}). "
            "pyserep requires symmetric K and M. "
            "Use enforce_symmetry(mat) to project onto its symmetric form.",
            UserWarning,
            stacklevel=4,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pair loader
# ─────────────────────────────────────────────────────────────────────────────

def load_matrices(
    stiffness_path: str,
    mass_path: str,
    verbose: bool = True,
    symmetry_tol: float = 1e-8,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Load stiffness and mass matrices from disk.

    Parameters
    ----------
    stiffness_path : str
        Path to the stiffness matrix file.
    mass_path : str
        Path to the mass matrix file.
    verbose : bool
        Print loading progress, matrix statistics, and memory usage.
    symmetry_tol : float
        Symmetry tolerance forwarded to :func:`load_matrix`.

    Returns
    -------
    K : scipy.sparse.csc_matrix
        Stiffness matrix.
    M : scipy.sparse.csc_matrix
        Mass matrix.

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    ValueError
        If K and M have incompatible shapes or are not square.

    Examples
    --------
    >>> K, M = load_matrices("K.mtx", "M.mtx")
    >>> K.shape == M.shape
    True
    """
    if verbose:
        print("[I/O] Loading stiffness matrix …", end=" ", flush=True)
    t0 = time.perf_counter()
    K = load_matrix(stiffness_path, symmetry_tol=symmetry_tol)
    if verbose:
        mem_mb = (K.data.nbytes + K.indices.nbytes + K.indptr.nbytes) / 1e6
        print(
            f"done  ({K.shape[0]:,} × {K.shape[1]:,}, "
            f"nnz={K.nnz:,}, {mem_mb:.1f} MB, "
            f"{time.perf_counter()-t0:.2f}s)"
        )

    if verbose:
        print("[I/O] Loading mass matrix    …", end=" ", flush=True)
    t0 = time.perf_counter()
    M = load_matrix(mass_path, symmetry_tol=symmetry_tol)
    if verbose:
        mem_mb = (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1e6
        print(
            f"done  ({M.shape[0]:,} × {M.shape[1]:,}, "
            f"nnz={M.nnz:,}, {mem_mb:.1f} MB, "
            f"{time.perf_counter()-t0:.2f}s)"
        )

    if K.shape != M.shape:
        raise ValueError(
            f"K and M have incompatible shapes: {K.shape} vs {M.shape}"
        )
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"Matrices must be square, got shape {K.shape}")

    if verbose:
        sparsity = 1.0 - K.nnz / (K.shape[0] ** 2)
        print(f"[I/O] N = {K.shape[0]:,} DOFs | Sparsity: {sparsity*100:.4f}%")

    return K, M


# ─────────────────────────────────────────────────────────────────────────────
# Enforce symmetry utility
# ─────────────────────────────────────────────────────────────────────────────

def enforce_symmetry(mat: sp.spmatrix) -> sp.csc_matrix:
    """
    Enforce symmetry: ``A ← 0.5 * (A + Aᵀ)``.

    Parameters
    ----------
    mat : scipy.sparse matrix

    Returns
    -------
    scipy.sparse.csc_matrix
        Symmetric version of *mat*.
    """
    return (0.5 * (mat + mat.T)).tocsc()



# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight symmetry and definiteness checks
# ─────────────────────────────────────────────────────────────────────────────

def check_symmetric_pd(
    K: sp.spmatrix,
    M: sp.spmatrix,
    raise_on_failure: bool = True,
) -> dict:
    """
    Verify that (K, M) satisfy the symmetry requirements for SEREP.

    pyserep requires:

    * K — real, symmetric, positive semi-definite
      (K can have zero eigenvalues for rigid-body modes)
    * M — real, symmetric, positive definite
      (M must be non-singular)

    Parameters
    ----------
    K, M : scipy.sparse matrices
    raise_on_failure : bool
        If True (default), raise ``ValueError`` if either check fails.
        If False, return the report dict regardless of outcome.

    Returns
    -------
    dict with keys:
        ``K_symmetry_error``, ``M_symmetry_error``,
        ``K_is_symmetric``, ``M_is_symmetric``,
        ``M_is_positive_definite``, ``passed``, ``message``

    Examples
    --------
    >>> from pyserep import load_matrices, check_symmetric_pd
    >>> K, M = load_matrices("K.mtx", "M.mtx", verbose=False)
    >>> report = check_symmetric_pd(K, M)
    >>> print(report["message"])
    """
    def _rel_asym(A):
        nA = sp.linalg.norm(A)
        return 0.0 if nA < 1e-300 else float(sp.linalg.norm(A - A.T) / nA)

    k_err = _rel_asym(K)
    m_err = _rel_asym(M)
    k_sym = k_err < 1e-6
    m_sym = m_err < 1e-6

    import numpy.linalg as nla
    n  = M.shape[0]
    ns = min(n, 500)
    try:
        Msub = M[:ns, :ns].toarray()
        nla.cholesky(Msub)
        m_pd = True
    except nla.LinAlgError:
        m_pd = False

    passed = k_sym and m_sym and m_pd

    k_msg = "PASS" if k_sym else "FAIL — use enforce_symmetry(K)"
    m_msg = "PASS" if m_sym else "FAIL — use enforce_symmetry(M)"
    p_msg = "PASS" if m_pd else "FAIL — M has non-positive eigenvalues"
    o_msg = "PASS — safe to proceed" if passed else "FAIL — fix issues above"

    message = (
        "Symmetry & definiteness check:\n"
        f"  K relative asymmetry : {k_err:.2e}  ({k_msg})\n"
        f"  M relative asymmetry : {m_err:.2e}  ({m_msg})\n"
        f"  M positive definite  : {p_msg}\n"
        f"  Overall              : {o_msg}"
    )

    if raise_on_failure and not passed:
        raise ValueError(
            message
            + "\n\npyserep requires symmetric positive (semi-)definite matrices."
            " Use enforce_symmetry(mat) to fix near-symmetric matrices."
        )

    return {
        "K_symmetry_error"       : k_err,
        "M_symmetry_error"       : m_err,
        "K_is_symmetric"         : k_sym,
        "M_is_symmetric"         : m_sym,
        "M_is_positive_definite" : m_pd,
        "passed"                 : passed,
        "message"                : message,
    }
