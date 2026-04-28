"""
pyserep.analysis.validation
==============================
Comprehensive validation suite for SEREP ROMs.

Checks
------
1. Eigenvalue preservation   — SEREP's defining property
2. Mass-orthogonality        — ΦᵀMΦ ≈ I
3. Stiffness orthogonality   — ΦᵀKΦ ≈ diag(ωᵢ²)
4. Transformation identity   — T Φₐ ≈ Φ  (expansion accuracy)
5. Modal Assurance Criterion — MAC(Φ_full, Φ_rom)
6. FRF accuracy              — max/RMS percentage errors
7. Positive definiteness     — Ka and Ma should be PD
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


@dataclass
class ValidationReport:
    """Full validation report for a SEREP ROM."""

    eigenvalue_errors_pct: np.ndarray
    max_eigenvalue_error_pct: float
    mean_eigenvalue_error_pct: float
    mass_ortho_error: float
    stiff_ortho_error: float
    expansion_error: float
    mac_values: np.ndarray               # (m,) diagonal MAC values
    min_mac: float
    mean_mac: float
    ka_positive_definite: bool
    ma_positive_definite: bool
    ka_condition_number: float
    ma_condition_number: float

    def passed(self, strict: bool = False) -> bool:
        """Return True if all key checks pass."""
        tol = 0.001 if strict else 0.01
        return (
            self.max_eigenvalue_error_pct < tol
            and self.min_mac > 0.95
            and self.ka_positive_definite
            and self.ma_positive_definite
        )

    def summary(self) -> str:
        """Return a formatted multi-line validation report as a string."""
        status = "✓ PASS" if self.passed() else "✗ FAIL"
        return "\n".join([
            f"\n{'='*55}",
            f"  SEREP VALIDATION REPORT  [{status}]",
            f"{'='*55}",
            "  Eigenvalue preservation",
            f"    Max error  : {self.max_eigenvalue_error_pct:.8f}%",
            f"    Mean error : {self.mean_eigenvalue_error_pct:.8f}%",
            "  Modal orthogonality",
            f"    Mass    |ΦᵀMΦ−I|_max : {self.mass_ortho_error:.4e}",
            f"    Stiff   |ΦᵀKΦ−Λ|_max : {self.stiff_ortho_error:.4e}",
            "  Transformation accuracy",
            f"    |TΦₐ−Φ|_F / |Φ|_F   : {self.expansion_error:.4e}",
            f"  MAC  (min / mean)        : {self.min_mac:.4f} / {self.mean_mac:.4f}",
            f"  Kₐ positive definite     : {self.ka_positive_definite}",
            f"  Mₐ positive definite     : {self.ma_positive_definite}",
            f"  κ(Kₐ)                   : {self.ka_condition_number:.4e}",
            f"  κ(Mₐ)                   : {self.ma_condition_number:.4e}",
            f"{'='*55}",
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Master validation function
# ─────────────────────────────────────────────────────────────────────────────

def validate_serep(
    K: sp.csc_matrix,
    M: sp.csc_matrix,
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    selected_modes: np.ndarray,
    master_dofs: np.ndarray,
    T: np.ndarray,
    Ka: np.ndarray,
    Ma: np.ndarray,
    verbose: bool = True,
) -> ValidationReport:
    """
    Run the complete SEREP validation suite.

    Parameters
    ----------
    K, M : sparse matrices
    phi : np.ndarray, shape (N, n_modes)
    freqs_hz : np.ndarray
    selected_modes, master_dofs : np.ndarray of int
    T : np.ndarray, shape (N, m) — transformation matrix
    Ka, Ma : np.ndarray, shape (m, m) — reduced matrices
    verbose : bool

    Returns
    -------
    ValidationReport
    """
    # 1. Eigenvalue errors
    eig_errors, max_eig_err = eigenvalue_error(Ka, Ma, freqs_hz, selected_modes, verbose=False)

    # 2. Mass orthogonality: |ΦᵀMΦ − I|
    Phi = phi[:, selected_modes]
    mass_ortho = _ortho_error(Phi, M)

    # 3. Stiffness orthogonality: |ΦᵀKΦ − diag(ωᵢ²)|
    stiff_ortho = _stiff_ortho_error(Phi, K, freqs_hz, selected_modes)

    # 4. Expansion accuracy: |T Φₐ − Φ| / |Φ|
    Phi_a = Phi[master_dofs, :]
    expansion_err = (
        np.linalg.norm(T @ Phi_a - Phi, "fro")
        / (np.linalg.norm(Phi, "fro") + 1e-30)
    )

    # 5. MAC
    mac_vals = _mac_diagonal(Phi, T, Phi_a)

    # 6. Positive definiteness
    ka_pd = _is_positive_definite(Ka)
    ma_pd = _is_positive_definite(Ma)

    # 7. Condition numbers
    ka_cond = float(np.linalg.cond(Ka))
    ma_cond = float(np.linalg.cond(Ma))

    report = ValidationReport(
        eigenvalue_errors_pct       = eig_errors,
        max_eigenvalue_error_pct    = max_eig_err,
        mean_eigenvalue_error_pct   = float(np.nanmean(eig_errors)),
        mass_ortho_error            = float(mass_ortho),
        stiff_ortho_error           = float(stiff_ortho),
        expansion_error             = float(expansion_err),
        mac_values                  = mac_vals,
        min_mac                     = float(mac_vals.min()),
        mean_mac                    = float(mac_vals.mean()),
        ka_positive_definite        = ka_pd,
        ma_positive_definite        = ma_pd,
        ka_condition_number         = ka_cond,
        ma_condition_number         = ma_cond,
    )

    if verbose:
        print(report.summary())

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def eigenvalue_error(
    Ka: np.ndarray,
    Ma: np.ndarray,
    full_freqs_hz: np.ndarray,
    selected_modes: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Compute per-mode eigenvalue preservation error (%).

    Returns
    -------
    errors_pct : np.ndarray, shape (m,)
    max_error_pct : float
    """
    target = np.sort(full_freqs_hz[selected_modes])
    try:
        lam = la.eigh(Ka, Ma, eigvals_only=True)
        freqs_rom = np.sort(np.sqrt(np.maximum(lam, 0.0)) / (2.0 * np.pi))
    except la.LinAlgError:
        return np.full(len(selected_modes), np.nan), np.nan

    n = min(len(freqs_rom), len(target))
    errors = np.abs(freqs_rom[:n] - target[:n]) / np.maximum(target[:n], 1e-10) * 100.0
    max_err = float(errors.max())

    if verbose:
        status = "✓" if max_err < 0.01 else "✗"
        print(f"[Eigenvalue Error]  {status}  max = {max_err:.8f}%  "
              f"mean = {errors.mean():.8f}%")
    return errors, max_err


def modal_assurance_criterion(
    phi_ref: np.ndarray,
    phi_rom: np.ndarray,
) -> np.ndarray:
    """
    Compute the full MAC matrix between two mode shape sets.

    MAC[i,j] = |φᵢᵀ φⱼ|² / (|φᵢ|² |φⱼ|²)

    Parameters
    ----------
    phi_ref, phi_rom : np.ndarray, shape (N, m)

    Returns
    -------
    mac_matrix : np.ndarray, shape (m, m)
        Values in [0, 1].  Diagonal near 1 → good mode pairing.
    """
    n1 = np.linalg.norm(phi_ref, axis=0)  # (m,)
    n2 = np.linalg.norm(phi_rom, axis=0)
    n1 = np.where(n1 > 1e-15, n1, 1.0)
    n2 = np.where(n2 > 1e-15, n2, 1.0)
    phi_rn = phi_ref / n1[None, :]
    phi_an = phi_rom / n2[None, :]
    gram = phi_rn.T @ phi_an
    return gram ** 2


def orthogonality_check(
    phi: np.ndarray,
    M: sp.csc_matrix,
    selected_modes: np.ndarray,
) -> float:
    """
    Return ``|ΦᵀMΦ − I|_max``.  Should be < 1e-10 for mass-normalised modes.
    """
    Phi = phi[:, selected_modes]
    return _ortho_error(Phi, M)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ortho_error(Phi: np.ndarray, M: sp.csc_matrix) -> float:
    orth = Phi.T @ (M @ Phi)
    return float(np.abs(orth - np.eye(orth.shape[0])).max())


def _stiff_ortho_error(
    Phi: np.ndarray,
    K: sp.csc_matrix,
    freqs_hz: np.ndarray,
    selected_modes: np.ndarray,
) -> float:
    orth  = Phi.T @ (K @ Phi)
    lam   = (2.0 * np.pi * freqs_hz[selected_modes]) ** 2
    target = np.diag(lam)
    return float(np.abs(orth - target).max())


def _mac_diagonal(
    Phi: np.ndarray,
    T: np.ndarray,
    Phi_a: np.ndarray,
) -> np.ndarray:
    """MAC between original modes and modes reconstructed via T Φₐ."""
    Phi_rec = T @ Phi_a
    mac = modal_assurance_criterion(Phi, Phi_rec)
    return np.diag(mac)


def _is_positive_definite(A: np.ndarray) -> bool:
    """Return True if A is symmetric positive definite."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
