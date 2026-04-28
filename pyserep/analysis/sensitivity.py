"""
pyserep.analysis.sensitivity
================================
Modal parameter sensitivity and uncertainty quantification for SEREP ROMs.

Functions
---------
eigenvalue_sensitivity      — ∂λᵢ/∂p via Nelson's method
mode_shape_sensitivity      — ∂φᵢ/∂p via Nelson's method
frf_sensitivity             — ∂H(ω)/∂p via direct differentiation
material_perturbation_study — FRF response under parametric perturbations
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import scipy.sparse as sp

# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def eigenvalue_sensitivity(
    K: sp.spmatrix,
    M: sp.spmatrix,
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    selected_modes: np.ndarray,
    dK_dp: sp.spmatrix,
    dM_dp: sp.spmatrix,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute eigenvalue sensitivities ∂λᵢ/∂p using Nelson's method.

    For mass-normalised modes (φᵢᵀ M φᵢ = 1):

        ∂λᵢ/∂p = φᵢᵀ (∂K/∂p − λᵢ ∂M/∂p) φᵢ

    Parameters
    ----------
    K, M : sparse matrices — full structural matrices
    phi : np.ndarray, shape (N, n_all_modes)
    freqs_hz : np.ndarray — full-model natural frequencies
    selected_modes : np.ndarray of int
    dK_dp, dM_dp : sparse matrices — parameter derivative matrices
        Represents ∂K/∂p and ∂M/∂p for a single parameter p.
        For a Young's modulus perturbation:  dK_dp = K/E, dM_dp = 0.
    verbose : bool

    Returns
    -------
    dlam_dp : np.ndarray, shape (m,) — ∂λᵢ/∂p in (rad/s)² per unit of p

    Examples
    --------
    Sensitivity of eigenvalues to a 1% stiffness increase:

    >>> dK_dp = 0.01 * K   # 1% increase in E
    >>> dM_dp = sp.csc_matrix(K.shape)   # mass unchanged
    >>> dlam = eigenvalue_sensitivity(K, M, phi, freqs, modes, dK_dp, dM_dp)
    >>> dfreq_hz = dlam / (2 * np.pi * freqs_hz[modes]) / (2 * np.pi)
    """
    Phi   = phi[:, selected_modes]
    lam   = (2.0 * np.pi * freqs_hz[selected_modes]) ** 2  # (m,)

    dK_phi = dK_dp @ Phi           # (N, m)
    dM_phi = dM_dp @ Phi           # (N, m)

    # ∂λᵢ/∂p = φᵢᵀ (∂K/∂p φᵢ - λᵢ ∂M/∂p φᵢ)
    dlam = np.einsum("ij,ij->j", Phi, dK_phi) - lam * np.einsum("ij,ij->j", Phi, dM_phi)

    if verbose:
        dfreq = dlam / (2.0 * np.pi * freqs_hz[selected_modes] + 1e-10) / (2.0 * np.pi)
        print(
            f"[Eigenvalue Sensitivity]  {len(selected_modes)} modes\n"
            f"  Max |∂f/∂p|: {np.abs(dfreq).max():.4e} Hz/unit  "
            f"  Max |∂λ/∂p|: {np.abs(dlam).max():.4e} (rad/s)²/unit"
        )
    return dlam


# ─────────────────────────────────────────────────────────────────────────────
# FRF sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def frf_sensitivity(
    Ka: np.ndarray,
    Ma: np.ndarray,
    dKa_dp: np.ndarray,
    dMa_dp: np.ndarray,
    local_force: List[int],
    local_output: List[int],
    freq_eval: np.ndarray,
    zeta: float = 0.001,
    damping_type: str = "modal",
) -> Dict[str, np.ndarray]:
    """
    Compute the FRF parameter sensitivity ∂H/∂p via direct differentiation.

    Differentiating Z(ω) H = I with respect to p gives:

        ∂H/∂p = −Z⁻¹ (∂Z/∂p) Z⁻¹

    where  ∂Z/∂p = ∂Kₐ/∂p − ω² ∂Mₐ/∂p  (for viscous damping proportional to Ka, Ma).

    Parameters
    ----------
    Ka, Ma : np.ndarray, shape (m, m) — reduced matrices
    dKa_dp, dMa_dp : np.ndarray, shape (m, m) — parameter derivatives of Ka, Ma
    local_force, local_output : list of int — local DOF indices within master set
    freq_eval : np.ndarray — evaluation frequencies (Hz)
    zeta, damping_type : damping parameters (same as compute_frf_direct)

    Returns
    -------
    dH_dp : dict — same key structure as :func:`compute_frf_direct`
        Values are complex arrays of shape (n_freq,) representing ∂H/∂p.

    Notes
    -----
    This gives the *first-order* sensitivity.  For a parameter change Δp,
    the FRF changes approximately by ΔH ≈ (∂H/∂p) Δp.
    """
    from pyserep.frf.direct_frf import _modal_ca, _rayleigh_ca

    m = Ka.shape[0]  # noqa: F841 — used in loop
    n_freq = len(freq_eval)

    if damping_type == "modal":
        Ca = _modal_ca(Ka, Ma, zeta)
    elif damping_type == "rayleigh":
        Ca = _rayleigh_ca(Ka, Ma, zeta)
    else:
        Ca = np.zeros_like(Ka)

    dH_dp: Dict[str, np.ndarray] = {
        f"f{fi}_o{oi}": np.zeros(n_freq, dtype=complex)
        for fi, oi in zip(local_force, local_output)
    }

    for k, f in enumerate(freq_eval):
        omega  = 2.0 * np.pi * f
        omega2 = omega ** 2

        Z  = Ka - omega2 * Ma + 1j * omega * Ca
        dZ = dKa_dp - omega2 * dMa_dp

        try:
            Zinv = np.linalg.inv(Z)
        except np.linalg.LinAlgError:
            Zinv = np.linalg.pinv(Z)

        dZ_Zinv = dZ @ Zinv           # (m, m)
        neg_Zinv_dZ_Zinv = -Zinv @ dZ_Zinv   # (m, m)

        for fi, oi in zip(local_force, local_output):
            key = f"f{fi}_o{oi}"
            dH_dp[key][k] = neg_Zinv_dZ_Zinv[oi, fi]

    return dH_dp


# ─────────────────────────────────────────────────────────────────────────────
# Material perturbation study
# ─────────────────────────────────────────────────────────────────────────────

def material_perturbation_study(
    Ka_nominal: np.ndarray,
    Ma_nominal: np.ndarray,
    local_force: List[int],
    local_output: List[int],
    freq_eval: np.ndarray,
    H_nominal: Dict[str, np.ndarray],
    param_values: List[float],
    Ka_func: Callable[[float], np.ndarray],
    Ma_func: Callable[[float], np.ndarray],
    zeta: float = 0.001,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute FRFs for a sweep of material parameter values.

    Useful for comparing Standard vs HyAPA material configurations, or
    for uncertainty quantification (±5% E, ±3% ρ, etc.).

    Parameters
    ----------
    Ka_nominal, Ma_nominal : np.ndarray — baseline reduced matrices
    local_force, local_output : list of int
    freq_eval : np.ndarray
    H_nominal : dict — baseline FRF (from compute_frf_direct)
    param_values : list of float — parameter values to evaluate
        e.g. [0.95, 0.97, 1.00, 1.03, 1.05] for ±5% sweep
    Ka_func : callable — Ka_func(p) → np.ndarray(m, m)
        Returns Ka for parameter value p.
        For stiffness scaling: ``lambda p: p * Ka_nominal``
    Ma_func : callable — Ma_func(p) → np.ndarray(m, m)
        For mass scaling: ``lambda p: p * Ma_nominal``
    zeta, verbose : see compute_frf_direct

    Returns
    -------
    dict with keys:
        ``"param_values"`` — np.ndarray of the swept values
        ``"H_sweep"`` — np.ndarray, shape (n_params, n_freq) — FRF magnitudes
        ``"max_deviation_pct"`` — np.ndarray, shape (n_params,)
    """
    from pyserep.frf.direct_frf import compute_frf_direct

    n_params = len(param_values)
    key_nom  = f"f{local_force[0]}_o{local_output[0]}"
    n_freq   = len(freq_eval)
    H_sweep  = np.zeros((n_params, n_freq))
    max_dev  = np.zeros(n_params)

    h_nom_mag = np.abs(H_nominal[key_nom])

    for i, p in enumerate(param_values):
        Ka_p = Ka_func(p)
        Ma_p = Ma_func(p)

        _, H_p = compute_frf_direct(
            Ka_p, Ma_p,
            force_dof_indices  = local_force,
            output_dof_indices = local_output,
            freq_eval          = freq_eval,
            zeta               = zeta,
            verbose            = False,
        )
        h_p_mag = np.abs(H_p[key_nom])
        H_sweep[i] = h_p_mag
        denom = np.where(h_nom_mag > 1e-30, h_nom_mag, 1e-30)
        max_dev[i] = float(np.abs(h_p_mag - h_nom_mag).max() / denom.max() * 100.0)

        if verbose:
            print(f"  p = {p:.4f}  max dev = {max_dev[i]:.4f}%")

    return {
        "param_values"      : np.array(param_values),
        "H_sweep"           : H_sweep,
        "max_deviation_pct" : max_dev,
        "freq_eval"         : freq_eval,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty quantification
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_frf(
    Ka_nominal: np.ndarray,
    Ma_nominal: np.ndarray,
    local_force: List[int],
    local_output: List[int],
    freq_eval: np.ndarray,
    E_cov_pct: float = 2.0,
    rho_cov_pct: float = 1.0,
    n_samples: int = 50,
    seed: int = 42,
    zeta: float = 0.001,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo FRF uncertainty quantification.

    Perturbs Ka (proportional to E uncertainty) and Ma (proportional to
    ρ uncertainty) and computes the FRF for each sample, returning the
    mean, standard deviation, and 5th/95th percentile bands.

    Parameters
    ----------
    Ka_nominal, Ma_nominal : np.ndarray, shape (m, m)
    local_force, local_output : list of int
    freq_eval : np.ndarray
    E_cov_pct : float — coefficient of variation for Young's modulus (%)
    rho_cov_pct : float — coefficient of variation for density (%)
    n_samples : int
    seed : int
    zeta, verbose

    Returns
    -------
    dict with keys:
        ``"H_mean"`` — mean FRF magnitude
        ``"H_std"``  — standard deviation
        ``"H_p5"``   — 5th percentile
        ``"H_p95"``  — 95th percentile
        ``"H_all"``  — all samples, shape (n_samples, n_freq)
    """
    from pyserep.frf.direct_frf import compute_frf_direct

    rng     = np.random.default_rng(seed)
    key     = f"f{local_force[0]}_o{local_output[0]}"
    n_freq  = len(freq_eval)
    H_all   = np.zeros((n_samples, n_freq))

    if verbose:
        print(
            f"[Monte Carlo FRF]  {n_samples} samples  "
            f"σ_E={E_cov_pct:.1f}%  σ_ρ={rho_cov_pct:.1f}%"
        )

    for i in range(n_samples):
        alpha_E   = 1.0 + rng.normal(0, E_cov_pct / 100.0)
        alpha_rho = 1.0 + rng.normal(0, rho_cov_pct / 100.0)
        Ka_s = np.clip(alpha_E,   0.5, 2.0) * Ka_nominal
        Ma_s = np.clip(alpha_rho, 0.5, 2.0) * Ma_nominal

        _, H_s = compute_frf_direct(
            Ka_s, Ma_s,
            force_dof_indices  = local_force,
            output_dof_indices = local_output,
            freq_eval          = freq_eval,
            zeta               = zeta,
            verbose            = False,
        )
        H_all[i] = np.abs(H_s[key])

        if verbose and i % max(1, n_samples // 10) == 0:
            print(f"  [{i/n_samples*100:.0f}%] sample {i+1}/{n_samples}", end="\r")

    if verbose:
        print(" " * 60, end="\r")

    return {
        "H_mean"  : H_all.mean(axis=0),
        "H_std"   : H_all.std(axis=0),
        "H_p5"    : np.percentile(H_all, 5,  axis=0),
        "H_p95"   : np.percentile(H_all, 95, axis=0),
        "H_all"   : H_all,
        "freq_eval": freq_eval,
    }
