"""
pyserep.frf.modal_frf
========================
Modal superposition FRF computation.

This module provides FRF computation via the classical modal expansion
formula.  It is primarily used to generate the *reference* FRF (using
all elastic modes of the full model), against which the ROM's direct FRF
is compared.

Theory
------
For a damped MDOF system with mass-normalised modes:

    H(f,o,ω) = Σᵢ  φᵢ(f) · φᵢ(o)
                    ─────────────────────────────
                    ωᵢ² − ω² + 2j ζᵢ ωᵢ ω

where φᵢ(·) is the mode shape component at the force/output DOF and
ωᵢ is the i-th natural angular frequency.

Note: This formula is approximate because it neglects the contribution
of out-of-band modes (residual effects).  The direct method in
``direct_frf.py`` avoids this truncation by using the physical matrices.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from pyserep.selection.band_selector import FrequencyBandSet


def compute_frf_modal(
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    mode_indices: np.ndarray,
    force_dofs: List[int],
    output_dofs: List[int],
    band_set: FrequencyBandSet,
    zeta: float = 0.001,
    per_mode_zeta: np.ndarray | None = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute FRF via modal superposition, evaluated only within band regions.

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_modes)
        Full modal matrix (mass-normalised).
    freqs_hz : np.ndarray, shape (n_modes,)
        Natural frequencies in Hz.
    mode_indices : np.ndarray of int
        Indices of modes to include in the superposition.
    force_dofs : list of int
        Force DOF global indices.
    output_dofs : list of int
        Output DOF global indices.
    band_set : FrequencyBandSet
        Defines the frequency evaluation grid.
    zeta : float
        Uniform damping ratio (used when ``per_mode_zeta`` is None).
    per_mode_zeta : np.ndarray, optional
        Per-mode damping ratios, shape (n_modes,).  Overrides *zeta*.
    verbose : bool

    Returns
    -------
    freq_eval : np.ndarray
        Evaluation frequencies (Hz).
    H : dict
        FRF arrays keyed by ``"f{fi}_o{oi}"``.
    """
    freq_eval = band_set.frequency_grid()
    omega_eval = 2.0 * np.pi * freq_eval          # (n_freq,)

    omega_n = 2.0 * np.pi * freqs_hz[mode_indices]  # (m,)
    Phi     = phi[:, mode_indices]                   # (N, m)

    if per_mode_zeta is None:
        zeta_vec = np.full(len(mode_indices), zeta)
    else:
        zeta_vec = per_mode_zeta[mode_indices]

    if verbose:
        print(
            f"[Modal FRF]  {len(mode_indices)} modes  |  "
            f"{len(freq_eval)} freq points  |  {band_set.n_bands} band(s)"
        )

    H: Dict[str, np.ndarray] = {}

    for fi, oi in zip(force_dofs, output_dofs):
        key    = f"f{fi}_o{oi}"
        phi_f  = Phi[fi, :]     # (m,)
        phi_o  = Phi[oi, :]     # (m,)
        nums   = phi_f * phi_o  # (m,) numerators

        # Denominator matrix D[i, k] = ωᵢ² − ωₖ² + 2j ζᵢ ωᵢ ωₖ
        D = (
            omega_n[:, None] ** 2
            - omega_eval[None, :] ** 2
            + 2j * zeta_vec[:, None] * omega_n[:, None] * omega_eval[None, :]
        )  # (m, n_freq)

        H[key] = np.sum(nums[:, None] / D, axis=0)   # (n_freq,)

    return freq_eval, H


def compute_frf_modal_reference(
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    rb_hz: float,
    force_dofs: List[int],
    output_dofs: List[int],
    band_set: FrequencyBandSet,
    zeta: float = 0.001,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute reference FRF using ALL elastic modes of the full model.

    This is used as the ground truth for FRF accuracy assessment.

    Parameters
    ----------
    phi, freqs_hz : full modal matrix and frequencies
    rb_hz : float
        Rigid-body exclusion threshold in Hz.
    force_dofs, output_dofs : list of int
    band_set : FrequencyBandSet
    zeta : float
    verbose : bool

    Returns
    -------
    H_ref : dict
    """
    elastic_idx = np.where(freqs_hz > rb_hz)[0]
    if verbose:
        print(
            f"[Modal FRF — Reference]  {len(elastic_idx)} elastic modes  "
            f"(f > {rb_hz} Hz)"
        )
    _, H_ref = compute_frf_modal(
        phi, freqs_hz, elastic_idx,
        force_dofs, output_dofs, band_set,
        zeta=zeta, verbose=False,
    )
    return H_ref
