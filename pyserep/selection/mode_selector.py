"""
pyserep.selection.mode_selector
===================================
Band-aware mode selection pipeline for SEREP.

Pipeline
--------
S_final = (MS1 filtered by MAC) ∪ MS2 ∪ MS3

+------+-----------------------------------------+-------------------------------+
| Step | Name                                    | Purpose                       |
+======+=========================================+===============================+
| MS1  | Frequency range filter                  | Coarse band relevance cut     |
+------+-----------------------------------------+-------------------------------+
| MAC  | Modal Assurance Criterion filter        | Remove spatially redundant    |
+------+-----------------------------------------+-------------------------------+
| MS2  | Band-weighted Modal Participation Factor| Capture participation tails   |
+------+-----------------------------------------+-------------------------------+
| MS3  | Spatial amplitude at target DOFs        | Catch near-nodal modes        |
+------+-----------------------------------------+-------------------------------+
| MS4  | Conditioning check                      | Verify κ(Φₐ) after DS4       |
+------+-----------------------------------------+-------------------------------+
"""

from __future__ import annotations

from typing import List, Set

import numpy as np

from pyserep.selection.band_selector import FrequencyBandSet


def select_modes(
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    force_dofs: List[int],
    output_dofs: List[int],
    band_set: "FrequencyBandSet | None" = None,
    f_max: float = 500.0,
    f_min: float = 0.1,
    rb_hz: float = 1.0,
    ms1_alpha: float = 1.5,
    ms2_threshold: float = 1.0,
    ms3_threshold: float = 5.0,
    mac_threshold: float = 0.90,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run the complete mode selection pipeline.

    Convenience wrapper that accepts either a :class:`FrequencyBandSet` or
    plain ``f_min`` / ``f_max`` scalars for single-band analysis.

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_modes)
        Mass-normalised modal matrix.
    freqs_hz : np.ndarray, shape (n_modes,)
        Natural frequencies in Hz.
    force_dofs : list of int
        Force DOF global indices.
    output_dofs : list of int
        Output DOF global indices.
    band_set : FrequencyBandSet, optional
        Multi-band specification.  If None, a single band
        ``[f_min, f_max]`` is created.
    f_max : float
        Upper frequency limit (Hz) when *band_set* is None.
    f_min : float
        Lower frequency limit (Hz) when *band_set* is None.
    rb_hz : float
        Rigid-body exclusion threshold (Hz).
    ms1_alpha : float
        MS1 safety factor: retains modes up to ``alpha × max(f_max)``.
    ms2_threshold : float
        MS2 band-weighted MPF threshold (% of dominant mode).
    ms3_threshold : float
        MS3 spatial amplitude threshold (% of global peak).
    mac_threshold : float
        MAC duplicate removal threshold.
    verbose : bool

    Returns
    -------
    np.ndarray of int
        Selected mode indices, sorted ascending.
    """
    if band_set is None:
        from pyserep.selection.band_selector import FrequencyBand
        band_set = FrequencyBandSet(
            [FrequencyBand(f_min, f_max, label="FullRange")]
        )

    return select_modes_pipeline(
        phi, freqs_hz, force_dofs, output_dofs, band_set,
        rb_hz=rb_hz, ms1_alpha=ms1_alpha,
        ms2_threshold=ms2_threshold, ms3_threshold=ms3_threshold,
        mac_threshold=mac_threshold, verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MS1 — Frequency range filter
# ─────────────────────────────────────────────────────────────────────────────

def ms1_frequency_range(
    freqs_hz: np.ndarray,
    band_set: FrequencyBandSet,
    rb_hz: float = 1.0,
    ms1_alpha: float = 1.5,
    verbose: bool = True,
) -> np.ndarray:
    """
    MS1: Retain modes whose natural frequency is relevant to at least one band.

    A mode at frequency f passes MS1 if:
      - f > rb_hz  (not rigid-body)
      - f ≤ alpha × max(band.f_max)  for at least one band

    Parameters
    ----------
    freqs_hz : np.ndarray, shape (n_modes,)
    band_set : FrequencyBandSet
    rb_hz : float
    ms1_alpha : float
    verbose : bool

    Returns
    -------
    np.ndarray of int
    """
    passed = [
        i for i, f in enumerate(freqs_hz)
        if band_set.mode_passes_ms1(f, rb_hz=rb_hz, alpha=ms1_alpha)
    ]
    result = np.array(passed, dtype=int)
    if verbose:
        cutoff = ms1_alpha * band_set.global_f_max
        print(
            f"[MS1] {len(result):>4} modes  "
            f"(f > {rb_hz} Hz, f ≤ {ms1_alpha}×{band_set.global_f_max:.0f} = {cutoff:.0f} Hz)"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAC filter — remove spatially redundant modes
# ─────────────────────────────────────────────────────────────────────────────

def mac_filter(
    phi: np.ndarray,
    candidate_indices: np.ndarray,
    freqs_hz: np.ndarray,
    force_dofs: List[int],
    output_dofs: List[int],
    mac_threshold: float = 0.90,
    verbose: bool = True,
) -> np.ndarray:
    """
    Remove spatially redundant modes from *candidate_indices*.

    Two modes are considered duplicates if their MAC exceeds *mac_threshold*.
    The mode with the lower participation score is discarded.

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_all_modes)
    candidate_indices : np.ndarray of int
    freqs_hz : np.ndarray
    force_dofs, output_dofs : list of int
    mac_threshold : float
    verbose : bool

    Returns
    -------
    np.ndarray of int  — filtered subset of *candidate_indices*
    """
    if len(candidate_indices) == 0:
        return candidate_indices

    phi_c  = phi[:, candidate_indices]
    n_cand = phi_c.shape[1]

    # Participation score for tiebreaking
    omega2 = np.maximum((2.0 * np.pi * freqs_hz[candidate_indices]) ** 2, 1e-10)
    scores = np.zeros(n_cand)
    for fi, oi in zip(force_dofs, output_dofs):
        scores += np.abs(phi_c[fi, :] * phi_c[oi, :]) / omega2

    # Normalised mode shapes for MAC
    norms = np.linalg.norm(phi_c, axis=0)
    norms = np.where(norms > 1e-15, norms, 1.0)
    phi_n = phi_c / norms[None, :]

    # Full MAC matrix
    gram       = phi_n.T @ phi_n         # (n_cand, n_cand)
    mac_matrix = gram ** 2

    # Greedy: process in descending score order, keep if not duplicate
    order   = np.argsort(scores)[::-1]
    kept: Set[int] = set()
    removed = 0

    for idx in order:
        duplicate = any(mac_matrix[idx, k] > mac_threshold for k in kept)
        if duplicate:
            removed += 1
        else:
            kept.add(idx)

    result = candidate_indices[sorted(kept)]
    if verbose:
        print(
            f"[MAC] {removed:>3} redundant modes removed "
            f"(MAC > {mac_threshold:.2f}).  {len(result)} remain."
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MS2 — Band-weighted Modal Participation Factor
# ─────────────────────────────────────────────────────────────────────────────

def ms2_participation_factor(
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    force_dofs: List[int],
    output_dofs: List[int],
    band_set: FrequencyBandSet,
    rb_hz: float = 1.0,
    threshold_pct: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    MS2: Retain modes with significant band-weighted Modal Participation Factor.

    For each band B and DOF pair (f, o), a mode passes MS2 if:
        C_i_B / C_dom_B  ≥  threshold_pct / 100

    where C_dom_B is the dominant elastic mode's MPF within band B.

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_modes)
    freqs_hz : np.ndarray
    force_dofs, output_dofs : list of int
    band_set : FrequencyBandSet
    rb_hz : float
    threshold_pct : float
    verbose : bool

    Returns
    -------
    np.ndarray of int
    """
    omega_n      = 2.0 * np.pi * freqs_hz
    elastic_mask = freqs_hz > rb_hz
    passed: Set[int] = set()

    for band in band_set.bands:
        for fi, oi in zip(force_dofs, output_dofs):
            C = band_set.band_weighted_mpf(phi[fi, :], phi[oi, :], omega_n, band)
            C_dom = C[elastic_mask].max() if elastic_mask.any() and C[elastic_mask].max() > 0 else None  # noqa: E501
            if C_dom is None:
                continue
            C_norm = C / C_dom * 100.0
            for i in np.where((freqs_hz > rb_hz) & (C_norm >= threshold_pct))[0]:
                passed.add(int(i))

    result = np.array(sorted(passed), dtype=int)
    if verbose:
        print(
            f"[MS2] {len(result):>4} modes  "
            f"(band-weighted MPF ≥ {threshold_pct:.1f}% of dominant, "
            f"{band_set.n_bands} band(s))"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MS3 — Spatial amplitude at target DOFs
# ─────────────────────────────────────────────────────────────────────────────

def ms3_spatial_amplitude(
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    target_dofs: List[int],
    rb_hz: float = 1.0,
    threshold_pct: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    MS3: Retain modes with significant spatial amplitude at target DOFs.

    A mode i passes MS3 if, at any target DOF d:
        |φᵢ(d)| / max_j(|φᵢ(j)|)  ≥  threshold_pct / 100

    This catches modes near nodal lines that MS2 might miss.

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_modes)
    freqs_hz : np.ndarray
    target_dofs : list of int
        Union of force and output DOFs.
    rb_hz : float
    threshold_pct : float
    verbose : bool

    Returns
    -------
    np.ndarray of int
    """
    unique_dofs = list(set(target_dofs))
    passed: Set[int] = set()

    for i in range(phi.shape[1]):
        if freqs_hz[i] <= rb_hz:
            continue
        global_peak = np.abs(phi[:, i]).max()
        if global_peak < 1e-15:
            continue
        for dof in unique_dofs:
            if np.abs(phi[dof, i]) / global_peak * 100.0 >= threshold_pct:
                passed.add(i)
                break

    result = np.array(sorted(passed), dtype=int)
    if verbose:
        print(
            f"[MS3] {len(result):>4} modes  "
            f"(amplitude ≥ {threshold_pct:.1f}% of global peak)"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MS4 — Conditioning check
# ─────────────────────────────────────────────────────────────────────────────

def ms4_conditioning_check(
    phi: np.ndarray,
    mode_indices: np.ndarray,
    master_dofs: np.ndarray,
    verbose: bool = True,
) -> float:
    """
    MS4: Compute κ(Φₐ) to verify the master DOF set is well conditioned.

    Parameters
    ----------
    phi : np.ndarray
    mode_indices, master_dofs : np.ndarray of int
    verbose : bool

    Returns
    -------
    float — condition number κ(Φₐ)
    """
    phi_a  = phi[np.ix_(master_dofs, mode_indices)]
    kappa  = float(np.linalg.cond(phi_a))
    rank   = int(np.linalg.matrix_rank(phi_a))
    m      = len(mode_indices)

    if verbose:
        label = ("EXCELLENT" if kappa < 1e2 else
                 "GOOD"      if kappa < 1e3 else
                 "MARGINAL"  if kappa < 1e6 else "POOR")
        print(
            f"[MS4] κ(Φₐ) = {kappa:.4e}  [{label}]  "
            f"rank = {rank}/{m}"
        )
    return kappa


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def select_modes_pipeline(
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    force_dofs: List[int],
    output_dofs: List[int],
    band_set: FrequencyBandSet,
    rb_hz: float = 1.0,
    ms1_alpha: float = 1.5,
    ms2_threshold: float = 1.0,
    ms3_threshold: float = 5.0,
    mac_threshold: float = 0.90,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run the complete mode selection pipeline.

    S_final = (MS1 filtered by MAC) ∪ MS2 ∪ MS3

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_modes)
    freqs_hz : np.ndarray, shape (n_modes,)
    force_dofs, output_dofs : list of int
    band_set : FrequencyBandSet
    rb_hz : float
    ms1_alpha, ms2_threshold, ms3_threshold, mac_threshold : float
    verbose : bool

    Returns
    -------
    np.ndarray of int — final selected mode indices, sorted ascending
    """
    if verbose:
        print(
            f"\n{'─'*55}\n  MODE SELECTION PIPELINE\n{'─'*55}\n"
            f"  Input: {len(freqs_hz)} modes  |  {band_set}\n"
        )

    s1 = ms1_frequency_range(
        freqs_hz, band_set, rb_hz=rb_hz, ms1_alpha=ms1_alpha, verbose=verbose
    )
    s1_mac = mac_filter(
        phi, s1, freqs_hz, force_dofs, output_dofs,
        mac_threshold=mac_threshold, verbose=verbose,
    )
    s2 = ms2_participation_factor(
        phi, freqs_hz, force_dofs, output_dofs, band_set,
        rb_hz=rb_hz, threshold_pct=ms2_threshold, verbose=verbose,
    )
    s3 = ms3_spatial_amplitude(
        phi, freqs_hz, list(set(force_dofs + output_dofs)),
        rb_hz=rb_hz, threshold_pct=ms3_threshold, verbose=verbose,
    )

    union   = sorted(set(s1_mac.tolist()) | set(s2.tolist()) | set(s3.tolist()))
    selected = np.array(union, dtype=int)

    if verbose:
        freq_sel = freqs_hz[selected]
        print(
            f"\n[Pipeline] Final: {len(selected)} modes selected\n"
            f"  |S1∩MAC| = {len(s1_mac)}  |S2| = {len(s2)}  |S3| = {len(s3)}\n"
            f"  Frequency range: {freq_sel.min():.3f} – {freq_sel.max():.2f} Hz\n"
        )

    return selected
