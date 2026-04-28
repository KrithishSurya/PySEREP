"""
pyserep.analysis.convergence
================================
Convergence studies for SEREP ROM accuracy.

Use these tools to answer:
  1. How does FRF accuracy change as more modes are retained?
  2. What is the minimum number of modes for a given accuracy target?
  3. How does the DOF-to-mode ratio affect condition number?

Functions
---------
mode_count_study        — sweep n_modes, report FRF error at each step
bandwidth_sensitivity   — vary f_max and report retained modes + error
dof_count_study         — vary n_master, track κ and FRF error
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class ConvergencePoint:
    """Single point in a convergence sweep."""
    param_value:    float
    n_modes:        int
    n_dofs:         int
    kappa:          float
    max_frf_err_pct: float
    rms_frf_err_pct: float
    max_freq_err_pct: float


@dataclass
class ConvergenceStudy:
    """Results from a full convergence sweep."""
    param_name:  str
    param_label: str
    points:      List[ConvergencePoint]

    def table(self) -> str:
        """Return the convergence results as a formatted ASCII table string."""
        hdr = (f"{'':>6}  {'Modes':>6}  {'DOFs':>6}  "
               f"{'κ(Φₐ)':>12}  {'FRF max%':>10}  {'FRF rms%':>10}  {'Freq err%':>10}")
        sep = "─" * len(hdr)
        rows = [f"\n{self.param_label} Convergence\n{sep}\n{hdr}\n{sep}"]
        for p in self.points:
            rows.append(
                f"{p.param_value:>6.1f}  {p.n_modes:>6}  {p.n_dofs:>6}  "
                f"{p.kappa:>12.4e}  {p.max_frf_err_pct:>10.6f}  "
                f"{p.rms_frf_err_pct:>10.6f}  {p.max_freq_err_pct:>10.8f}"
            )
        rows.append(sep)
        return "\n".join(rows)

    def plot(self, save_path: Optional[str] = None, show: bool = False) -> None:
        """Plot FRF error and condition number vs parameter."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        vals    = [p.param_value for p in self.points]
        frf_max = [p.max_frf_err_pct for p in self.points]
        frf_rms = [p.rms_frf_err_pct for p in self.points]
        kappas  = [p.kappa for p in self.points]
        n_modes = [p.n_modes for p in self.points]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{self.param_label} Convergence Study", fontweight="bold")

        axes[0].semilogy(vals, frf_max, "o-", color="coral",   lw=1.5, label="max%")
        axes[0].semilogy(vals, frf_rms, "s--", color="navy",   lw=1.5, label="rms%")
        axes[0].set_xlabel(self.param_label)
        axes[0].set_ylabel("FRF Error (%)")
        axes[0].set_title("FRF Error")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(vals, kappas, "D-", color="steelblue", lw=1.5)
        axes[1].set_xlabel(self.param_label)
        axes[1].set_ylabel("κ(Φₐ)")
        axes[1].set_title("Condition Number")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(vals, n_modes, "^-", color="seagreen", lw=1.5)
        axes[2].set_xlabel(self.param_label)
        axes[2].set_ylabel("Retained modes")
        axes[2].set_title("Mode Count")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[convergence] Plot saved: {save_path}")
        if show:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Study 1: Mode count sweep (vary MS1 alpha or f_max cutoff)
# ─────────────────────────────────────────────────────────────────────────────

def mode_count_study(
    K: sp.csc_matrix,
    M: sp.csc_matrix,
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    force_dofs: List[int],
    output_dofs: List[int],
    f_max: float,
    f_max_values: List[float],
    zeta: float = 0.001,
    rb_hz: float = 1.0,
    n_freq: int = 500,
    verbose: bool = True,
) -> ConvergenceStudy:
    """
    Study FRF convergence as the upper frequency cutoff is varied.

    For each value in *f_max_values*, runs a full SEREP pipeline
    (mode selection → DOF selection → ROM build → direct FRF) and
    records accuracy metrics.

    Parameters
    ----------
    K, M : sparse matrices
    phi, freqs_hz : modal matrix and frequencies (pre-computed)
    force_dofs, output_dofs : list of int
    f_max : float
        Upper limit of the reference FRF (Hz).
    f_max_values : list of float
        Cutoff frequencies to sweep (Hz). Each defines the upper edge of
        a single analysis band.
    zeta : float
    rb_hz : float
    n_freq : int
        Number of FRF evaluation points.
    verbose : bool

    Returns
    -------
    ConvergenceStudy
    """
    from pyserep.core.rom_builder import build_serep_rom, verify_eigenvalues
    from pyserep.frf.direct_frf import compute_frf_direct
    from pyserep.frf.modal_frf import compute_frf_modal_reference
    from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
    from pyserep.selection.dof_selector import select_dofs_eid
    from pyserep.selection.mode_selector import select_modes_pipeline

    points: List[ConvergencePoint] = []
    ref_band = FrequencyBandSet([FrequencyBand(rb_hz, f_max)], n_points_per_band=n_freq)
    freq_eval = ref_band.frequency_grid()

    # Reference FRF (all elastic modes, fixed grid)
    elastic = np.where(freqs_hz > rb_hz)[0]  # noqa: F841 — used below
    H_ref = compute_frf_modal_reference(
        phi, freqs_hz, rb_hz, force_dofs, output_dofs, ref_band,
        zeta=zeta, verbose=False,
    )

    for cutoff in f_max_values:
        if verbose:
            print(f"  [convergence] f_max = {cutoff:.1f} Hz …", end=" ", flush=True)

        band_set = FrequencyBandSet([FrequencyBand(rb_hz, cutoff)], n_points_per_band=200)

        try:
            selected = select_modes_pipeline(
                phi, freqs_hz, force_dofs, output_dofs, band_set,
                rb_hz=rb_hz, verbose=False,
            )
            if len(selected) == 0:
                if verbose:
                    print("0 modes — skip")
                continue

            dofs, kappa = select_dofs_eid(phi, selected, verbose=False)
            T, Ka, Ma   = build_serep_rom(K, M, phi, selected, dofs, verbose=False)
            _, max_feq  = verify_eigenvalues(Ka, Ma, freqs_hz, selected, verbose=False)

            # Direct FRF on reference grid
            dof_map = {int(d): i for i, d in enumerate(dofs)}
            lf = [dof_map[d] for d in force_dofs if d in dof_map]
            lo = [dof_map[d] for d in output_dofs if d in dof_map]
            if not lf:
                if verbose:
                    print("DOF not in master — skip")
                continue

            _, H_rom = compute_frf_direct(
                Ka, Ma, lf, lo, freq_eval, zeta=zeta, verbose=False
            )

            # Error vs reference
            key_rom = f"f{lf[0]}_o{lo[0]}"
            key_ref = f"f{force_dofs[0]}_o{output_dofs[0]}"
            h_r = np.abs(H_rom.get(key_rom, np.zeros(len(freq_eval))))
            h_f = np.abs(H_ref.get(key_ref, np.ones(len(freq_eval))))
            denom = np.where(h_f > 1e-30, h_f, 1e-30)
            err_pct = np.abs(h_r - h_f) / denom * 100.0

            pt = ConvergencePoint(
                param_value     = float(cutoff),
                n_modes         = len(selected),
                n_dofs          = len(dofs),
                kappa           = kappa,
                max_frf_err_pct = float(err_pct.max()),
                rms_frf_err_pct = float(np.sqrt(np.mean(err_pct**2))),
                max_freq_err_pct = float(max_feq) if not np.isnan(max_feq) else np.nan,
            )
            points.append(pt)
            if verbose:
                print(f"{len(selected)} modes  κ={kappa:.2e}  FRF_max={pt.max_frf_err_pct:.4f}%")

        except Exception as exc:
            if verbose:
                print(f"ERROR: {exc}")
            continue

    return ConvergenceStudy(
        param_name  = "f_max",
        param_label = "Upper Frequency Cutoff (Hz)",
        points      = points,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Study 2: DOF count sweep (vary n_master at fixed modes)
# ─────────────────────────────────────────────────────────────────────────────

def dof_count_study(
    K: sp.csc_matrix,
    M: sp.csc_matrix,
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    selected_modes: np.ndarray,
    force_dofs: List[int],
    output_dofs: List[int],
    n_master_values: List[int],
    freq_eval: np.ndarray,
    H_ref: dict,
    zeta: float = 0.001,
    verbose: bool = True,
) -> ConvergenceStudy:
    """
    Study how FRF accuracy and condition number change with n_master.

    Fixes the mode set and varies the number of master DOFs from
    ``len(selected_modes)`` up to ``max(n_master_values)``.

    Parameters
    ----------
    K, M, phi, freqs_hz : full model
    selected_modes : np.ndarray of int — fixed mode set
    force_dofs, output_dofs : list of int
    n_master_values : list of int — DOF counts to sweep
    freq_eval : np.ndarray — frequency grid for FRF
    H_ref : dict — reference FRF on *freq_eval*
    zeta : float
    verbose : bool

    Returns
    -------
    ConvergenceStudy
    """
    from pyserep.core.rom_builder import build_serep_rom
    from pyserep.frf.direct_frf import compute_frf_direct
    from pyserep.selection.dof_selector import select_dofs_eid

    m = len(selected_modes)
    points: List[ConvergencePoint] = []

    for n_m in n_master_values:
        if n_m < m:
            if verbose:
                print(f"  [dof sweep] n_master={n_m} < m={m} — skip")
            continue

        if verbose:
            print(f"  [dof sweep] n_master = {n_m} …", end=" ", flush=True)

        try:
            dofs, kappa = select_dofs_eid(phi, selected_modes, n_master=n_m, verbose=False)
            T, Ka, Ma   = build_serep_rom(K, M, phi, selected_modes, dofs, verbose=False)

            dof_map = {int(d): i for i, d in enumerate(dofs)}
            lf = [dof_map[d] for d in force_dofs if d in dof_map]
            lo = [dof_map[d] for d in output_dofs if d in dof_map]
            if not lf:
                continue

            _, H_rom = compute_frf_direct(Ka, Ma, lf, lo, freq_eval, zeta=zeta, verbose=False)

            key_rom = f"f{lf[0]}_o{lo[0]}"
            key_ref = f"f{force_dofs[0]}_o{output_dofs[0]}"
            h_r = np.abs(H_rom.get(key_rom, np.zeros(len(freq_eval))))
            h_f = np.abs(H_ref.get(key_ref, np.ones(len(freq_eval))))
            denom = np.where(h_f > 1e-30, h_f, 1e-30)
            err_pct = np.abs(h_r - h_f) / denom * 100.0

            pt = ConvergencePoint(
                param_value     = float(n_m),
                n_modes         = m,
                n_dofs          = n_m,
                kappa           = kappa,
                max_frf_err_pct = float(err_pct.max()),
                rms_frf_err_pct = float(np.sqrt(np.mean(err_pct**2))),
                max_freq_err_pct = np.nan,
            )
            points.append(pt)
            if verbose:
                print(f"κ={kappa:.2e}  FRF_max={pt.max_frf_err_pct:.4f}%")

        except Exception as exc:
            if verbose:
                print(f"ERROR: {exc}")
            continue

    return ConvergenceStudy(
        param_name  = "n_master",
        param_label = "Number of Master DOFs",
        points      = points,
    )
