"""
pyserep.pipeline.serep_pipeline
===================================
Main SEREP ROM pipeline orchestrator.

Step sequence
-------------
1.  Load K, M matrices
2.  Solve generalised eigenproblem
3.  Build FrequencyBandSet
4.  Mode selection: (MS1 + MAC) ∪ MS2 ∪ MS3
5.  DOF selection: chosen method (default: DS4 EID)
6.  Build SEREP ROM:  T, Kₐ, Mₐ
7.  Verify eigenvalues (SEREP exact-preservation check)
8.  Compute FRF — direct method (Kₐ, Mₐ) + modal reference
9.  Run validation suite
10. Collect performance metrics
11. Export all results
12. Generate plots
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from pyserep.analysis.performance import PerformanceMetrics, summarise_performance
from pyserep.analysis.validation import ValidationReport, validate_serep
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.core.rom_builder import build_serep_rom, verify_eigenvalues
from pyserep.frf.direct_frf import (
    FRFResult,
    compute_frf_direct,
)
from pyserep.frf.modal_frf import compute_frf_modal_reference
from pyserep.io.exporter import save_results
from pyserep.io.matrix_loader import load_matrices
from pyserep.pipeline.config import ROMConfig
from pyserep.selection.band_selector import FrequencyBandSet
from pyserep.selection.dof_selector import (
    select_dofs_eid,
    select_dofs_kinetic,
    select_dofs_modal_disp,
    select_dofs_svd,
)
from pyserep.selection.mode_selector import select_modes_pipeline
from pyserep.utils.timers import Timer

_DOF_SELECTOR_MAP = {
    "eid":         select_dofs_eid,
    "kinetic":     select_dofs_kinetic,
    "modal_disp":  select_dofs_modal_disp,
    "svd":         select_dofs_svd,
}


# ─────────────────────────────────────────────────────────────────────────────
# Results container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResults:
    """
    All outputs from a completed SEREP ROM pipeline run.

    Attributes
    ----------
    config : ROMConfig
    freqs_hz : np.ndarray
        All computed natural frequencies (Hz).
    phi : np.ndarray
        Full modal matrix (N × n_modes).
    selected_modes : np.ndarray of int
    master_dofs : np.ndarray of int
    T : np.ndarray
        SEREP transformation matrix (N × m).
    Ka, Ma : np.ndarray
        Reduced stiffness and mass matrices (m × m).
    kappa : float
        Condition number κ(Φₐ).
    freq_errors : np.ndarray
        Per-mode eigenvalue preservation errors (%).
    max_freq_err : float
    frf : FRFResult
    validation : ValidationReport
    performance : PerformanceMetrics
    elapsed_total_s : float
    saved_files : dict
    """

    config: ROMConfig

    freqs_hz:       np.ndarray = field(default_factory=lambda: np.array([]))
    phi:            np.ndarray = field(default_factory=lambda: np.array([[]]))
    selected_modes: np.ndarray = field(default_factory=lambda: np.array([], int))
    master_dofs:    np.ndarray = field(default_factory=lambda: np.array([], int))

    T:  Optional[np.ndarray] = None
    Ka: Optional[np.ndarray] = None
    Ma: Optional[np.ndarray] = None

    kappa:         float = float("inf")
    freq_errors:   Optional[np.ndarray] = None
    max_freq_err:  float = float("nan")

    frf:        Optional[FRFResult] = None
    validation: Optional[ValidationReport] = None
    performance: Optional[PerformanceMetrics] = None
    saved_files: Dict[str, str] = field(default_factory=dict)

    elapsed_total_s: float = 0.0

    def summary(self) -> str:
        """Return a formatted string with all key pipeline results and metrics."""
        N = self.phi.shape[0]
        m = len(self.selected_modes)
        a = len(self.master_dofs)
        lines = [
            "\n" + "=" * 58,
            "  PIPELINE RESULTS SUMMARY",
            "=" * 58,
            f"  Wall time         : {self.elapsed_total_s:.2f}s",
            f"  FRF method        : {self.config.frf_method}",
            f"  Full-model DOFs   : {N:,}",
            f"  Selected modes    : {m}",
            f"  Master DOFs       : {a}",
            f"  DOF retention     : {a/N*100:.4f}%",
            f"  κ(Φₐ)            : {self.kappa:.4e}",
            f"  Max freq error    : {self.max_freq_err:.8f}%",
        ]
        if self.frf:
            for key, errs in self.frf.errors.items():
                lines.append(
                    f"  FRF {key:15s}: max={errs['max_pct']:.4f}%  "
                    f"rms={errs['rms_pct']:.4f}%"
                )
        if self.performance:
            lines.append(f"  FLOP speedup      : {self.performance.frf_speedup:.1f}×")
        lines.append("=" * 58)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SereпPipeline:
    """
    Orchestrates the complete SEREP ROM pipeline.

    Parameters
    ----------
    config : ROMConfig

    Examples
    --------
    Full-range analysis with direct FRF:

    >>> cfg = ROMConfig(
    ...     stiffness_file="K.mtx",
    ...     mass_file="M.mtx",
    ...     force_dofs=[3000],
    ...     output_dofs=[3000],
    ...     freq_range=(0.1, 500.0),
    ...     frf_method="direct",
    ... )
    >>> results = SereпPipeline(cfg).run()

    Selective bands:

    >>> from pyserep import FrequencyBand
    >>> cfg = ROMConfig(
    ...     stiffness_file="K.mtx",
    ...     mass_file="M.mtx",
    ...     force_dofs=[3000],
    ...     output_dofs=[3000],
    ...     bands=[FrequencyBand(0, 100), FrequencyBand(400, 500)],
    ... )
    >>> results = SereпPipeline(cfg).run()
    """

    def __init__(self, config: ROMConfig) -> None:
        self.cfg     = config
        self._results = PipelineResults(config=config)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> PipelineResults:
        """Execute the full pipeline and return :class:`PipelineResults`."""
        t_start = time.perf_counter()
        cfg = self.cfg
        v   = cfg.verbose

        if v:
            self._print_header()

        os.makedirs(cfg.export_folder, exist_ok=True)

        timings: Dict[str, float] = {}

        # ── 1. Load matrices ──────────────────────────────────────────────────
        with Timer("Load matrices", verbose=v) as t:
            K, M = load_matrices(cfg.stiffness_file, cfg.mass_file, verbose=v)
        timings["load"] = t.elapsed
        N = K.shape[0]

        # ── 2. Eigenproblem ───────────────────────────────────────────────────
        with Timer("Eigensolver", verbose=v) as t:
            freqs_hz, phi = solve_eigenproblem(
                K, M,
                n_modes  = cfg.num_modes_eigsh,
                sigma    = cfg.eigsh_sigma,
                tol      = cfg.eigsh_tol,
                verbose  = v,
            )
        timings["eigensolver"] = t.elapsed
        self._results.freqs_hz = freqs_hz
        self._results.phi      = phi

        # ── 3. FrequencyBandSet ───────────────────────────────────────────────
        band_set = FrequencyBandSet(cfg.effective_bands, cfg.n_points_per_band)
        if v:
            print(f"\n{band_set.summary()}")

        # ── 4. Mode selection ─────────────────────────────────────────────────
        with Timer("Mode selection", verbose=v) as t:
            selected = select_modes_pipeline(
                phi, freqs_hz,
                force_dofs    = cfg.force_dofs,
                output_dofs   = cfg.output_dofs,
                band_set      = band_set,
                rb_hz         = cfg.rb_hz,
                ms1_alpha     = cfg.ms1_alpha,
                ms2_threshold = cfg.ms2_threshold,
                ms3_threshold = cfg.ms3_threshold,
                mac_threshold = cfg.mac_threshold,
                verbose       = v,
            )
        timings["mode_select"] = t.elapsed
        self._results.selected_modes = selected
        m = len(selected)

        if m == 0:
            raise RuntimeError(
                "Mode selection returned 0 modes.  "
                "Check band settings and thresholds."
            )

        # ── 5. DOF selection ──────────────────────────────────────────────────
        # Force/output DOFs must always be in the master set
        required = np.unique(np.array(cfg.force_dofs + cfg.output_dofs, dtype=int))
        dof_fn = _DOF_SELECTOR_MAP.get(cfg.dof_method, select_dofs_eid)
        with Timer(f"DOF selection ({cfg.dof_method.upper()})", verbose=v) as t:
            if cfg.dof_method == "eid":
                master_dofs, kappa = dof_fn(
                    phi, selected,
                    ke_prescreen_frac = cfg.ke_prescreen_frac,
                    required_dofs     = required,
                    verbose           = v,
                )
            else:
                master_dofs, kappa = dof_fn(phi, selected, verbose=v)
        timings["dof_select"] = t.elapsed
        self._results.master_dofs = master_dofs
        self._results.kappa       = kappa

        # ── 6. Build ROM ──────────────────────────────────────────────────────
        with Timer("ROM build", verbose=v) as t:
            T, Ka, Ma = build_serep_rom(K, M, phi, selected, master_dofs, verbose=v)
        timings["rom_build"] = t.elapsed
        self._results.T  = T
        self._results.Ka = Ka
        self._results.Ma = Ma

        # ── 7. Verify eigenvalues ─────────────────────────────────────────────
        freq_errors, max_err = verify_eigenvalues(Ka, Ma, freqs_hz, selected, verbose=v)
        self._results.freq_errors  = freq_errors
        self._results.max_freq_err = max_err

        # ── 8. FRF computation ────────────────────────────────────────────────
        freq_eval = band_set.frequency_grid()
        n_freq    = len(freq_eval)

        # Map global force/output DOFs → local indices within master_dofs
        dof_map      = {int(d): i for i, d in enumerate(master_dofs)}
        local_force  = [dof_map[d] for d in cfg.force_dofs  if d in dof_map]
        local_output = [dof_map[d] for d in cfg.output_dofs if d in dof_map]

        if len(local_force) != cfg.n_pairs:
            import warnings
            warnings.warn(
                "Some force/output DOFs are not in the master set.  "
                "FRF will only be computed for the intersection.",
                UserWarning,
            )

        with Timer("FRF — ROM (direct)", verbose=v) as t:
            if cfg.frf_method == "direct":
                _, H_rom = compute_frf_direct(
                    Ka, Ma,
                    force_dof_indices  = local_force,
                    output_dof_indices = local_output,
                    freq_eval          = freq_eval,
                    zeta               = cfg.zeta,
                    damping_type       = cfg.damping_type,
                    verbose            = v,
                )
            else:
                from pyserep.frf.modal_frf import compute_frf_modal
                _, H_rom = compute_frf_modal(
                    phi, freqs_hz, selected,
                    cfg.force_dofs, cfg.output_dofs, band_set,
                    zeta=cfg.zeta, verbose=v,
                )

        with Timer("FRF — reference (modal)", verbose=v):
            H_ref = compute_frf_modal_reference(
                phi, freqs_hz, cfg.rb_hz,
                cfg.force_dofs, cfg.output_dofs, band_set,
                zeta=cfg.zeta, verbose=v,
            )
        timings["frf"] = t.elapsed

        # Align H_ref keys to H_rom keys
        H_ref_aligned = {}
        for (fi_g, oi_g), (fi_l, oi_l) in zip(
            zip(cfg.force_dofs, cfg.output_dofs),
            zip(local_force, local_output),
        ):
            g_key = f"f{fi_g}_o{oi_g}"
            l_key = f"f{fi_l}_o{oi_l}"
            if g_key in H_ref and l_key in H_rom:
                H_ref_aligned[l_key] = H_ref[g_key]

        band_masks = {
            band.label: (freq_eval >= band.f_min) & (freq_eval <= band.f_max)
            for band in band_set.bands
        }
        self._results.frf = FRFResult(
            freqs_hz   = freq_eval,
            H_rom      = H_rom,
            H_ref      = H_ref_aligned,
            band_masks = band_masks,
            method     = cfg.frf_method,
        )

        # ── 9. Validation ─────────────────────────────────────────────────────
        val = validate_serep(
            K, M, phi, freqs_hz, selected, master_dofs, T, Ka, Ma, verbose=v
        )
        self._results.validation = val

        # ── 10. Performance metrics ───────────────────────────────────────────
        from pyserep.analysis.performance import flop_count
        self._results.performance = summarise_performance(
            n_full_dofs      = N,
            n_selected_modes = m,
            n_master_dofs    = len(master_dofs),
            n_all_modes      = len(freqs_hz),
            kappa            = kappa,
            n_freq           = n_freq,
            n_bands          = band_set.n_bands,
            n_pairs          = len(local_force),
            frf_method       = cfg.frf_method,
            frf_flops_rom    = flop_count(m, n_freq, len(local_force), cfg.frf_method),
            frf_flops_ref    = flop_count(len(freqs_hz), n_freq, cfg.n_pairs, "modal"),
            t_eigensolver_s  = timings.get("eigensolver", 0.0),
            t_mode_select_s  = timings.get("mode_select", 0.0),
            t_dof_select_s   = timings.get("dof_select", 0.0),
            t_rom_build_s    = timings.get("rom_build", 0.0),
            t_frf_s          = timings.get("frf", 0.0),
            t_total_s        = time.perf_counter() - t_start,
        )

        # ── 11. Export ────────────────────────────────────────────────────────
        self._results.saved_files = save_results(
            self._results, cfg.export_folder, cfg.save_prefix,
            save_matrices=cfg.save_matrices, verbose=v,
        )

        # ── 12. Plots ─────────────────────────────────────────────────────────
        if cfg.plot:
            self._generate_plots()

        self._results.elapsed_total_s = time.perf_counter() - t_start

        if v:
            print(self._results.summary())
            if self._results.performance:
                print(self._results.performance.summary())

        return self._results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_plots(self) -> None:
        from pyserep.visualization.frf_plots import plot_frf_comparison
        from pyserep.visualization.summary_plots import plot_performance_dashboard

        cfg = self.cfg
        frf_path   = os.path.join(cfg.export_folder, f"{cfg.save_prefix}_frf.png")
        dash_path  = os.path.join(cfg.export_folder, f"{cfg.save_prefix}_dashboard.png")

        try:
            plot_frf_comparison(self._results, save_path=frf_path)
        except Exception as exc:
            print(f"[plot] FRF plot failed: {exc}")

        try:
            plot_performance_dashboard(self._results, save_path=dash_path)
        except Exception as exc:
            print(f"[plot] Dashboard plot failed: {exc}")

    def _print_header(self) -> None:
        print(
            "\n" + "=" * 58 + "\n"
            "  pyserep v3.0  —  SEREP ROM Pipeline\n"
            "  Direct FRF | Selective Bands | DS1–DS4\n"
            + "=" * 58 + "\n"
            + self.cfg.summary() + "\n"
        )
