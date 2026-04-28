"""
pyserep.analysis.performance
================================
Performance metrics: FLOP counts, timing, reduction ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PerformanceMetrics:
    """All performance and efficiency metrics for a pipeline run."""

    # Size
    n_full_dofs:      int
    n_selected_modes: int
    n_master_dofs:    int

    # Reduction
    dof_reduction_pct:  float    # = n_master / n_full × 100
    mode_retention_pct: float    # = n_selected / n_total × 100

    # Condition
    kappa:  float

    # FRF
    frf_method:       str
    n_freq_points:    int
    n_bands:          int
    frf_flops_rom:    int
    frf_flops_ref:    int
    frf_speedup:      float      # ref_flops / rom_flops

    # Timing
    t_eigensolver_s:   float
    t_mode_select_s:   float
    t_dof_select_s:    float
    t_rom_build_s:     float
    t_frf_s:           float
    t_total_s:         float

    def summary(self) -> str:
        """Return a formatted multi-line performance summary string."""
        return "\n".join([
            f"\n{'─'*55}",
            "  PERFORMANCE SUMMARY",
            f"{'─'*55}",
            f"  Full DOFs          : {self.n_full_dofs:>12,}",
            f"  Master DOFs        : {self.n_master_dofs:>12,}",
            f"  DOF reduction      : {self.dof_reduction_pct:>11.4f}%",
            f"  Modes retained     : {self.n_selected_modes:>12,}",
            f"  Mode retention     : {self.mode_retention_pct:>11.4f}%",
            f"  κ(Φₐ)             : {self.kappa:>12.4e}",
            f"{'─'*55}",
            f"  FRF method         : {self.frf_method:>12s}",
            f"  Freq points        : {self.n_freq_points:>12,}",
            f"  FLOPs (ROM)        : {self.frf_flops_rom:>12,}",
            f"  FLOPs (reference)  : {self.frf_flops_ref:>12,}",
            f"  FLOP speedup       : {self.frf_speedup:>11.1f}×",
            f"{'─'*55}",
            f"  Eigensolver        : {self.t_eigensolver_s:>9.2f}s",
            f"  Mode selection     : {self.t_mode_select_s:>9.2f}s",
            f"  DOF selection      : {self.t_dof_select_s:>9.2f}s",
            f"  ROM build          : {self.t_rom_build_s:>9.2f}s",
            f"  FRF computation    : {self.t_frf_s:>9.2f}s",
            f"  TOTAL              : {self.t_total_s:>9.2f}s",
            f"{'─'*55}",
        ])


def flop_count(
    n_modes: int,
    n_freq: int,
    n_pairs: int,
    method: str = "direct",
) -> int:
    """
    Estimate FRF computation FLOP count.

    Parameters
    ----------
    n_modes : int
        Number of retained modes (m for ROM, all elastic for reference).
    n_freq : int
        Number of evaluation frequency points.
    n_pairs : int
        Number of force/output DOF pairs.
    method : str
        ``"direct"`` or ``"modal"``.

    Returns
    -------
    int
        Approximate floating-point operation count.

    Notes
    -----
    **Modal FRF**: per frequency per mode per pair: ~8 FLOPs
    (2 multiplications + complex division + addition)

    **Direct FRF**: per frequency: one m×m complex LU factor + m×m backsolve
    LU factor:   (2/3) m³ FLOPs
    Backsolve:   m² FLOPs per right-hand side × n_pairs
    """
    if method == "modal":
        return 8 * n_modes * n_freq * n_pairs
    elif method == "direct":
        lu_flops  = int(2 / 3 * n_modes ** 3) * n_freq
        sol_flops = n_modes ** 2 * n_pairs * n_freq
        return lu_flops + sol_flops
    else:
        raise ValueError(f"Unknown method '{method}'")


def reduction_metrics(
    n_full_dofs: int,
    n_master_dofs: int,
    n_all_modes: int,
    n_selected_modes: int,
) -> Dict[str, float]:
    """
    Compute reduction ratio metrics.

    Returns
    -------
    dict with keys:
        ``dof_reduction_ratio``, ``dof_retention_pct``,
        ``mode_retention_pct``, ``size_ratio``
    """
    return {
        "dof_reduction_ratio": n_master_dofs / n_full_dofs,
        "dof_retention_pct":   n_master_dofs / n_full_dofs * 100.0,
        "mode_retention_pct":  n_selected_modes / n_all_modes * 100.0,
        "size_ratio":          n_master_dofs / n_full_dofs,
    }


def summarise_performance(
    n_full_dofs: int,
    n_selected_modes: int,
    n_master_dofs: int,
    n_all_modes: int,
    kappa: float,
    n_freq: int,
    n_bands: int,
    n_pairs: int,
    frf_method: str = "direct",
    frf_flops_rom: Optional[int] = None,
    frf_flops_ref: Optional[int] = None,
    t_eigensolver_s: float = 0.0,
    t_mode_select_s: float = 0.0,
    t_dof_select_s: float = 0.0,
    t_rom_build_s: float = 0.0,
    t_frf_s: float = 0.0,
    t_total_s: float = 0.0,
) -> PerformanceMetrics:
    """
    Collect all performance metrics into a :class:`PerformanceMetrics`.

    Parameters are self-explanatory from the return type documentation.
    """
    if frf_flops_rom is None:
        frf_flops_rom = flop_count(n_selected_modes, n_freq, n_pairs, frf_method)
    if frf_flops_ref is None:
        n_elastic = n_all_modes  # approximate
        frf_flops_ref = flop_count(n_elastic, n_freq, n_pairs, "modal")

    return PerformanceMetrics(
        n_full_dofs        = n_full_dofs,
        n_selected_modes   = n_selected_modes,
        n_master_dofs      = n_master_dofs,
        dof_reduction_pct  = n_master_dofs / max(n_full_dofs, 1) * 100.0,
        mode_retention_pct = n_selected_modes / max(n_all_modes, 1) * 100.0,
        kappa              = kappa,
        frf_method         = frf_method,
        n_freq_points      = n_freq,
        n_bands            = n_bands,
        frf_flops_rom      = frf_flops_rom,
        frf_flops_ref      = frf_flops_ref,
        frf_speedup        = frf_flops_ref / max(frf_flops_rom, 1),
        t_eigensolver_s    = t_eigensolver_s,
        t_mode_select_s    = t_mode_select_s,
        t_dof_select_s     = t_dof_select_s,
        t_rom_build_s      = t_rom_build_s,
        t_frf_s            = t_frf_s,
        t_total_s          = t_total_s,
    )
