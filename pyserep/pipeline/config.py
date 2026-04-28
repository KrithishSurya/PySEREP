"""
pyserep.pipeline.config
==========================
ROMConfig — the single configuration object for the SEREP pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pyserep.selection.band_selector import FrequencyBand


@dataclass
class ROMConfig:
    """
    Complete configuration for a SEREP ROM pipeline run.

    Required fields
    ---------------
    stiffness_file, mass_file : str
        Paths to K and M matrix files.
    force_dofs, output_dofs : list of int
        Global DOF indices (0-based).  Must have equal length.

    Frequency specification (choose one)
    -------------------------------------
    bands : list of FrequencyBand
        Selective multi-band analysis.
    freq_range : (f_min, f_max) tuple
        Single contiguous band (backward-compatible fallback).

    FRF method
    ----------
    frf_method : str
        ``"direct"``  — impedance inversion of Kₐ, Mₐ  (recommended)
        ``"modal"``   — modal superposition (v2 compatibility)

    Damping
    -------
    zeta : float
        Uniform damping ratio.
    damping_type : str
        ``"modal"`` | ``"rayleigh"`` | ``"hysteretic"`` | ``"none"``

    Examples
    --------
    >>> cfg = ROMConfig(
    ...     stiffness_file="K.mtx",
    ...     mass_file="M.mtx",
    ...     force_dofs=[3000],
    ...     output_dofs=[3000],
    ...     bands=[FrequencyBand(0, 100), FrequencyBand(400, 500)],
    ...     frf_method="direct",
    ... )
    """

    # ── Required ──────────────────────────────────────────────────────────────
    stiffness_file: str = ""
    mass_file:      str = ""
    force_dofs:     List[int] = field(default_factory=lambda: [3000])
    output_dofs:    List[int] = field(default_factory=lambda: [3000])

    # ── Frequency bands ───────────────────────────────────────────────────────
    bands:       Optional[List[FrequencyBand]] = None
    freq_range:  Tuple[float, float] = (0.1, 500.0)

    # ── FRF ───────────────────────────────────────────────────────────────────
    frf_method:      str   = "direct"    # "direct" or "modal"
    damping_type:    str   = "modal"     # "modal", "rayleigh", "hysteretic", "none"
    zeta:            float = 0.001
    n_points_per_band: int = 2000

    # ── Eigensolver ───────────────────────────────────────────────────────────
    num_modes_eigsh: int   = 100
    eigsh_sigma:     float = 0.01
    eigsh_tol:       float = 1e-10

    # ── Mode selection ────────────────────────────────────────────────────────
    ms1_alpha:      float = 1.5
    ms2_threshold:  float = 1.0
    ms3_threshold:  float = 5.0
    mac_threshold:  float = 0.90
    rb_hz:          float = 1.0

    # ── DOF selection ─────────────────────────────────────────────────────────
    dof_method:          str   = "eid"   # "eid", "kinetic", "modal_disp", "svd"
    ke_prescreen_frac:   float = 0.5

    # ── Output ────────────────────────────────────────────────────────────────
    export_folder:   str  = "pyserep_output"
    save_prefix:     str  = "SEREP"
    save_matrices:   bool = True
    plot:            bool = True
    verbose:         bool = True

    # ── Derived (populated in __post_init__) ──────────────────────────────────
    _effective_bands: List[FrequencyBand] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._validate()
        self._build_bands()

    def _validate(self) -> None:
        for attr, name in [("stiffness_file", "K"), ("mass_file", "M")]:
            p = getattr(self, attr)
            if p and not os.path.exists(p):
                raise FileNotFoundError(f"Matrix file not found ({name}): '{p}'")
        if len(self.force_dofs) != len(self.output_dofs):
            raise ValueError(
                f"force_dofs length ({len(self.force_dofs)}) ≠ "
                f"output_dofs length ({len(self.output_dofs)})"
            )
        if self.frf_method not in ("direct", "modal"):
            raise ValueError(f"frf_method must be 'direct' or 'modal', got '{self.frf_method}'")
        if self.dof_method not in ("eid", "kinetic", "modal_disp", "svd"):
            raise ValueError(f"Unknown dof_method '{self.dof_method}'")

    def _build_bands(self) -> None:
        if self.bands:
            self._effective_bands = list(self.bands)
        else:
            f0, f1 = self.freq_range
            self._effective_bands = [FrequencyBand(f0, f1, label="FullRange")]

    @property
    def effective_bands(self) -> List[FrequencyBand]:
        """Resolved list of FrequencyBand objects (from bands or freq_range)."""
        return self._effective_bands

    @property
    def global_f_max(self) -> float:
        """Maximum frequency across all analysis bands (Hz)."""
        return max(b.f_max for b in self._effective_bands)

    @property
    def global_f_min(self) -> float:
        """Minimum frequency across all analysis bands (Hz)."""
        return min(b.f_min for b in self._effective_bands)

    @property
    def n_bands(self) -> int:
        """Number of analysis frequency bands."""
        return len(self._effective_bands)

    @property
    def n_pairs(self) -> int:
        """Number of force/output DOF pairs."""
        return len(self.force_dofs)

    @property
    def is_selective(self) -> bool:
        """True when there are two or more bands (gap regions exist)."""
        return len(self._effective_bands) > 1

    def summary(self) -> str:
        """Return a formatted string listing all configuration parameters."""
        lines = [
            "ROMConfig",
            f"  K file          : {self.stiffness_file}",
            f"  M file          : {self.mass_file}",
            f"  Force DOFs      : {self.force_dofs}",
            f"  Output DOFs     : {self.output_dofs}",
            f"  FRF method      : {self.frf_method}  ({self.damping_type} damping, ζ={self.zeta})",
        ]
        if self.is_selective:
            lines.append("  Frequency bands : SELECTIVE")
            for b in self._effective_bands:
                lines.append(f"    {b.label:20s}  [{b.f_min:.1f}, {b.f_max:.1f}] Hz")
        else:
            b = self._effective_bands[0]
            lines.append(f"  Freq range      : [{b.f_min:.1f}, {b.f_max:.1f}] Hz")
        lines += [
            f"  MS1 α / cutoff  : {self.ms1_alpha} / {self.ms1_alpha*self.global_f_max:.1f} Hz",
            f"  MS2 threshold   : {self.ms2_threshold:.1f}%",
            f"  MS3 threshold   : {self.ms3_threshold:.1f}%",
            f"  MAC threshold   : {self.mac_threshold:.2f}",
            f"  DOF selector    : {self.dof_method.upper()}",
            f"  Modes (eigsh)   : {self.num_modes_eigsh}",
            f"  Points/band     : {self.n_points_per_band}",
            f"  Export folder   : {self.export_folder}",
        ]
        return "\n".join(lines)
