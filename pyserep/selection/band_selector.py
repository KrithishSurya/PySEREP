"""
pyserep.selection.band_selector
===================================
FrequencyBand and FrequencyBandSet — frequency band management for
selective SEREP analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FrequencyBand:
    """
    A single contiguous analysis band [f_min, f_max] Hz.

    Parameters
    ----------
    f_min : float
        Lower bound (Hz).  Use 0.0 for DC.
    f_max : float
        Upper bound (Hz).
    label : str, optional
        Human-readable label.  Auto-generated if not supplied.
    n_points : int, optional
        Number of frequency evaluation points in this band.
        Overrides the global ``n_points_per_band`` when set.

    Examples
    --------
    >>> band = FrequencyBand(0, 100, label="LowBand")
    >>> band.contains(50)
    True
    >>> band.span
    100.0
    """

    f_min: float
    f_max: float
    label: Optional[str] = None
    n_points: Optional[int] = None

    def __post_init__(self) -> None:
        if self.f_max <= self.f_min:
            raise ValueError(
                f"FrequencyBand requires f_max > f_min; got [{self.f_min}, {self.f_max}]"
            )
        if self.f_min < 0:
            raise ValueError("f_min must be >= 0")
        if self.label is None:
            object.__setattr__(
                self, "label",
                f"Band_{self.f_min:.0f}-{self.f_max:.0f}Hz",
            )

    @property
    def span(self) -> float:
        """Width of the band in Hz."""
        return self.f_max - self.f_min

    @property
    def centre(self) -> float:
        """Centre frequency in Hz."""
        return (self.f_min + self.f_max) / 2.0

    def contains(self, freq: float) -> bool:
        """Return True if *freq* lies within this band (inclusive)."""
        return self.f_min <= freq <= self.f_max

    def expanded(self, alpha: float) -> "FrequencyBand":
        """Return a new band with f_max scaled by *alpha* (MS1 safety factor)."""
        return FrequencyBand(
            self.f_min, self.f_max * alpha,
            label=f"{self.label}_x{alpha:.2f}",
        )

    def __repr__(self) -> str:
        return f"FrequencyBand({self.f_min:.1f}–{self.f_max:.1f} Hz, '{self.label}')"


class FrequencyBandSet:
    """
    Container for a collection of FrequencyBand objects.

    Manages the multi-band frequency grid, mode relevance checks,
    and band-weighted Modal Participation Factor computation.

    Parameters
    ----------
    bands : sequence of FrequencyBand
    n_points_per_band : int
        Default frequency evaluation points per band.

    Examples
    --------
    >>> bset = FrequencyBandSet([FrequencyBand(0, 100), FrequencyBand(400, 500)])
    >>> bset.n_bands
    2
    >>> len(bset.frequency_grid())
    4000
    """

    def __init__(
        self,
        bands: Sequence[FrequencyBand],
        n_points_per_band: int = 2000,
    ) -> None:
        if not bands:
            raise ValueError("At least one FrequencyBand is required.")
        self._bands: List[FrequencyBand] = sorted(bands, key=lambda b: b.f_min)
        self._n_default = n_points_per_band
        self._validate()

    def _validate(self) -> None:
        for i in range(len(self._bands) - 1):
            a, b = self._bands[i], self._bands[i + 1]
            if a.f_max > b.f_min:
                warnings.warn(
                    f"Bands '{a.label}' and '{b.label}' overlap "
                    f"({a.f_max:.1f} Hz > {b.f_min:.1f} Hz).",
                    UserWarning, stacklevel=3,
                )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def bands(self) -> List[FrequencyBand]:
        """Return a copy of the band list, sorted by f_min."""
        return list(self._bands)

    @property
    def n_bands(self) -> int:
        """Number of frequency bands in the set."""
        return len(self._bands)

    @property
    def global_f_min(self) -> float:
        """Lower bound of the lowest band (Hz)."""
        return self._bands[0].f_min

    @property
    def global_f_max(self) -> float:
        """Upper bound of the highest band (Hz)."""
        return self._bands[-1].f_max

    @property
    def is_selective(self) -> bool:
        """True when there are gaps between bands."""
        return self.n_bands > 1

    # ── Frequency grid ────────────────────────────────────────────────────────

    def frequency_grid(self) -> np.ndarray:
        """
        Build the evaluation frequency array (Hz) as the union of all bands.

        Returns
        -------
        np.ndarray, sorted and deduplicated.
        """
        grids = []
        for band in self._bands:
            n = band.n_points if band.n_points is not None else self._n_default
            grids.append(np.linspace(band.f_min, band.f_max, n))
        return np.sort(np.unique(np.concatenate(grids)))

    def frequency_mask(self, freqs_hz: np.ndarray) -> np.ndarray:
        """Boolean mask: True where *freqs_hz* falls inside any band."""
        mask = np.zeros(freqs_hz.shape, dtype=bool)
        for band in self._bands:
            mask |= (freqs_hz >= band.f_min) & (freqs_hz <= band.f_max)
        return mask

    # ── Mode relevance ────────────────────────────────────────────────────────

    def mode_passes_ms1(
        self,
        freq_hz: float,
        rb_hz: float = 1.0,
        alpha: float = 1.5,
    ) -> bool:
        """True if a mode passes the MS1 frequency-range criterion."""
        if freq_hz <= rb_hz:
            return False
        return any(freq_hz <= alpha * b.f_max for b in self._bands)

    def band_weighted_mpf(
        self,
        phi_f: np.ndarray,
        phi_o: np.ndarray,
        omega_n: np.ndarray,
        band: FrequencyBand,
    ) -> np.ndarray:
        """
        Band-weighted Modal Participation Factor for all modes.

        C_i = |phi_f_i * phi_o_i| * max_ω_in_band(1 / |ωᵢ² − ω²|)

        Parameters
        ----------
        phi_f, phi_o : (n_modes,)
        omega_n : (n_modes,)
        band : FrequencyBand

        Returns
        -------
        (n_modes,) band-weighted MPF
        """
        n = band.n_points if band.n_points is not None else self._n_default
        omega_eval = 2.0 * np.pi * np.linspace(band.f_min, band.f_max, n)
        numerator  = np.abs(phi_f * phi_o)
        denom = np.abs(omega_n[:, None] ** 2 - omega_eval[None, :] ** 2)
        eps   = max(1e-6 * (omega_n.mean() ** 2), 1e-6)
        denom = np.maximum(denom, eps)
        return numerator * (1.0 / denom).max(axis=1)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable multi-line summary of all bands and gaps."""
        lines = [f"FrequencyBandSet  ({self.n_bands} band(s))"]
        for b in self._bands:
            n = b.n_points if b.n_points is not None else self._n_default
            lines.append(
                f"  {b.label:25s}  [{b.f_min:8.2f}, {b.f_max:8.2f}] Hz  {n} pts"
            )
        if self.is_selective:
            for i in range(len(self._bands) - 1):
                lo = self._bands[i].f_max
                hi = self._bands[i + 1].f_min
                if hi > lo:
                    lines.append(f"  GAP: [{lo:.1f}, {hi:.1f}] Hz (ignored)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = ", ".join(f"[{b.f_min:.0f}–{b.f_max:.0f}Hz]" for b in self._bands)
        return f"FrequencyBandSet({parts})"
