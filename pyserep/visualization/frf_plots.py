"""
pyserep.visualization.frf_plots
====================================
FRF comparison and error plots with selective-band shading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from pyserep.frf.direct_frf import FRFResult
    from pyserep.pipeline.serep_pipeline import PipelineResults
    from pyserep.selection.band_selector import FrequencyBand


def plot_frf_comparison(
    results: "PipelineResults",
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 150,
) -> None:
    """
    Four-panel FRF comparison figure.

    Panel 1: FRF magnitude (dB re 1 m/N) — ROM vs reference
    Panel 2: FRF phase (degrees) — ROM vs reference
    Panel 3: Percentage FRF error (log scale)
    Panel 4: Eigenvalue preservation error per mode

    Gap regions between selective bands are shaded grey.

    Parameters
    ----------
    results : PipelineResults
    save_path : str, optional
    show : bool
    dpi : int
    """
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available — skipping.")
        return

    frf    = results.frf
    bands  = results.config.effective_bands
    freqs  = frf.freqs_hz
    keys   = list(frf.H_rom.keys())
    colours = plt.cm.tab10.colors

    fig = plt.figure(figsize=(13, 11))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    title = (
        f"SEREP ROM — Direct FRF  |  "
        f"{len(results.selected_modes)} modes  |  "
        f"{len(results.master_dofs)} master DOFs  |  "
        f"κ = {results.kappa:.2e}"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # ── Shared band decoration ────────────────────────────────────────────────
    def _decorate(ax):
        _shade_gaps(ax, bands, freqs)
        for band in bands:
            ax.axvline(band.f_min, color="steelblue", lw=0.7, ls=":", alpha=0.7)
            ax.axvline(band.f_max, color="steelblue", lw=0.7, ls=":", alpha=0.7)
        ax.set_xlim(freqs.min(), freqs.max())
        ax.grid(True, which="both", alpha=0.25)

    # ── Panel 1: Magnitude (dB) ───────────────────────────────────────────────
    ax = axes[0]
    for idx, key in enumerate(keys):
        c = colours[idx % len(colours)]
        ref_db = 20 * np.log10(np.maximum(np.abs(frf.H_ref[key]), 1e-30))
        rom_db = 20 * np.log10(np.maximum(np.abs(frf.H_rom[key]), 1e-30))
        ax.plot(freqs, ref_db, color=c, lw=1.2, ls="-",  alpha=0.55, label=f"Ref {key}")
        ax.plot(freqs, rom_db, color=c, lw=1.8, ls="--",             label=f"ROM {key}")
    _decorate(ax)
    ax.set_ylabel("|H| (dB re 1 m/N)", fontsize=9)
    ax.set_title("FRF Magnitude — dashed = ROM, solid = reference", fontsize=9)
    ax.legend(fontsize=7, ncol=min(len(keys) * 2, 4), loc="upper right")

    # ── Panel 2: Phase ────────────────────────────────────────────────────────
    ax = axes[1]
    for idx, key in enumerate(keys):
        c = colours[idx % len(colours)]
        ax.plot(freqs, np.degrees(np.angle(frf.H_ref[key])),
                color=c, lw=1.2, ls="-", alpha=0.55)
        ax.plot(freqs, np.degrees(np.angle(frf.H_rom[key])),
                color=c, lw=1.8, ls="--")
    _decorate(ax)
    ax.set_ylabel("Phase (°)", fontsize=9)
    ax.set_ylim(-195, 195)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_title("FRF Phase", fontsize=9)

    # ── Panel 3: Error (%) ────────────────────────────────────────────────────
    ax = axes[2]
    for idx, key in enumerate(frf.errors):
        c = colours[idx % len(colours)]
        h_r = np.abs(frf.H_rom[key])
        h_f = np.abs(frf.H_ref[key])
        err = np.abs(h_r - h_f) / np.where(h_f > 1e-30, h_f, 1e-30) * 100.0
        ax.semilogy(freqs, np.maximum(err, 1e-8), color=c, lw=1.0,
                    label=f"{key}: max={frf.errors[key]['max_pct']:.4f}%")
    _decorate(ax)
    ax.set_ylabel("FRF Error (%)", fontsize=9)
    ax.set_title("Relative FRF Error (log scale)", fontsize=9)
    ax.legend(fontsize=7)

    # ── Panel 4: Eigenvalue error ─────────────────────────────────────────────
    ax = axes[3]
    if results.freq_errors is not None and len(results.freq_errors) > 0:
        f_sel = np.sort(results.freqs_hz[results.selected_modes])[:len(results.freq_errors)]
        ax.semilogy(f_sel, np.maximum(results.freq_errors, 1e-12),
                    "o-", ms=4, color="coral", lw=1.2, label="Freq error")
        ax.axhline(0.01,  color="navy",  ls="--", lw=0.8, label="0.01% (PASS)")
        ax.axhline(1e-6,  color="green", ls=":",  lw=0.8, label="10⁻⁶% (exact)")
        ax.set_ylabel("Freq Error (%)", fontsize=9)
        ax.set_title("Eigenvalue Preservation Error per Mode", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlim(freqs.min(), freqs.max())

    for ax in axes:
        ax.set_xlabel("Frequency (Hz)", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[plot] Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_frf_overlay(
    frf_result: "FRFResult",
    title: str = "FRF Overlay",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Simple FRF magnitude overlay — ROM vs reference, single figure.

    Useful for quick inspection without a full PipelineResults object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    freqs   = frf_result.freqs_hz
    colours = plt.cm.tab10.colors

    for idx, key in enumerate(frf_result.H_rom):
        c = colours[idx % len(colours)]
        ax.semilogy(freqs, np.abs(frf_result.H_ref[key]),
                    color=c, lw=1.2, ls="-", alpha=0.5, label=f"Ref {key}")
        ax.semilogy(freqs, np.abs(frf_result.H_rom[key]),
                    color=c, lw=1.8, ls="--",            label=f"ROM {key}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|H| (m/N)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shade_gaps(ax, bands: "List[FrequencyBand]", freqs: np.ndarray) -> None:
    """Shade frequency gap regions between selective bands."""
    for i in range(len(bands) - 1):
        lo = bands[i].f_max
        hi = bands[i + 1].f_min
        if hi > lo:
            ax.axvspan(lo, hi, color="0.88", alpha=0.7,
                       label="Gap (ignored)" if i == 0 else "")
