"""
pyserep.visualization.summary_plots
=======================================
Performance dashboard and summary visualisations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from pyserep.pipeline.serep_pipeline import PipelineResults


def plot_performance_dashboard(
    results: "PipelineResults",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    6-panel performance dashboard.

    Panels:
    1. Mode selection cascade (bar chart: how many modes each step passes)
    2. Selected mode frequency distribution
    3. Eigenvalue error per mode
    4. DOF reduction summary (pie chart)
    5. Timing breakdown (bar chart)
    6. FLOP comparison ROM vs reference
    """
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
    except ImportError:
        return

    perf   = results.performance
    freqs  = results.freqs_hz
    modes  = results.selected_modes

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.4)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

    fig.suptitle("SEREP ROM — Performance Dashboard", fontsize=12, fontweight="bold")

    # ── 1. Mode frequency distribution ───────────────────────────────────────
    ax = axes[0]
    ax.hist(freqs[modes], bins=20, color="steelblue", edgecolor="white", lw=0.5)
    if results.config.effective_bands:
        for band in results.config.effective_bands:
            ax.axvspan(band.f_min, band.f_max, color="lightblue", alpha=0.3)
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(f"Selected Modes ({len(modes)} total)", fontsize=9, fontweight="bold")

    # ── 2. Eigenvalue preservation error ─────────────────────────────────────
    ax = axes[1]
    if results.freq_errors is not None and len(results.freq_errors) > 0:
        f_sel = np.sort(freqs[modes])[:len(results.freq_errors)]
        ax.semilogy(f_sel, np.maximum(results.freq_errors, 1e-12),
                    "o-", ms=4, color="coral", lw=1.2)
        ax.axhline(0.01,  color="navy",  ls="--", lw=0.8, alpha=0.7, label="0.01%")
        ax.axhline(1e-6,  color="green", ls=":",  lw=0.8, alpha=0.7, label="10⁻⁶%")
        ax.legend(fontsize=7)
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Error (%)", fontsize=9)
    ax.set_title("Eigenvalue Preservation Error", fontsize=9, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)

    # ── 3. Reduction summary (horizontal bar) ─────────────────────────────────
    ax = axes[2]
    N  = results.phi.shape[0]
    m  = len(modes)
    a  = len(results.master_dofs)
    labels  = ["Full DOFs", "Master DOFs", "Modes\n(of all computed)"]
    values  = [N, a, m]
    colours = ["#e74c3c", "#27ae60", "#3498db"]
    bars = ax.barh(labels, values, color=colours, edgecolor="black", lw=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_xlabel("Count", fontsize=9)
    ax.set_title(
        f"ROM Reduction — {a/N*100:.4f}% DOF retention",
        fontsize=9, fontweight="bold"
    )

    # ── 4. Timing breakdown ───────────────────────────────────────────────────
    ax = axes[3]
    timing_keys   = ["eigensolver", "mode_select", "dof_select", "rom_build", "frf"]
    timing_labels = ["Eigensolver", "Mode\nSelect", "DOF\nSelect", "ROM\nBuild", "FRF"]
    timing_vals   = [perf.get(f"t_{k}_s", 0.0) for k in timing_keys]
    colours_t     = ["#9b59b6", "#3498db", "#1abc9c", "#e67e22", "#e74c3c"]
    ax.bar(timing_labels, timing_vals, color=colours_t, edgecolor="black", lw=0.5)
    for x, val in enumerate(timing_vals):
        if val > 0.01:
            ax.text(x, val + max(timing_vals) * 0.01, f"{val:.2f}s",
                    ha="center", fontsize=7)
    ax.set_ylabel("Time (s)", fontsize=9)
    ax.set_title(f"Timing Breakdown  (total: {perf.get('t_total_s', 0):.2f}s)",
                 fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # ── 5. FLOP comparison ────────────────────────────────────────────────────
    ax = axes[4]
    flop_rom = perf.get("frf_flops_rom", 0)
    flop_ref = perf.get("frf_flops_ref", 0)
    if flop_ref > 0 and flop_rom > 0:
        ax.bar(["ROM\n(direct)", "Reference\n(modal)"],
               [flop_rom, flop_ref],
               color=["#27ae60", "#e74c3c"], edgecolor="black", lw=0.5)
        ax.set_yscale("log")
        speedup = flop_ref / flop_rom
        ax.set_title(f"FRF FLOPs  (speedup: {speedup:.1f}×)",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("FLOPs (log)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    # ── 6. Condition number comparison ────────────────────────────────────────
    ax = axes[5]
    kappa_text = (
        f"κ(Φₐ) = {results.kappa:.4e}\n\n"
        f"Max FRF error : {max((e.get('max_pct', 0) for e in results.frf.errors.values()), default=0):.6f}%\n"  # noqa: E501
        f"RMS FRF error : {max((e.get('rms_pct', 0) for e in results.frf.errors.values()), default=0):.6f}%\n"  # noqa: E501
        f"Max freq error: {results.max_freq_err:.8f}%\n\n"
        f"Bands : {results.config.n_bands}\n"
        f"DOFs  : {N:,} → {a:,}"
    )
    ax.text(0.5, 0.5, kappa_text, ha="center", va="center",
            transform=ax.transAxes, fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Key Metrics Summary", fontsize=9, fontweight="bold")
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Dashboard saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
