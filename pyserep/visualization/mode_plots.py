"""
pyserep.visualization.mode_plots
=====================================
Mode shape and DOF selection visualisation tools.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def plot_mode_shapes(
    phi: np.ndarray,
    freqs_hz: np.ndarray,
    mode_indices: np.ndarray,
    master_dofs: Optional[np.ndarray] = None,
    n_cols: int = 4,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot mode shape amplitudes for the selected modes.

    Shows the full-DOF modal amplitude profile for each selected mode.
    Master DOFs are highlighted with red markers if provided.

    Parameters
    ----------
    phi : np.ndarray, shape (N, n_modes)
    freqs_hz : np.ndarray
    mode_indices : np.ndarray of int
    master_dofs : np.ndarray of int, optional
    n_cols : int
    save_path, show : optional
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    m      = len(mode_indices)
    n_rows = (m + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).ravel()

    for idx, mode_idx in enumerate(mode_indices):
        ax  = axes[idx]
        amp = np.abs(phi[:, mode_idx])
        ax.plot(amp, lw=0.8, color="steelblue", alpha=0.7)
        if master_dofs is not None:
            ax.scatter(master_dofs, amp[master_dofs],
                       s=12, c="red", zorder=5, label="Master DOFs")
        ax.set_title(
            f"Mode {mode_idx}\n{freqs_hz[mode_idx]:.2f} Hz",
            fontsize=8,
        )
        ax.set_xlabel("DOF", fontsize=7)
        ax.set_ylabel("|φ|", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(m, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Selected Mode Shapes ({m} modes)", fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[plot] Mode shapes saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_mac_matrix(
    mac_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot the MAC matrix as a colour map.

    Parameters
    ----------
    mac_matrix : np.ndarray, shape (m, m)
    labels : list of str, optional
    save_path, show : optional
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    m = mac_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, m // 3), max(5, m // 3)))
    im = ax.imshow(mac_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="MAC value")

    if labels:
        ax.set_xticks(range(m))
        ax.set_yticks(range(m))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)

    ax.set_title("Modal Assurance Criterion (MAC) Matrix")
    ax.set_xlabel("ROM modes")
    ax.set_ylabel("Reference modes")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_dof_selector_comparison(
    comparison: dict,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Bar chart comparing κ values of all four DOF selectors.

    Parameters
    ----------
    comparison : dict
        Output of :func:`~pyserep.selection.dof_selector.compare_dof_selectors`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(comparison.keys())
    kappas  = [comparison[m]["kappa"] for m in methods]
    colours = ["#e74c3c", "#e67e22", "#f1c40f", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, kappas, color=colours[:len(methods)], edgecolor="black", lw=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Condition number κ(Φₐ)  [log scale]")
    ax.set_title("DOF Selector Comparison — Condition Number κ(Φₐ)")

    for bar, kappa in zip(bars, kappas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{kappa:.1e}", ha="center", va="bottom", fontsize=9)

    ax.axhline(1e3, color="green", ls="--", lw=1, label="Good (κ < 10³)")
    ax.axhline(1e6, color="orange", ls="--", lw=1, label="Marginal (κ < 10⁶)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
