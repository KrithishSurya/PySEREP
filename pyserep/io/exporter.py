"""
pyserep.io.exporter
=====================
Save and reload pipeline results in multiple formats.

Outputs
-------
``<prefix>_master_dofs.npy``        — master DOF indices
``<prefix>_selected_modes.npy``     — selected mode indices
``<prefix>_freqs_selected.npy``     — corresponding natural frequencies (Hz)
``<prefix>_Ka.npy``                 — reduced stiffness matrix
``<prefix>_Ma.npy``                 — reduced mass matrix
``<prefix>_T.npy``                  — SEREP transformation matrix
``<prefix>_frf.npz``                — ROM & reference FRF arrays
``<prefix>_metrics.json``           — all scalar performance metrics
``<prefix>_summary.txt``            — human-readable summary
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from pyserep.pipeline.serep_pipeline import PipelineResults


def save_results(
    results: "PipelineResults",
    folder: str,
    prefix: str = "SEREP",
    save_matrices: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Save all pipeline results to *folder*.

    Parameters
    ----------
    results : PipelineResults
    folder : str
        Output directory (created if it does not exist).
    prefix : str
        Filename prefix for all output files.
    save_matrices : bool
        If True, save Ka, Ma, and T matrices.  These can be large;
        set to False to save only indices, FRF, and metrics.
    verbose : bool

    Returns
    -------
    dict
        Mapping of result type → absolute file path.
    """
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, prefix)
    saved: Dict[str, str] = {}

    # ── Indices ──────────────────────────────────────────────────────────────
    np.save(f"{p}_master_dofs.npy", results.master_dofs)
    np.save(f"{p}_selected_modes.npy", results.selected_modes)
    np.save(f"{p}_freqs_selected.npy",
            results.freqs_hz[results.selected_modes])
    saved.update(
        master_dofs     = f"{p}_master_dofs.npy",
        selected_modes  = f"{p}_selected_modes.npy",
        freqs_selected  = f"{p}_freqs_selected.npy",
    )

    # ── Reduced matrices ──────────────────────────────────────────────────────
    if save_matrices and results.Ka is not None:
        np.save(f"{p}_Ka.npy", results.Ka)
        np.save(f"{p}_Ma.npy", results.Ma)
        np.save(f"{p}_T.npy",  results.T)
        saved.update(Ka=f"{p}_Ka.npy", Ma=f"{p}_Ma.npy", T=f"{p}_T.npy")

    # ── FRF ───────────────────────────────────────────────────────────────────
    if results.frf is not None:
        frf_path = f"{p}_frf.npz"
        np.savez(
            frf_path,
            freqs_hz = results.frf.freqs_hz,
            **{f"rom_{k}": v for k, v in results.frf.H_rom.items()},
            **{f"ref_{k}": v for k, v in results.frf.H_ref.items()},
        )
        saved["frf"] = frf_path

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics = _build_metrics_dict(results)
    metrics_path = f"{p}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    saved["metrics"] = metrics_path

    # ── Human-readable summary ────────────────────────────────────────────────
    summary_path = f"{p}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(results.summary())
        f.write(f"\n\nSaved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    saved["summary"] = summary_path

    if verbose:
        print(f"\n[I/O] Results saved to '{folder}/'")
        for key, path in saved.items():
            print(f"  {key:20s}: {os.path.basename(path)}")

    return saved


def _build_metrics_dict(results: "PipelineResults") -> Dict[str, Any]:
    """Collect all scalar metrics into a JSON-serialisable dictionary."""
    m: Dict[str, Any] = {
        "version":          "3.0.0",
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_full_dofs":      int(results.phi.shape[0]),
        "n_selected_modes": int(len(results.selected_modes)),
        "n_master_dofs":    int(len(results.master_dofs)),
        "reduction_ratio":  float(len(results.master_dofs) / results.phi.shape[0]),
        "kappa":            float(results.kappa),
        "max_freq_error_pct": float(results.max_freq_err),
        "elapsed_total_s":  float(results.elapsed_total_s),
    }
    if results.frf is not None:
        m["frf_errors"] = {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in results.frf.errors.items()
        }
    if results.performance:
        import dataclasses
        perf_dict = dataclasses.asdict(results.performance)
        m["performance"] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
            for k, v in perf_dict.items()
        }
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_frf_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load a saved FRF ``.npz`` file.

    Returns
    -------
    dict with keys:
        ``freqs_hz``  — evaluation frequencies (Hz)
        ``rom_*``     — ROM FRF arrays (complex)
        ``ref_*``     — reference FRF arrays (complex)
    """
    data = np.load(path, allow_pickle=False)
    return dict(data)


def load_metrics(folder: str, prefix: str = "SEREP") -> Dict[str, Any]:
    """Load a saved metrics JSON file."""
    path = os.path.join(folder, f"{prefix}_metrics.json")
    with open(path, "r") as f:
        return json.load(f)


def load_reduced_matrices(
    folder: str,
    prefix: str = "SEREP",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load saved Ka, Ma, and T matrices.

    Returns
    -------
    Ka, Ma, T : np.ndarray
    """
    p = os.path.join(folder, prefix)
    Ka = np.load(f"{p}_Ka.npy")
    Ma = np.load(f"{p}_Ma.npy")
    T  = np.load(f"{p}_T.npy")
    return Ka, Ma, T
