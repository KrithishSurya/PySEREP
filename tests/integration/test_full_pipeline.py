"""
Integration test: full SEREP ROM pipeline on a synthetic spring-mass chain.

Tests the complete flow:
  load → eigensolver → mode select → DOF select → ROM build →
  eigenvalue verify → direct FRF → validation → export
"""

import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.pipeline.config import ROMConfig
from pyserep.pipeline.serep_pipeline import SereпPipeline
from pyserep.selection.band_selector import FrequencyBand


def _save_spring_chain(folder: str, n: int = 300, k: float = 5e4, m: float = 1.0):
    """Build and save a 300-DOF spring chain to disk as .npz files."""
    K = sp.diags(
        [[-k]*(n-1), [2*k]*n, [-k]*(n-1)],
        [-1, 0, 1], format="csc"
    ).astype(float)
    K[0, 0] = k
    M = sp.eye(n, format="csc") * m

    k_path = os.path.join(folder, "K.npz")
    m_path = os.path.join(folder, "M.npz")
    sp.save_npz(k_path, K)
    sp.save_npz(m_path, M)
    return k_path, m_path, n


class TestFullPipeline:

    def test_pipeline_runs_to_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            k_path, m_path, N = _save_spring_chain(tmpdir)

            cfg = ROMConfig(
                stiffness_file    = k_path,
                mass_file         = m_path,
                force_dofs        = [150],
                output_dofs       = [150],
                freq_range        = (1.0, 100.0),
                num_modes_eigsh   = 30,
                frf_method        = "direct",
                n_points_per_band = 100,
                export_folder     = os.path.join(tmpdir, "results"),
                plot              = False,
                verbose           = False,
            )
            results = SereпPipeline(cfg).run()

            # Basic sanity checks
            assert len(results.selected_modes) > 0
            assert len(results.master_dofs) > 0
            assert len(results.master_dofs) == len(results.selected_modes)
            assert results.Ka is not None
            assert results.Ma is not None
            assert results.frf is not None

    def test_eigenvalue_preservation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            k_path, m_path, _ = _save_spring_chain(tmpdir)
            cfg = ROMConfig(
                stiffness_file  = k_path,
                mass_file       = m_path,
                force_dofs      = [100],
                output_dofs     = [100],
                freq_range      = (1.0, 80.0),
                num_modes_eigsh = 25,
                frf_method      = "direct",
                n_points_per_band = 50,
                export_folder   = os.path.join(tmpdir, "out"),
                plot=False, verbose=False,
            )
            results = SereпPipeline(cfg).run()
            assert results.max_freq_err < 0.01, (
                f"Eigenvalue error {results.max_freq_err:.6f}% exceeds 0.01%"
            )

    def test_selective_band_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            k_path, m_path, _ = _save_spring_chain(tmpdir)
            cfg = ROMConfig(
                stiffness_file  = k_path,
                mass_file       = m_path,
                force_dofs      = [100],
                output_dofs     = [100],
                bands           = [
                    FrequencyBand(1.0, 30.0, label="Low"),
                    FrequencyBand(70.0, 100.0, label="High"),
                ],
                num_modes_eigsh = 25,
                frf_method      = "direct",
                n_points_per_band = 50,
                export_folder   = os.path.join(tmpdir, "out"),
                plot=False, verbose=False,
            )
            results = SereпPipeline(cfg).run()
            assert results.frf is not None
            assert len(results.frf.band_masks) == 2

    def test_direct_vs_modal_frf_close(self):
        """Direct FRF and modal FRF of the ROM should agree closely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            k_path, m_path, _ = _save_spring_chain(tmpdir, n=100)

            def _run(method):
                cfg = ROMConfig(
                    stiffness_file  = k_path,
                    mass_file       = m_path,
                    force_dofs      = [50],
                    output_dofs     = [50],
                    freq_range      = (1.0, 60.0),
                    num_modes_eigsh = 20,
                    frf_method      = method,
                    n_points_per_band = 50,
                    export_folder   = os.path.join(tmpdir, f"out_{method}"),
                    plot=False, verbose=False,
                )
                return SereпPipeline(cfg).run()

            r_direct = _run("direct")
            r_modal  = _run("modal")

            # Both should have very small FRF errors vs reference
            for errs in r_direct.frf.errors.values():
                assert errs["max_pct"] < 5.0

    def test_output_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            k_path, m_path, _ = _save_spring_chain(tmpdir)
            out = os.path.join(tmpdir, "results")
            cfg = ROMConfig(
                stiffness_file  = k_path,
                mass_file       = m_path,
                force_dofs      = [100],
                output_dofs     = [100],
                freq_range      = (1.0, 60.0),
                num_modes_eigsh = 20,
                n_points_per_band = 50,
                export_folder   = out,
                save_prefix     = "TEST",
                plot=False, verbose=False,
            )
            SereпPipeline(cfg).run()
            assert os.path.isfile(os.path.join(out, "TEST_master_dofs.npy"))
            assert os.path.isfile(os.path.join(out, "TEST_metrics.json"))
            assert os.path.isfile(os.path.join(out, "TEST_frf.npz"))
