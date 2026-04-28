"""Unit tests for pyserep.analysis.convergence."""

import numpy as np
import pytest

from pyserep.models.synthetic import spring_chain
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.analysis.convergence import (
    mode_count_study, dof_count_study,
    ConvergenceStudy, ConvergencePoint,
)
from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.selection.mode_selector import select_modes_pipeline
from pyserep.selection.dof_selector import select_dofs_eid
from pyserep.frf.modal_frf import compute_frf_modal_reference


def _small_model():
    K, M = spring_chain(n=80, k=1e4)
    freqs, phi = solve_eigenproblem(K, M, n_modes=25, verbose=False)
    return K, M, freqs, phi


class TestConvergencePoint:

    def test_fields_accessible(self):
        pt = ConvergencePoint(
            param_value=50.0, n_modes=10, n_dofs=10,
            kappa=25.0, max_frf_err_pct=0.01,
            rms_frf_err_pct=0.005, max_freq_err_pct=1e-6,
        )
        assert pt.param_value == 50.0
        assert pt.n_modes == 10
        assert pt.kappa == 25.0


class TestModeCountStudy:

    def setup_method(self):
        self.K, self.M, self.freqs, self.phi = _small_model()

    def test_returns_convergence_study(self):
        study = mode_count_study(
            self.K, self.M, self.phi, self.freqs,
            force_dofs=[40], output_dofs=[40],
            f_max=60.0,
            f_max_values=[20.0, 40.0, 60.0],
            zeta=0.01, n_freq=100, verbose=False,
        )
        assert isinstance(study, ConvergenceStudy)

    def test_points_increase_with_cutoff(self):
        study = mode_count_study(
            self.K, self.M, self.phi, self.freqs,
            force_dofs=[40], output_dofs=[40],
            f_max=60.0,
            f_max_values=[20.0, 40.0, 60.0],
            zeta=0.01, n_freq=80, verbose=False,
        )
        if len(study.points) >= 2:
            # More modes retained as cutoff grows
            n_modes = [p.n_modes for p in study.points]
            assert n_modes == sorted(n_modes)

    def test_table_is_string(self):
        study = mode_count_study(
            self.K, self.M, self.phi, self.freqs,
            force_dofs=[40], output_dofs=[40],
            f_max=50.0, f_max_values=[30.0, 50.0],
            zeta=0.01, n_freq=80, verbose=False,
        )
        table = study.table()
        assert isinstance(table, str)
        assert "Convergence" in table

    def test_param_name(self):
        study = mode_count_study(
            self.K, self.M, self.phi, self.freqs,
            force_dofs=[40], output_dofs=[40],
            f_max=50.0, f_max_values=[30.0],
            zeta=0.01, n_freq=80, verbose=False,
        )
        assert study.param_name == "f_max"


class TestDofCountStudy:

    def setup_method(self):
        K, M, freqs, phi = _small_model()
        self.K, self.M, self.freqs, self.phi = K, M, freqs, phi

        band = FrequencyBandSet([FrequencyBand(1.0, 40.0)], n_points_per_band=50)
        self.sel = select_modes_pipeline(phi, freqs, [40], [40], band, verbose=False)
        self.freq_eval = band.frequency_grid()
        self.H_ref = compute_frf_modal_reference(
            phi, freqs, 1.0, [40], [40], band, zeta=0.01, verbose=False
        )

    def test_returns_convergence_study(self):
        m = len(self.sel)
        study = dof_count_study(
            self.K, self.M, self.phi, self.freqs, self.sel,
            [40], [40],
            n_master_values=[m, m + 5],
            freq_eval=self.freq_eval,
            H_ref=self.H_ref,
            zeta=0.01, verbose=False,
        )
        assert isinstance(study, ConvergenceStudy)
        assert study.param_name == "n_master"

    def test_skips_n_master_below_m(self):
        m = len(self.sel)
        study = dof_count_study(
            self.K, self.M, self.phi, self.freqs, self.sel,
            [40], [40],
            n_master_values=[m - 5, m, m + 5],  # m-5 should be skipped
            freq_eval=self.freq_eval,
            H_ref=self.H_ref,
            zeta=0.01, verbose=False,
        )
        for pt in study.points:
            assert pt.n_dofs >= m
