"""Unit tests for pyserep.analysis.sensitivity."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.models.synthetic import spring_chain
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.selection.mode_selector import select_modes_pipeline
from pyserep.selection.dof_selector import select_dofs_eid
from pyserep.core.rom_builder import build_serep_rom
from pyserep.analysis.sensitivity import (
    eigenvalue_sensitivity,
    frf_sensitivity,
    material_perturbation_study,
    monte_carlo_frf,
)
from pyserep.frf.direct_frf import compute_frf_direct
from pyserep.utils.sparse_ops import build_dof_map


def _build_rom():
    """Build a small ROM for sensitivity tests."""
    N = 60
    K, M = spring_chain(N, k=5e3)
    freqs, phi = solve_eigenproblem(K, M, n_modes=18, verbose=False)
    band = FrequencyBandSet([FrequencyBand(1.0, 40.0)], n_points_per_band=80)
    sel  = select_modes_pipeline(phi, freqs, [N//2], [N//2], band, verbose=False)
    req  = np.array([N//2])
    dofs, _ = select_dofs_eid(phi, sel, required_dofs=req, verbose=False)
    T, Ka, Ma = build_serep_rom(K, M, phi, sel, dofs, verbose=False)
    freq_eval = band.frequency_grid()
    lf, lo = build_dof_map(dofs, [N//2], [N//2])
    _, H_nom = compute_frf_direct(Ka, Ma, lf, lo, freq_eval, zeta=0.01, verbose=False)
    return K, M, phi, freqs, sel, dofs, Ka, Ma, lf, lo, freq_eval, H_nom


class TestEigenvalueSensitivity:

    def setup_method(self):
        res = _build_rom()
        (self.K, self.M, self.phi, self.freqs,
         self.sel, self.dofs, self.Ka, self.Ma,
         self.lf, self.lo, self.freq_eval, self.H_nom) = res

    def test_output_shape(self):
        dK = 0.01 * self.K
        dM = sp.csc_matrix(self.K.shape, dtype=float)
        dlam = eigenvalue_sensitivity(
            self.K, self.M, self.phi, self.freqs, self.sel,
            dK, dM, verbose=False,
        )
        assert dlam.shape == (len(self.sel),)

    def test_stiffness_increase_raises_frequencies(self):
        """∂λᵢ/∂E > 0: stiffening raises natural frequencies."""
        dK = 0.01 * self.K
        dM = sp.csc_matrix(self.K.shape, dtype=float)
        dlam = eigenvalue_sensitivity(
            self.K, self.M, self.phi, self.freqs, self.sel,
            dK, dM, verbose=False,
        )
        # Most modes should have positive sensitivity to stiffness
        assert np.sum(dlam > 0) > np.sum(dlam < 0)

    def test_zero_dK_gives_zero_sensitivity(self):
        dK = sp.csc_matrix(self.K.shape, dtype=float)
        dM = sp.csc_matrix(self.K.shape, dtype=float)
        dlam = eigenvalue_sensitivity(
            self.K, self.M, self.phi, self.freqs, self.sel,
            dK, dM, verbose=False,
        )
        assert np.allclose(dlam, 0.0, atol=1e-10)


class TestFRFSensitivity:

    def setup_method(self):
        res = _build_rom()
        (_, _, _, _,
         _, _, self.Ka, self.Ma,
         self.lf, self.lo, self.freq_eval, self.H_nom) = res

    def test_output_shape(self):
        dKa = 0.01 * self.Ka
        dMa = np.zeros_like(self.Ma)
        dH = frf_sensitivity(
            self.Ka, self.Ma, dKa, dMa,
            self.lf, self.lo, self.freq_eval,
            zeta=0.01,
        )
        key = f"f{self.lf[0]}_o{self.lo[0]}"
        assert key in dH
        assert dH[key].shape == (len(self.freq_eval),)

    def test_complex_output(self):
        dKa = 0.01 * self.Ka
        dMa = np.zeros_like(self.Ma)
        dH = frf_sensitivity(
            self.Ka, self.Ma, dKa, dMa,
            self.lf, self.lo, self.freq_eval,
            zeta=0.01,
        )
        key = f"f{self.lf[0]}_o{self.lo[0]}"
        assert np.iscomplexobj(dH[key])

    def test_zero_perturbation_gives_zero(self):
        dKa = np.zeros_like(self.Ka)
        dMa = np.zeros_like(self.Ma)
        dH = frf_sensitivity(
            self.Ka, self.Ma, dKa, dMa,
            self.lf, self.lo, self.freq_eval[:5],
            zeta=0.01,
        )
        key = f"f{self.lf[0]}_o{self.lo[0]}"
        assert np.allclose(np.abs(dH[key]), 0.0, atol=1e-20)


class TestMaterialPerturbation:

    def setup_method(self):
        res = _build_rom()
        (_, _, _, _,
         _, _, self.Ka, self.Ma,
         self.lf, self.lo, self.freq_eval, self.H_nom) = res

    def test_returns_dict_with_expected_keys(self):
        result = material_perturbation_study(
            self.Ka, self.Ma, self.lf, self.lo, self.freq_eval,
            self.H_nom, param_values=[0.95, 1.0, 1.05],
            Ka_func=lambda p: p * self.Ka,
            Ma_func=lambda p: self.Ma,
            zeta=0.01, verbose=False,
        )
        assert "param_values" in result
        assert "H_sweep" in result
        assert "max_deviation_pct" in result

    def test_nominal_gives_zero_deviation(self):
        result = material_perturbation_study(
            self.Ka, self.Ma, self.lf, self.lo, self.freq_eval,
            self.H_nom, param_values=[1.0],
            Ka_func=lambda p: p * self.Ka,
            Ma_func=lambda p: p * self.Ma,
            zeta=0.01, verbose=False,
        )
        assert result["max_deviation_pct"][0] < 1e-8

    def test_sweep_shape(self):
        params = [0.9, 0.95, 1.0, 1.05, 1.1]
        result = material_perturbation_study(
            self.Ka, self.Ma, self.lf, self.lo, self.freq_eval,
            self.H_nom, param_values=params,
            Ka_func=lambda p: p * self.Ka,
            Ma_func=lambda p: self.Ma,
            zeta=0.01, verbose=False,
        )
        assert result["H_sweep"].shape == (len(params), len(self.freq_eval))


class TestMonteCarlo:

    def setup_method(self):
        res = _build_rom()
        (_, _, _, _,
         _, _, self.Ka, self.Ma,
         self.lf, self.lo, self.freq_eval, _) = res

    def test_returns_expected_keys(self):
        mc = monte_carlo_frf(
            self.Ka, self.Ma, self.lf, self.lo, self.freq_eval,
            E_cov_pct=2.0, rho_cov_pct=1.0,
            n_samples=10, seed=0, zeta=0.01, verbose=False,
        )
        for key in ("H_mean", "H_std", "H_p5", "H_p95", "H_all"):
            assert key in mc

    def test_output_shapes(self):
        n_freq = len(self.freq_eval)
        mc = monte_carlo_frf(
            self.Ka, self.Ma, self.lf, self.lo, self.freq_eval,
            n_samples=5, zeta=0.01, verbose=False,
        )
        assert mc["H_mean"].shape == (n_freq,)
        assert mc["H_all"].shape  == (5, n_freq)

    def test_std_positive(self):
        mc = monte_carlo_frf(
            self.Ka, self.Ma, self.lf, self.lo, self.freq_eval,
            E_cov_pct=5.0, n_samples=10, zeta=0.01, verbose=False,
        )
        assert np.all(mc["H_std"] >= 0)

    def test_p5_le_p95(self):
        mc = monte_carlo_frf(
            self.Ka, self.Ma, self.lf, self.lo, self.freq_eval,
            n_samples=15, zeta=0.01, verbose=False,
        )
        assert np.all(mc["H_p5"] <= mc["H_p95"] + 1e-15)

    def test_deterministic_with_seed(self):
        kw = dict(Ka_nominal=self.Ka, Ma_nominal=self.Ma,
                  local_force=self.lf, local_output=self.lo,
                  freq_eval=self.freq_eval,
                  n_samples=8, seed=42, zeta=0.01, verbose=False)
        mc1 = monte_carlo_frf(**kw)
        mc2 = monte_carlo_frf(**kw)
        assert np.allclose(mc1["H_mean"], mc2["H_mean"])
