"""Unit tests for pyserep.core.rom_builder."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.core.rom_builder import build_serep_rom, verify_eigenvalues
from pyserep.selection.dof_selector import select_dofs_eid


def _spring_chain(n=60, k=1e4, m=1.0):
    K = sp.diags([[-k]*(n-1), [2*k]*n, [-k]*(n-1)], [-1,0,1], format="csc").astype(float)
    K[0,0] = k
    return K, sp.eye(n, format="csc") * m


class TestBuildSereпROM:

    def setup_method(self):
        n_modes = 10
        self.K, self.M = _spring_chain(n=60)
        self.freqs, self.phi = solve_eigenproblem(
            self.K, self.M, n_modes=n_modes, verbose=False
        )
        self.sel  = np.arange(1, n_modes)   # skip mode 0 (lowest)
        self.dofs, self.kappa = select_dofs_eid(
            self.phi, self.sel, verbose=False
        )

    def test_output_shapes(self):
        m = len(self.sel)
        T, Ka, Ma = build_serep_rom(
            self.K, self.M, self.phi, self.sel, self.dofs, verbose=False
        )
        assert T.shape  == (60, m)
        assert Ka.shape == (m, m)
        assert Ma.shape == (m, m)

    def test_ka_symmetric(self):
        _, Ka, _ = build_serep_rom(
            self.K, self.M, self.phi, self.sel, self.dofs, verbose=False
        )
        assert np.abs(Ka - Ka.T).max() < 1e-10

    def test_ma_symmetric(self):
        _, _, Ma = build_serep_rom(
            self.K, self.M, self.phi, self.sel, self.dofs, verbose=False
        )
        assert np.abs(Ma - Ma.T).max() < 1e-10

    def test_eigenvalue_preservation(self):
        _, Ka, Ma = build_serep_rom(
            self.K, self.M, self.phi, self.sel, self.dofs, verbose=False
        )
        _, max_err = verify_eigenvalues(Ka, Ma, self.freqs, self.sel, verbose=False)
        assert max_err < 0.01, f"Max eigenvalue error {max_err:.4f}% exceeds 0.01%"

    def test_ma_positive_definite(self):
        _, _, Ma = build_serep_rom(
            self.K, self.M, self.phi, self.sel, self.dofs, verbose=False
        )
        try:
            np.linalg.cholesky(Ma)
        except np.linalg.LinAlgError:
            pytest.fail("Mₐ is not positive definite")
