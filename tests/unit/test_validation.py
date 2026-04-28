"""Unit tests for pyserep.analysis.validation."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.core.rom_builder import build_serep_rom, verify_eigenvalues
from pyserep.selection.dof_selector import select_dofs_eid
from pyserep.analysis.validation import (
    validate_serep,
    eigenvalue_error,
    modal_assurance_criterion,
    orthogonality_check,
    _is_positive_definite,
)


class TestValidateSEREP:

    def setup_method(self):
        from pyserep.models.synthetic import spring_chain
        K, M = spring_chain(n=60)
        self.K, self.M = K, M
        self.freqs, self.phi = solve_eigenproblem(K, M, n_modes=18, verbose=False)
        self.sel  = np.arange(1, 11)
        self.dofs, _ = select_dofs_eid(self.phi, self.sel, verbose=False)
        self.T, self.Ka, self.Ma = build_serep_rom(
            K, M, self.phi, self.sel, self.dofs, verbose=False
        )

    def test_report_returns(self):
        rep = validate_serep(
            self.K, self.M, self.phi, self.freqs,
            self.sel, self.dofs, self.T, self.Ka, self.Ma, verbose=False,
        )
        assert rep is not None

    def test_eigenvalue_error_low(self):
        rep = validate_serep(
            self.K, self.M, self.phi, self.freqs,
            self.sel, self.dofs, self.T, self.Ka, self.Ma, verbose=False,
        )
        assert rep.max_eigenvalue_error_pct < 0.01

    def test_mac_diagonal_near_one(self):
        rep = validate_serep(
            self.K, self.M, self.phi, self.freqs,
            self.sel, self.dofs, self.T, self.Ka, self.Ma, verbose=False,
        )
        assert rep.min_mac > 0.90

    def test_ma_positive_definite(self):
        rep = validate_serep(
            self.K, self.M, self.phi, self.freqs,
            self.sel, self.dofs, self.T, self.Ka, self.Ma, verbose=False,
        )
        assert rep.ma_positive_definite is True

    def test_report_passed(self):
        rep = validate_serep(
            self.K, self.M, self.phi, self.freqs,
            self.sel, self.dofs, self.T, self.Ka, self.Ma, verbose=False,
        )
        assert rep.passed()


class TestMAC:

    def test_identity_gives_ones(self):
        phi = np.eye(5)
        mac = modal_assurance_criterion(phi, phi)
        assert np.allclose(np.diag(mac), 1.0, atol=1e-10)

    def test_orthogonal_gives_zeros(self):
        # Two truly orthogonal subspaces (non-overlapping support)
        phi1 = np.zeros((6, 3)); phi1[0,0]=phi1[1,1]=phi1[2,2]=1.0
        phi2 = np.zeros((6, 3)); phi2[3,0]=phi2[4,1]=phi2[5,2]=1.0
        mac  = modal_assurance_criterion(phi1, phi2)
        assert np.allclose(mac, 0.0, atol=1e-10)

    def test_shape(self):
        A = np.random.randn(50, 8)
        B = np.random.randn(50, 8)
        mac = modal_assurance_criterion(A, B)
        assert mac.shape == (8, 8)

    def test_values_in_range(self):
        A = np.random.randn(40, 6)
        B = np.random.randn(40, 6)
        mac = modal_assurance_criterion(A, B)
        assert np.all(mac >= -1e-10) and np.all(mac <= 1.0 + 1e-10)


class TestOrthogonalityCheck:

    def test_mass_orthogonality_near_zero(self):
        from pyserep.models.synthetic import spring_chain
        K, M = spring_chain(50)
        freqs, phi = solve_eigenproblem(K, M, n_modes=10, verbose=False)
        sel = np.arange(1, 10)
        err = orthogonality_check(phi, M, sel)
        assert err < 1e-8

    def test_unnormalised_has_higher_error(self):
        from pyserep.models.synthetic import spring_chain
        K, M = spring_chain(50)
        freqs, phi = solve_eigenproblem(K, M, n_modes=10, verbose=False)
        phi_bad = phi * 5.0   # destroy mass normalisation
        sel = np.arange(1, 10)
        err_good = orthogonality_check(phi,     M, sel)
        err_bad  = orthogonality_check(phi_bad, M, sel)
        assert err_bad > err_good * 2


class TestPositiveDefinite:

    def test_identity_is_pd(self):
        assert _is_positive_definite(np.eye(5))

    def test_negative_diagonal_not_pd(self):
        A = np.diag([-1.0, 2.0, 3.0])
        assert not _is_positive_definite(A)

    def test_singular_not_pd(self):
        A = np.zeros((4, 4))
        assert not _is_positive_definite(A)
