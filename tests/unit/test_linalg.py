"""Unit tests for pyserep.utils.linalg."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.utils.linalg import (
    condition_number_estimate,
    rank_revealing_qr,
    safe_pinv,
    symmetrise,
    force_positive_definite,
    modal_strain_energy,
    modal_residues,
)
from pyserep.models.synthetic import spring_chain
from pyserep.core.eigensolver import solve_eigenproblem


class TestConditionNumber:

    def test_identity_kappa_one(self):
        A = np.eye(5)
        assert abs(condition_number_estimate(A) - 1.0) < 1e-8

    def test_known_kappa(self):
        lam = np.array([1.0, 10.0, 100.0])
        Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))
        A = Q @ np.diag(lam) @ Q.T
        kappa = condition_number_estimate(A)
        assert abs(kappa - 100.0) < 1.0

    def test_fast_vs_exact_close(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((20, 8))
        k_exact = condition_number_estimate(A, method="exact")
        k_fast  = condition_number_estimate(A, method="fast")
        ratio = max(k_exact, k_fast) / min(k_exact, k_fast)
        assert ratio < 5.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            condition_number_estimate(np.eye(3), method="bad")


class TestRankRevealingQR:

    def test_full_rank(self):
        A = np.random.default_rng(0).standard_normal((10, 5))
        Q, R, perm, rank = rank_revealing_qr(A)
        assert rank == 5

    def test_rank_deficient(self):
        A = np.zeros((8, 4))
        A[:, 0] = 1; A[:, 1] = 2    # only 2 independent columns
        Q, R, perm, rank = rank_revealing_qr(A, tol=1e-10)
        assert rank <= 2

    def test_perm_is_valid_permutation(self):
        A = np.random.default_rng(1).standard_normal((6, 4))
        _, _, perm, _ = rank_revealing_qr(A)
        assert sorted(perm) == list(range(4))


class TestSafePinv:

    def test_inverse_identity(self):
        A = np.eye(4) * 3.0
        Ainv = safe_pinv(A)
        assert np.allclose(Ainv, np.eye(4) / 3.0)

    def test_shape(self):
        A = np.random.default_rng(0).standard_normal((6, 4))
        Ainv = safe_pinv(A)
        assert Ainv.shape == (4, 6)


class TestSymmetrise:

    def test_already_symmetric(self):
        A = np.diag([1.0, 2.0, 3.0])
        B = symmetrise(A)
        assert np.allclose(A, B)

    def test_asymmetric_dense(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = symmetrise(A)
        assert np.allclose(B, B.T)

    def test_sparse(self):
        A = sp.csc_matrix([[1.0, 2.0], [3.0, 4.0]])
        B = symmetrise(A)
        assert sp.issparse(B)
        assert np.allclose((B - B.T).toarray(), 0.0)


class TestForcePositiveDefinite:

    def test_already_pd_zero_shift(self):
        A = np.diag([2.0, 3.0, 4.0])
        A_pd, shift = force_positive_definite(A)
        assert shift == 0.0
        np.linalg.cholesky(A_pd)

    def test_indefinite_becomes_pd(self):
        A = np.array([[1.0, 0.0], [0.0, -1.0]])
        A_pd, shift = force_positive_definite(A)
        assert shift > 0.0
        np.linalg.cholesky(A_pd)


class TestModalStrainEnergy:

    def test_equals_omega_squared_for_normalised(self):
        K, M = spring_chain(40)
        freqs, phi = solve_eigenproblem(K, M, n_modes=10, verbose=False)
        sel = np.arange(1, 9)
        mse = modal_strain_energy(phi, K, sel)
        omega2 = (2.0 * np.pi * freqs[sel]) ** 2
        assert np.allclose(mse, omega2, rtol=1e-4)


class TestModalResidues:

    def test_shape(self):
        phi = np.random.randn(50, 10)
        R = modal_residues(phi, [5, 10], [5, 10], np.arange(5))
        assert R.shape == (5, 2)

    def test_auto_residue(self):
        """Auto-residue Rᵢ = φᵢ(f)² is always non-negative."""
        phi = np.random.randn(50, 10)
        R = modal_residues(phi, [5], [5], np.arange(8))
        assert np.all(R >= 0)
