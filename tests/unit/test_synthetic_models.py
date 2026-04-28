"""Unit tests for pyserep.models.synthetic."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.models.synthetic import (
    spring_chain,
    euler_beam,
    plate_2d,
    random_symmetric_pd,
    model_info,
)
from pyserep.core.eigensolver import solve_eigenproblem


class TestSpringChain:

    def test_shape(self):
        K, M = spring_chain(50)
        assert K.shape == (50, 50)
        assert M.shape == (50, 50)

    def test_symmetric(self):
        K, M = spring_chain(30)
        assert np.abs(K - K.T).max() < 1e-10
        assert np.abs(M - M.T).max() < 1e-10

    def test_sparse(self):
        K, M = spring_chain(100)
        assert sp.issparse(K) and sp.issparse(M)

    def test_eigenvalues_real_positive(self):
        K, M = spring_chain(30)
        freqs, _ = solve_eigenproblem(K, M, n_modes=10, verbose=False)
        assert np.all(freqs >= 0)

    def test_fixed_right(self):
        K1, _ = spring_chain(20, fixed_right=False)
        K2, _ = spring_chain(20, fixed_right=True)
        # Last diagonal entry differs
        assert K2[-1, -1] != K1[-1, -1]


class TestEulerBeam:

    def test_shape(self):
        K, M = euler_beam(n_elements=10)
        n_dof = 2 * 11
        assert K.shape == (n_dof, n_dof)

    def test_symmetric(self):
        K, M = euler_beam(n_elements=8)
        assert np.abs(K - K.T).max() < 1e-8

    def test_positive_natural_freqs(self):
        K, M = euler_beam(n_elements=12)
        freqs, _ = solve_eigenproblem(K, M, n_modes=5, verbose=False)
        assert np.all(freqs >= 0)


class TestPlate2D:

    def test_shape(self):
        K, M = plate_2d(nx=6, ny=6)
        n_int = 5 * 5
        assert K.shape == (n_int, n_int)

    def test_symmetric(self):
        K, M = plate_2d(nx=5, ny=5)
        assert np.abs(K - K.T).max() < 1e-8

    def test_positive_mass(self):
        _, M = plate_2d(nx=6, ny=6)
        assert M.diagonal().min() > 0


class TestRandomSymmetricPD:

    def test_positive_definite(self):
        K, M = random_symmetric_pd(n=20)
        Kd = K.toarray(); Md = M.toarray()
        # Cholesky should succeed
        np.linalg.cholesky(Kd)
        np.linalg.cholesky(Md)

    def test_symmetric(self):
        K, M = random_symmetric_pd(n=15)
        assert np.abs(K - K.T).max() < 1e-8


class TestModelInfo:

    def test_returns_string(self):
        K, M = spring_chain(50)
        info = model_info(K, M, "Test")
        assert isinstance(info, str)
        assert "50" in info
