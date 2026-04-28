"""Unit tests for pyserep.core.eigensolver."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.core.eigensolver import solve_eigenproblem, _mass_normalise


def _make_spring_chain(n: int = 50, k: float = 1e5, m: float = 1.0):
    """Build a simple 1D spring-mass chain as sparse CSC matrices."""
    K = sp.diags(
        [[-k] * (n - 1), [2 * k] * n, [-k] * (n - 1)],
        [-1, 0, 1], format="csc"
    ).astype(float)
    K[0, 0] = k  # fix one end
    M = sp.eye(n, format="csc") * m
    return K, M


class TestSolveEigenproblem:

    def test_output_shapes(self):
        K, M = _make_spring_chain(40)
        freqs, phi = solve_eigenproblem(K, M, n_modes=10, verbose=False)
        assert freqs.shape == (10,), "Wrong freq shape"
        assert phi.shape   == (40, 10), "Wrong phi shape"

    def test_frequencies_sorted_ascending(self):
        K, M = _make_spring_chain(40)
        freqs, _ = solve_eigenproblem(K, M, n_modes=10, verbose=False)
        assert np.all(np.diff(freqs) >= -1e-10), "Frequencies not sorted"

    def test_frequencies_non_negative(self):
        K, M = _make_spring_chain(40)
        freqs, _ = solve_eigenproblem(K, M, n_modes=10, verbose=False)
        assert np.all(freqs >= 0), "Negative natural frequencies"

    def test_mass_orthogonality(self):
        K, M = _make_spring_chain(40)
        _, phi = solve_eigenproblem(K, M, n_modes=15, verbose=False)
        orth = phi.T @ (M @ phi)
        err  = np.abs(orth - np.eye(15)).max()
        assert err < 1e-8, f"Mass orthogonality error: {err:.2e}"

    def test_raises_for_too_many_modes(self):
        K, M = _make_spring_chain(10)
        with pytest.raises(ValueError, match="n_modes"):
            solve_eigenproblem(K, M, n_modes=10, verbose=False)


class TestMassNormalise:

    def test_normalisation(self):
        K, M = _make_spring_chain(30)
        _, phi_raw = solve_eigenproblem(K, M, n_modes=8, verbose=False)
        # Manually un-normalise
        phi_scaled = phi_raw * 5.0
        phi_norm   = _mass_normalise(phi_scaled, M)
        orth = phi_norm.T @ (M @ phi_norm)
        err  = np.abs(orth - np.eye(8)).max()
        assert err < 1e-8
