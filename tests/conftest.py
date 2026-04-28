"""
tests/conftest.py
=================
Shared pytest fixtures used by both unit and integration tests.
"""

from __future__ import annotations
import pytest
import numpy as np
import scipy.sparse as sp

from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.models.synthetic import spring_chain, euler_beam


# ─────────────────────────────────────────────────────────────────────────────
# Small spring chain (fast: used by most unit tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_chain_matrices():
    """60-DOF spring-mass chain — K and M."""
    return spring_chain(n=60, k=1e4)


@pytest.fixture(scope="session")
def small_chain_modes(small_chain_matrices):
    """Pre-solved eigenproblem for the 60-DOF chain."""
    K, M = small_chain_matrices
    freqs, phi = solve_eigenproblem(K, M, n_modes=20, verbose=False)
    return freqs, phi


@pytest.fixture(scope="session")
def small_chain_selected(small_chain_modes):
    """A fixed selection of modes for DS tests."""
    freqs, phi = small_chain_modes
    return np.arange(1, 12)   # 11 modes


# ─────────────────────────────────────────────────────────────────────────────
# Medium chain (used by integration tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def medium_chain_matrices():
    """300-DOF spring-mass chain."""
    return spring_chain(n=300, k=5e4)


@pytest.fixture(scope="session")
def medium_chain_modes(medium_chain_matrices):
    """Pre-solved eigenproblem for the 300-DOF chain."""
    K, M = medium_chain_matrices
    freqs, phi = solve_eigenproblem(K, M, n_modes=40, verbose=False)
    return freqs, phi


# ─────────────────────────────────────────────────────────────────────────────
# Simple 2-DOF system (analytical checks)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def two_dof_system():
    """2-DOF spring-mass system with known analytical natural frequencies."""
    Ka = np.array([[200.0, -100.0], [-100.0, 100.0]])
    Ma = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Natural frequencies: ω = sqrt(eig(Ka, Ma)) / (2π)
    import scipy.linalg as la
    lam = la.eigh(Ka, Ma, eigvals_only=True)
    freqs_hz = np.sqrt(np.maximum(lam, 0)) / (2 * np.pi)
    return Ka, Ma, freqs_hz


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def freq_array():
    """Standard FRF evaluation grid: 1–150 Hz, 300 points."""
    return np.linspace(1.0, 150.0, 300)
