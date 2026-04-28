"""Unit tests for pyserep.frf.direct_frf — the core new feature."""

import numpy as np
import pytest

from pyserep.frf.direct_frf import (
    compute_frf_direct,
    _modal_ca,
    _rayleigh_ca,
)


def _simple_2dof():
    """2-DOF system with known analytical FRF."""
    Ka = np.array([[200.0, -100.0], [-100.0, 100.0]])
    Ma = np.array([[1.0, 0.0], [0.0, 1.0]])
    return Ka, Ma


class TestComputeFRFDirect:

    def test_output_shape(self):
        Ka, Ma = _simple_2dof()
        freqs  = np.linspace(1, 50, 200)
        _, H   = compute_frf_direct(
            Ka, Ma, [0], [0], freqs, zeta=0.01, verbose=False
        )
        assert "f0_o0" in H
        assert H["f0_o0"].shape == (200,)

    def test_output_complex(self):
        Ka, Ma = _simple_2dof()
        freqs  = np.linspace(1, 50, 100)
        _, H   = compute_frf_direct(
            Ka, Ma, [0], [0], freqs, zeta=0.01, verbose=False
        )
        assert np.iscomplexobj(H["f0_o0"])

    def test_resonance_peak_exists(self):
        """The FRF should have a clear peak near a natural frequency."""
        Ka, Ma = _simple_2dof()
        import scipy.linalg as sla
        lam = sla.eigh(Ka, Ma, eigvals_only=True)
        fn  = np.sqrt(lam) / (2 * np.pi)   # natural frequencies Hz

        freqs = np.linspace(0.5, fn[-1] * 1.5, 2000)
        _, H  = compute_frf_direct(
            Ka, Ma, [0], [0], freqs, zeta=0.005, verbose=False
        )
        mag_peak_idx = np.argmax(np.abs(H["f0_o0"]))
        peak_freq    = freqs[mag_peak_idx]

        # Peak should be near one of the natural frequencies
        min_dist = np.min(np.abs(fn - peak_freq))
        assert min_dist < 1.0, f"Peak at {peak_freq:.2f} Hz far from any natural freq {fn}"

    def test_multiple_pairs(self):
        Ka, Ma = _simple_2dof()
        freqs  = np.linspace(1, 50, 100)
        _, H   = compute_frf_direct(
            Ka, Ma, [0, 1], [0, 1], freqs, zeta=0.01, verbose=False
        )
        assert "f0_o0" in H
        assert "f1_o1" in H

    def test_reciprocity(self):
        """H_ij = H_ji (Maxwell reciprocity)."""
        Ka, Ma = _simple_2dof()
        freqs  = np.linspace(1, 50, 200)
        _, H1  = compute_frf_direct(Ka, Ma, [0], [1], freqs, zeta=0.01, verbose=False)
        _, H2  = compute_frf_direct(Ka, Ma, [1], [0], freqs, zeta=0.01, verbose=False)
        err = np.abs(H1["f0_o1"] - H2["f1_o0"]).max()
        assert err < 1e-10, f"Reciprocity violated: max error = {err:.2e}"

    def test_damping_types(self):
        Ka, Ma = _simple_2dof()
        freqs  = np.linspace(1, 50, 100)
        for dt in ("modal", "rayleigh", "hysteretic", "none"):
            _, H = compute_frf_direct(
                Ka, Ma, [0], [0], freqs,
                zeta=0.01, damping_type=dt, eta=0.01, verbose=False
            )
            assert H["f0_o0"].shape == (100,), f"Failed for {dt}"

    def test_invalid_damping_type_raises(self):
        Ka, Ma = _simple_2dof()
        with pytest.raises(ValueError, match="damping_type"):
            compute_frf_direct(Ka, Ma, [0], [0], [1.0], damping_type="bad", verbose=False)


class TestDampingMatrixBuilders:

    def test_modal_ca_shape(self):
        Ka, Ma = _simple_2dof()
        Ca = _modal_ca(Ka, Ma, 0.02)
        assert Ca.shape == (2, 2)
        assert np.abs(Ca - Ca.T).max() < 1e-10

    def test_rayleigh_ca_shape(self):
        Ka, Ma = _simple_2dof()
        Ca = _rayleigh_ca(Ka, Ma, 0.02)
        assert Ca.shape == (2, 2)
