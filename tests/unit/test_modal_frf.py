"""Unit tests for pyserep.frf.modal_frf."""

import numpy as np
import pytest

from pyserep.models.synthetic import spring_chain
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.frf.modal_frf import compute_frf_modal, compute_frf_modal_reference


def _model():
    K, M = spring_chain(n=50)
    freqs, phi = solve_eigenproblem(K, M, n_modes=15, verbose=False)
    return freqs, phi


class TestComputeFRFModal:

    def setup_method(self):
        self.freqs, self.phi = _model()
        self.band = FrequencyBandSet([FrequencyBand(1.0, 60.0)], n_points_per_band=100)
        self.modes = np.arange(1, 12)
        self.fdofs = [25]
        self.odofs = [25]

    def test_output_shape(self):
        freq_eval, H = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            self.fdofs, self.odofs, self.band,
            zeta=0.01, verbose=False,
        )
        key = f"f{self.fdofs[0]}_o{self.odofs[0]}"
        assert key in H
        assert H[key].shape == (len(freq_eval),)

    def test_complex_output(self):
        _, H = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            self.fdofs, self.odofs, self.band,
            zeta=0.01, verbose=False,
        )
        key = f"f{self.fdofs[0]}_o{self.odofs[0]}"
        assert np.iscomplexobj(H[key])

    def test_reciprocity(self):
        """H(f→o) ≈ H(o→f) for proportional damping."""
        _, H_fo = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            [25], [30], self.band, zeta=0.01, verbose=False,
        )
        _, H_of = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            [30], [25], self.band, zeta=0.01, verbose=False,
        )
        assert np.allclose(H_fo["f25_o30"], H_of["f30_o25"], rtol=1e-10)

    def test_peak_near_natural_frequency(self):
        """FRF magnitude should peak near the natural frequencies."""
        freq_eval, H = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            self.fdofs, self.odofs, self.band,
            zeta=0.01, verbose=False,
        )
        key = f"f{self.fdofs[0]}_o{self.odofs[0]}"
        peak_freq = freq_eval[np.argmax(np.abs(H[key]))]
        f_sel = self.freqs[self.modes]
        min_dist = np.min(np.abs(f_sel - peak_freq))
        assert min_dist < 5.0   # peak within 5 Hz of a natural frequency

    def test_multiple_bands_grid_has_gaps(self):
        multi = FrequencyBandSet([
            FrequencyBand(1.0, 20.0),
            FrequencyBand(40.0, 60.0),
        ], n_points_per_band=50)
        freq_eval, _ = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            self.fdofs, self.odofs, multi,
            zeta=0.01, verbose=False,
        )
        # No frequencies in the gap (20, 40) Hz
        assert not np.any((freq_eval > 20.0) & (freq_eval < 40.0))

    def test_low_damping_gives_higher_peak(self):
        """Lower damping ratio should produce a higher FRF peak at resonance."""
        _, H_hi = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            self.fdofs, self.odofs, self.band,
            zeta=0.10, verbose=False,
        )
        _, H_lo = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            self.fdofs, self.odofs, self.band,
            zeta=0.001, verbose=False,
        )
        key = f"f{self.fdofs[0]}_o{self.odofs[0]}"
        assert np.abs(H_lo[key]).max() > np.abs(H_hi[key]).max()

    def test_per_mode_zeta(self):
        zeta_vec = np.linspace(0.005, 0.05, len(self.freqs))
        _, H = compute_frf_modal(
            self.phi, self.freqs, self.modes,
            self.fdofs, self.odofs, self.band,
            per_mode_zeta=zeta_vec, verbose=False,
        )
        key = f"f{self.fdofs[0]}_o{self.odofs[0]}"
        assert H[key].shape[0] > 0


class TestComputeFRFModalReference:

    def setup_method(self):
        self.freqs, self.phi = _model()
        self.band = FrequencyBandSet([FrequencyBand(1.0, 60.0)], n_points_per_band=80)

    def test_excludes_rigid_body(self):
        """Reference should only use elastic modes (f > rb_hz)."""
        H_ref = compute_frf_modal_reference(
            self.phi, self.freqs, rb_hz=1.0,
            force_dofs=[25], output_dofs=[25],
            band_set=self.band, zeta=0.01, verbose=False,
        )
        key = "f25_o25"
        assert key in H_ref
        assert H_ref[key].shape[0] > 0

    def test_output_is_dict(self):
        H_ref = compute_frf_modal_reference(
            self.phi, self.freqs, rb_hz=1.0,
            force_dofs=[25], output_dofs=[25],
            band_set=self.band, zeta=0.01, verbose=False,
        )
        assert isinstance(H_ref, dict)
