"""Unit tests for the mode selection pipeline."""

import numpy as np
import pytest

from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.selection.mode_selector import (
    ms1_frequency_range,
    ms2_participation_factor,
    ms3_spatial_amplitude,
    mac_filter,
    select_modes_pipeline,
    select_modes,
)


class TestMS1FrequencyRange:

    def test_excludes_rigid_body(self, small_chain_modes):
        freqs, phi = small_chain_modes
        band_set = FrequencyBandSet([FrequencyBand(0.1, 50.0)])
        sel = ms1_frequency_range(freqs, band_set, rb_hz=1.0, verbose=False)
        assert all(freqs[i] > 1.0 for i in sel)

    def test_respects_alpha_cutoff(self, small_chain_modes):
        freqs, phi = small_chain_modes
        band_set = FrequencyBandSet([FrequencyBand(0.1, 30.0)])
        sel = ms1_frequency_range(freqs, band_set, ms1_alpha=1.0, rb_hz=0.5, verbose=False)
        assert all(freqs[i] <= 30.0 for i in sel)

    def test_returns_indices_sorted(self, small_chain_modes):
        freqs, phi = small_chain_modes
        band_set = FrequencyBandSet([FrequencyBand(0.1, 60.0)])
        sel = ms1_frequency_range(freqs, band_set, verbose=False)
        assert np.all(np.diff(sel) >= 0)


class TestMACFilter:

    def test_removes_nothing_when_threshold_1(self, small_chain_modes):
        freqs, phi = small_chain_modes
        idx = np.arange(5, 15)
        result = mac_filter(phi, idx, freqs, [30], [30], mac_threshold=1.0, verbose=False)
        assert len(result) == len(idx)

    def test_never_increases_count(self, small_chain_modes):
        freqs, phi = small_chain_modes
        idx = np.arange(0, 18)
        result = mac_filter(phi, idx, freqs, [30], [30], mac_threshold=0.95, verbose=False)
        assert len(result) <= len(idx)

    def test_output_subset_of_input(self, small_chain_modes):
        freqs, phi = small_chain_modes
        idx = np.arange(2, 14)
        result = mac_filter(phi, idx, freqs, [30], [30], mac_threshold=0.9, verbose=False)
        assert all(r in idx for r in result)


class TestSelectModes:

    def test_functional_wrapper(self, small_chain_matrices, small_chain_modes):
        K, M = small_chain_matrices
        freqs, phi = small_chain_modes
        sel = select_modes(phi, freqs, force_dofs=[30], output_dofs=[30],
                           f_max=50.0, verbose=False)
        assert len(sel) > 0

    def test_selective_bands_fewer_than_full(self, small_chain_matrices, small_chain_modes):
        K, M = small_chain_matrices
        freqs, phi = small_chain_modes
        from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
        full_band = FrequencyBandSet([FrequencyBand(1.0, 60.0)])
        narrow_band = FrequencyBandSet([FrequencyBand(1.0, 20.0)])
        sel_full = select_modes(phi, freqs, [30], [30], band_set=full_band, verbose=False)
        sel_narrow = select_modes(phi, freqs, [30], [30], band_set=narrow_band, verbose=False)
        assert len(sel_narrow) <= len(sel_full)

    def test_pipeline_modes_sorted(self, small_chain_modes):
        freqs, phi = small_chain_modes
        band_set = FrequencyBandSet([FrequencyBand(1.0, 50.0)])
        sel = select_modes_pipeline(phi, freqs, [30], [30], band_set, verbose=False)
        assert np.all(np.diff(sel) > 0), "Selected modes not strictly sorted"
