"""Unit tests for pyserep.selection.band_selector."""

import warnings

import numpy as np
import pytest

from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet


class TestFrequencyBand:

    def test_basic_creation(self):
        b = FrequencyBand(0.0, 100.0)
        assert b.f_min == 0.0
        assert b.f_max == 100.0

    def test_auto_label(self):
        b = FrequencyBand(10.0, 50.0)
        assert "10" in b.label and "50" in b.label

    def test_custom_label(self):
        b = FrequencyBand(0.0, 100.0, label="MyBand")
        assert b.label == "MyBand"

    def test_span(self):
        b = FrequencyBand(10.0, 60.0)
        assert b.span == pytest.approx(50.0)

    def test_centre(self):
        b = FrequencyBand(0.0, 100.0)
        assert b.centre == pytest.approx(50.0)

    def test_contains(self):
        b = FrequencyBand(10.0, 50.0)
        assert b.contains(10.0)
        assert b.contains(30.0)
        assert b.contains(50.0)
        assert not b.contains(9.9)
        assert not b.contains(50.1)

    def test_f_max_must_exceed_f_min(self):
        with pytest.raises(ValueError):
            FrequencyBand(50.0, 50.0)
        with pytest.raises(ValueError):
            FrequencyBand(60.0, 50.0)

    def test_negative_f_min_raises(self):
        with pytest.raises(ValueError):
            FrequencyBand(-1.0, 50.0)

    def test_expanded(self):
        b = FrequencyBand(0.0, 100.0)
        b2 = b.expanded(1.5)
        assert b2.f_max == pytest.approx(150.0)
        assert b2.f_min == pytest.approx(0.0)

    def test_frozen_immutable(self):
        b = FrequencyBand(0.0, 100.0)
        with pytest.raises(Exception):
            b.f_min = 5.0  # frozen dataclass


class TestFrequencyBandSet:

    def test_single_band(self):
        bs = FrequencyBandSet([FrequencyBand(0.0, 100.0)])
        assert bs.n_bands == 1

    def test_multi_band(self):
        bs = FrequencyBandSet([
            FrequencyBand(0.0, 50.0),
            FrequencyBand(80.0, 120.0),
        ])
        assert bs.n_bands == 2

    def test_sorted_ascending(self):
        bs = FrequencyBandSet([
            FrequencyBand(80.0, 120.0),
            FrequencyBand(0.0, 50.0),
        ])
        assert bs.bands[0].f_min < bs.bands[1].f_min

    def test_global_f_min_max(self):
        bs = FrequencyBandSet([
            FrequencyBand(10.0, 50.0),
            FrequencyBand(80.0, 120.0),
        ])
        assert bs.global_f_min == pytest.approx(10.0)
        assert bs.global_f_max == pytest.approx(120.0)

    def test_is_selective_single(self):
        bs = FrequencyBandSet([FrequencyBand(0.0, 100.0)])
        assert not bs.is_selective

    def test_is_selective_multi(self):
        bs = FrequencyBandSet([
            FrequencyBand(0.0, 50.0),
            FrequencyBand(80.0, 120.0),
        ])
        assert bs.is_selective

    def test_frequency_grid_length(self):
        bs = FrequencyBandSet([FrequencyBand(0.0, 100.0)], n_points_per_band=200)
        grid = bs.frequency_grid()
        assert len(grid) == 200

    def test_frequency_grid_multi_band(self):
        bs = FrequencyBandSet([
            FrequencyBand(0.0, 50.0),
            FrequencyBand(80.0, 100.0),
        ], n_points_per_band=100)
        grid = bs.frequency_grid()
        # Points in gap [50, 80] should not appear
        assert not np.any((grid > 50.0) & (grid < 80.0))

    def test_frequency_grid_sorted(self):
        bs = FrequencyBandSet([
            FrequencyBand(0.0, 50.0),
            FrequencyBand(80.0, 100.0),
        ])
        grid = bs.frequency_grid()
        assert np.all(np.diff(grid) >= 0)

    def test_frequency_mask(self):
        bs = FrequencyBandSet([
            FrequencyBand(10.0, 30.0),
            FrequencyBand(60.0, 80.0),
        ])
        freqs = np.array([5.0, 20.0, 45.0, 70.0, 90.0])
        mask = bs.frequency_mask(freqs)
        assert mask.tolist() == [False, True, False, True, False]

    def test_mode_passes_ms1(self):
        bs = FrequencyBandSet([FrequencyBand(0.0, 100.0)])
        assert bs.mode_passes_ms1(50.0)
        assert bs.mode_passes_ms1(100.0)
        assert bs.mode_passes_ms1(140.0, alpha=1.5)
        assert not bs.mode_passes_ms1(200.0, alpha=1.5)
        assert not bs.mode_passes_ms1(0.5, rb_hz=1.0)

    def test_overlap_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FrequencyBandSet([
                FrequencyBand(0.0, 60.0),
                FrequencyBand(50.0, 100.0),  # overlaps with first
            ])
            assert len(w) == 1
            assert "overlap" in str(w[0].message).lower()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            FrequencyBandSet([])
