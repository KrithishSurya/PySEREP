"""Unit tests for pyserep.pipeline.config (ROMConfig)."""

import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.pipeline.config import ROMConfig
from pyserep.selection.band_selector import FrequencyBand


def _write_matrix(folder, filename, n=10):
    """Write a trivial sparse matrix to disk for config validation."""
    import scipy.io
    K = sp.eye(n, format="csc")
    scipy.io.mmwrite(os.path.join(folder, filename), K)


class TestROMConfig:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        _write_matrix(self.tmp, "K.mtx")
        _write_matrix(self.tmp, "M.mtx")
        self.kp = os.path.join(self.tmp, "K.mtx")
        self.mp = os.path.join(self.tmp, "M.mtx")

    def test_basic_creation(self):
        cfg = ROMConfig(
            stiffness_file=self.kp,
            mass_file=self.mp,
            force_dofs=[3],
            output_dofs=[3],
        )
        assert cfg.force_dofs == [3]
        assert cfg.output_dofs == [3]

    def test_default_frf_method_is_direct(self):
        cfg = ROMConfig(stiffness_file=self.kp, mass_file=self.mp)
        assert cfg.frf_method == "direct"

    def test_default_dof_method_is_eid(self):
        cfg = ROMConfig(stiffness_file=self.kp, mass_file=self.mp)
        assert cfg.dof_method == "eid"

    def test_freq_range_creates_one_band(self):
        cfg = ROMConfig(
            stiffness_file=self.kp, mass_file=self.mp,
            freq_range=(5.0, 200.0),
        )
        assert cfg.n_bands == 1
        assert cfg.global_f_max == pytest.approx(200.0)
        assert cfg.global_f_min == pytest.approx(5.0)

    def test_explicit_bands(self):
        cfg = ROMConfig(
            stiffness_file=self.kp, mass_file=self.mp,
            bands=[FrequencyBand(0.1, 100.0), FrequencyBand(400.0, 500.0)],
        )
        assert cfg.n_bands == 2
        assert cfg.is_selective

    def test_single_band_not_selective(self):
        cfg = ROMConfig(stiffness_file=self.kp, mass_file=self.mp)
        assert not cfg.is_selective

    def test_n_pairs(self):
        cfg = ROMConfig(
            stiffness_file=self.kp, mass_file=self.mp,
            force_dofs=[1, 2, 3],
            output_dofs=[1, 2, 3],
        )
        assert cfg.n_pairs == 3

    def test_mismatched_dof_lengths_raises(self):
        with pytest.raises(ValueError, match="force_dofs length"):
            ROMConfig(
                stiffness_file=self.kp, mass_file=self.mp,
                force_dofs=[1, 2],
                output_dofs=[1],
            )

    def test_invalid_frf_method_raises(self):
        with pytest.raises(ValueError, match="frf_method"):
            ROMConfig(
                stiffness_file=self.kp, mass_file=self.mp,
                frf_method="bad_method",
            )

    def test_invalid_dof_method_raises(self):
        with pytest.raises(ValueError, match="dof_method"):
            ROMConfig(
                stiffness_file=self.kp, mass_file=self.mp,
                dof_method="random",
            )

    def test_missing_stiffness_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ROMConfig(stiffness_file="/no/such/K.mtx", mass_file=self.mp)

    def test_missing_mass_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ROMConfig(stiffness_file=self.kp, mass_file="/no/such/M.mtx")

    def test_summary_is_string(self):
        cfg = ROMConfig(stiffness_file=self.kp, mass_file=self.mp)
        s = cfg.summary()
        assert isinstance(s, str)
        assert "direct" in s.lower() or "frf" in s.lower()

    def test_global_f_max_from_bands(self):
        cfg = ROMConfig(
            stiffness_file=self.kp, mass_file=self.mp,
            bands=[
                FrequencyBand(0.0, 100.0),
                FrequencyBand(300.0, 600.0),
            ],
        )
        assert cfg.global_f_max == pytest.approx(600.0)

    def test_ms1_cutoff_derived(self):
        cfg = ROMConfig(
            stiffness_file=self.kp, mass_file=self.mp,
            freq_range=(0.0, 200.0),
            ms1_alpha=1.5,
        )
        # ms1 cutoff = alpha * global_f_max = 1.5 * 200 = 300 Hz
        assert cfg.ms1_alpha * cfg.global_f_max == pytest.approx(300.0)
