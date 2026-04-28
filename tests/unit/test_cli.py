"""Unit tests for pyserep.cli (argument parsing and band parsing)."""

import os
import tempfile

import pytest
import scipy.io
import scipy.sparse as sp

from pyserep.cli import _build_parser, _parse_band, main


def _write_tiny_matrix(folder, name):
    scipy.io.mmwrite(
        os.path.join(folder, name),
        sp.eye(10, format="csc"),
    )


class TestParser:

    def setup_method(self):
        self.parser = _build_parser()

    def test_required_args_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            _write_tiny_matrix(d, "K.mtx")
            _write_tiny_matrix(d, "M.mtx")
            kp = os.path.join(d, "K.mtx")
            mp = os.path.join(d, "M.mtx")
            args = self.parser.parse_args(["-k", kp, "-m", mp, "-f", "3", "-o", "3"])
            assert args.stiffness == kp
            assert args.force_dofs == [3]
            assert args.output_dofs == [3]

    def test_default_frf_method(self):
        with tempfile.TemporaryDirectory() as d:
            _write_tiny_matrix(d, "K.mtx")
            _write_tiny_matrix(d, "M.mtx")
            args = self.parser.parse_args([
                "-k", os.path.join(d, "K.mtx"),
                "-m", os.path.join(d, "M.mtx"),
                "-f", "0", "-o", "0",
            ])
            assert args.frf_method == "direct"

    def test_default_dof_method(self):
        with tempfile.TemporaryDirectory() as d:
            _write_tiny_matrix(d, "K.mtx")
            _write_tiny_matrix(d, "M.mtx")
            args = self.parser.parse_args([
                "-k", os.path.join(d, "K.mtx"),
                "-m", os.path.join(d, "M.mtx"),
                "-f", "0", "-o", "0",
            ])
            assert args.dof_method == "eid"

    def test_multiple_dof_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            _write_tiny_matrix(d, "K.mtx")
            _write_tiny_matrix(d, "M.mtx")
            args = self.parser.parse_args([
                "-k", os.path.join(d, "K.mtx"),
                "-m", os.path.join(d, "M.mtx"),
                "-f", "1", "2", "3",
                "-o", "1", "2", "3",
            ])
            assert args.force_dofs == [1, 2, 3]
            assert args.output_dofs == [1, 2, 3]

    def test_bands_flag(self):
        with tempfile.TemporaryDirectory() as d:
            _write_tiny_matrix(d, "K.mtx")
            _write_tiny_matrix(d, "M.mtx")
            args = self.parser.parse_args([
                "-k", os.path.join(d, "K.mtx"),
                "-m", os.path.join(d, "M.mtx"),
                "-f", "0", "-o", "0",
                "--bands", "0,100,Low", "400,500,High",
            ])
            assert args.bands == ["0,100,Low", "400,500,High"]

    def test_no_plot_flag(self):
        with tempfile.TemporaryDirectory() as d:
            _write_tiny_matrix(d, "K.mtx")
            _write_tiny_matrix(d, "M.mtx")
            args = self.parser.parse_args([
                "-k", os.path.join(d, "K.mtx"),
                "-m", os.path.join(d, "M.mtx"),
                "-f", "0", "-o", "0",
                "--no-plot",
            ])
            assert args.no_plot is True


class TestParseBand:

    def test_two_values(self):
        from pyserep.selection.band_selector import FrequencyBand
        b = _parse_band("0,100")
        assert isinstance(b, FrequencyBand)
        assert b.f_min == pytest.approx(0.0)
        assert b.f_max == pytest.approx(100.0)

    def test_three_values_with_label(self):
        b = _parse_band("400,500,HighBand")
        assert b.label == "HighBand"

    def test_bad_format_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_band("100")   # only one value — needs at least two


class TestMainReturnCodes:

    def test_missing_file_returns_nonzero(self):
        import sys
        from io import StringIO
        # Patch sys.argv to pass a non-existent file
        old_argv = sys.argv
        sys.argv = [
            "pyserep", "-k", "/no/K.mtx", "-m", "/no/M.mtx",
            "-f", "0", "-o", "0",
        ]
        try:
            rc = main()
            assert rc == 1
        finally:
            sys.argv = old_argv
