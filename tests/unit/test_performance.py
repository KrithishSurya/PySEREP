"""Unit tests for pyserep.analysis.performance."""

import pytest

from pyserep.analysis.performance import (
    flop_count,
    reduction_metrics,
    summarise_performance,
    PerformanceMetrics,
)


class TestFlopCount:

    def test_modal_scales_with_modes(self):
        f1 = flop_count(n_modes=10,  n_freq=100, n_pairs=1, method="modal")
        f2 = flop_count(n_modes=100, n_freq=100, n_pairs=1, method="modal")
        assert f2 > f1 * 5   # 10× more modes → 10× more FLOPs

    def test_modal_scales_with_freq(self):
        f1 = flop_count(n_modes=20, n_freq=100,  n_pairs=1, method="modal")
        f2 = flop_count(n_modes=20, n_freq=1000, n_pairs=1, method="modal")
        assert f2 == f1 * 10

    def test_direct_scales_with_freq(self):
        f1 = flop_count(n_modes=20, n_freq=100,  n_pairs=1, method="direct")
        f2 = flop_count(n_modes=20, n_freq=1000, n_pairs=1, method="direct")
        assert f2 == f1 * 10

    def test_direct_cubic_in_modes(self):
        """Direct method: LU dominates → O(m³) per frequency."""
        f1 = flop_count(n_modes=10, n_freq=1, n_pairs=1, method="direct")
        f2 = flop_count(n_modes=20, n_freq=1, n_pairs=1, method="direct")
        # 2× modes → ~8× FLOPs (cubic LU dominates)
        assert f2 / f1 > 5

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            flop_count(10, 100, 1, method="unknown")

    def test_returns_positive_integer(self):
        f = flop_count(20, 200, 2, method="modal")
        assert isinstance(f, int)
        assert f > 0


class TestReductionMetrics:

    def test_full_retention(self):
        m = reduction_metrics(
            n_full_dofs=1000, n_master_dofs=1000,
            n_all_modes=100, n_selected_modes=100,
        )
        assert m["dof_retention_pct"] == pytest.approx(100.0)

    def test_zero_retention_gives_zero_percent(self):
        m = reduction_metrics(
            n_full_dofs=1000, n_master_dofs=0,
            n_all_modes=100, n_selected_modes=0,
        )
        assert m["dof_retention_pct"] == pytest.approx(0.0)

    def test_garteur_benchmark_values(self):
        """Verify Garteur benchmark metrics from thesis."""
        m = reduction_metrics(
            n_full_dofs=66525, n_master_dofs=37,
            n_all_modes=100, n_selected_modes=37,
        )
        assert m["dof_retention_pct"] == pytest.approx(37 / 66525 * 100, rel=1e-4)
        # Should be approximately 0.056%
        assert m["dof_retention_pct"] < 0.06

    def test_all_keys_present(self):
        m = reduction_metrics(1000, 50, 100, 10)
        for key in ("dof_reduction_ratio", "dof_retention_pct",
                    "mode_retention_pct", "size_ratio"):
            assert key in m

    def test_size_ratio_matches_retention(self):
        m = reduction_metrics(1000, 37, 100, 10)
        assert m["size_ratio"] == pytest.approx(m["dof_reduction_ratio"])


class TestSummarisePerformance:

    def test_returns_performance_metrics(self):
        pm = summarise_performance(
            n_full_dofs=10000, n_selected_modes=30, n_master_dofs=30,
            n_all_modes=100, kappa=25.0,
            n_freq=500, n_bands=1, n_pairs=1,
        )
        assert isinstance(pm, PerformanceMetrics)

    def test_speedup_computed(self):
        pm = summarise_performance(
            n_full_dofs=50000, n_selected_modes=30, n_master_dofs=30,
            n_all_modes=100, kappa=25.0,
            n_freq=500, n_bands=1, n_pairs=1,
            frf_method="direct",
        )
        # ROM (direct, m=30) vs reference (modal, n_modes=100)
        # Reference modal is always more FLOPs than direct with small m
        assert pm.frf_speedup > 0

    def test_summary_is_string(self):
        pm = summarise_performance(
            n_full_dofs=10000, n_selected_modes=20, n_master_dofs=20,
            n_all_modes=80, kappa=15.0,
            n_freq=200, n_bands=1, n_pairs=1,
        )
        s = pm.summary()
        assert isinstance(s, str)
        assert "10,000" in s or "10000" in s   # DOF count appears

    def test_dof_reduction_pct(self):
        pm = summarise_performance(
            n_full_dofs=100000, n_selected_modes=37, n_master_dofs=37,
            n_all_modes=100, kappa=23.0,
            n_freq=1000, n_bands=2, n_pairs=1,
        )
        assert pm.dof_reduction_pct == pytest.approx(37 / 100000 * 100, rel=1e-4)

    def test_timing_fields_default_zero(self):
        pm = summarise_performance(
            n_full_dofs=1000, n_selected_modes=10, n_master_dofs=10,
            n_all_modes=50, kappa=5.0,
            n_freq=100, n_bands=1, n_pairs=1,
        )
        assert pm.t_total_s == pytest.approx(0.0)
        assert pm.t_eigensolver_s == pytest.approx(0.0)
