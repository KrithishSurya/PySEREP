"""Unit tests for all four DOF selection methods."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.selection.dof_selector import (
    select_dofs_eid,
    select_dofs_kinetic,
    select_dofs_modal_disp,
    select_dofs_svd,
    compare_dof_selectors,
)


def _chain(n=80, k=1e4):
    K = sp.diags([[-k]*(n-1), [2*k]*n, [-k]*(n-1)], [-1,0,1], format="csc").astype(float)
    K[0, 0] = k
    return K, sp.eye(n, format="csc")


SELECTORS = [select_dofs_kinetic, select_dofs_modal_disp, select_dofs_svd, select_dofs_eid]


class TestAllDOFSelectors:

    def setup_method(self):
        K, M = _chain(n=80)
        self.freqs, self.phi = solve_eigenproblem(K, M, n_modes=15, verbose=False)
        self.sel = np.arange(2, 12)   # 10 selected modes

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_output_length(self, fn):
        dofs, _ = fn(self.phi, self.sel, verbose=False)
        assert len(dofs) == len(self.sel), f"{fn.__name__}: wrong DOF count"

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_dofs_in_range(self, fn):
        dofs, _ = fn(self.phi, self.sel, verbose=False)
        N = self.phi.shape[0]
        assert np.all(dofs >= 0) and np.all(dofs < N)

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_dofs_sorted(self, fn):
        dofs, _ = fn(self.phi, self.sel, verbose=False)
        assert np.all(np.diff(dofs) > 0), f"{fn.__name__}: DOFs not sorted"

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_kappa_returned(self, fn):
        _, kappa = fn(self.phi, self.sel, verbose=False)
        assert np.isfinite(kappa) and kappa > 0

    def test_eid_better_than_kinetic(self):
        """DS4 (EID) should consistently produce lower κ than DS1 (KE)."""
        _, k_eid = select_dofs_eid(self.phi, self.sel, verbose=False)
        _, k_ke  = select_dofs_kinetic(self.phi, self.sel, verbose=False)
        assert k_eid < k_ke * 10, (
            f"EID ({k_eid:.2e}) should be better than KE ({k_ke:.2e})"
        )

    def test_compare_returns_all_methods(self):
        cmp = compare_dof_selectors(self.phi, self.sel, verbose=False)
        assert set(cmp.keys()) == {"DS1", "DS2", "DS3", "DS4"}

    def test_n_master_override(self):
        dofs, _ = select_dofs_eid(self.phi, self.sel, n_master=5, verbose=False)
        assert len(dofs) == 5
