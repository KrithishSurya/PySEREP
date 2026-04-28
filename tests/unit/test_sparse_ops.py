"""Unit tests for pyserep.utils.sparse_ops."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.utils.sparse_ops import (
    memory_mb, sparsity, is_diagonal, bandwidth,
    diagonal_scaling, apply_bcs, reorder_rcm,
    ansys_dof, dof_to_ansys, build_dof_map,
)
from pyserep.models.synthetic import spring_chain


class TestMemoryAndSparsity:

    def test_memory_positive(self):
        K, _ = spring_chain(50)
        assert memory_mb(K) > 0

    def test_sparsity_tridiagonal(self):
        K, _ = spring_chain(100)
        s = sparsity(K)
        # 100×100 tridiagonal: nnz ≈ 298, sparsity ≈ 0.97
        assert s > 0.95

    def test_sparsity_dense(self):
        A = sp.csc_matrix(np.ones((10, 10)))
        assert sparsity(A) == 0.0


class TestIsDiagonal:

    def test_lumped_mass_true(self):
        M = sp.eye(20, format="csc")
        assert is_diagonal(M)

    def test_consistent_mass_false(self):
        _, M = spring_chain(10)   # consistent mass here is identity but let's test the chain K
        K, _ = spring_chain(10)
        assert not is_diagonal(K)


class TestBandwidth:

    def test_diagonal_zero_bandwidth(self):
        A = sp.eye(10, format="csc")
        assert bandwidth(A) == 0

    def test_tridiagonal_bandwidth_one(self):
        K, _ = spring_chain(10)
        assert bandwidth(K) <= 1


class TestDiagonalScaling:

    def test_output_shapes(self):
        K, M = spring_chain(30)
        Ks, Ms, d = diagonal_scaling(K, M)
        assert Ks.shape == K.shape
        assert Ms.shape == M.shape
        assert len(d) == K.shape[0]

    def test_scaled_M_diagonal_approx_identity(self):
        """For lumped (diagonal) mass, scaled M should be near identity."""
        n = 20
        K = spring_chain(n)[0]
        M = sp.eye(n, format="csc") * 2.0
        _, Ms, _ = diagonal_scaling(K, M)
        err = np.abs(Ms - sp.eye(n)).max()
        assert err < 1e-10

    def test_symmetric(self):
        K, M = spring_chain(20)
        Ks, Ms, _ = diagonal_scaling(K, M)
        assert np.abs(Ks - Ks.T).max() < 1e-10
        assert np.abs(Ms - Ms.T).max() < 1e-10


class TestApplyBCs:

    def test_penalty_applied(self):
        K, M = spring_chain(10)
        K0 = K[0, 0].copy()
        K_bc, M_bc = apply_bcs(K, M, [0], penalty=1e10)
        assert K_bc[0, 0] > K0

    def test_shape_preserved(self):
        K, M = spring_chain(20)
        K_bc, M_bc = apply_bcs(K, M, [0, 5, 19])
        assert K_bc.shape == K.shape


class TestReorderRCM:

    def test_output_shapes(self):
        K, M = spring_chain(30)
        Kr, Mr, perm = reorder_rcm(K, M)
        assert Kr.shape == K.shape
        assert len(perm) == K.shape[0]

    def test_perm_is_permutation(self):
        K, M = spring_chain(20)
        _, _, perm = reorder_rcm(K, M)
        assert sorted(perm) == list(range(20))


class TestAnsysDOF:

    def test_node_1_ux(self):
        assert ansys_dof(1, 0) == 0

    def test_node_1001_ux(self):
        assert ansys_dof(1001, 0) == 3000

    def test_node_1001_uz(self):
        assert ansys_dof(1001, 2) == 3002

    def test_round_trip(self):
        for node in [1, 5, 100, 1001]:
            for dirn in [0, 1, 2]:
                dof = ansys_dof(node, dirn)
                n2, d2 = dof_to_ansys(dof)
                assert n2 == node and d2 == dirn

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            ansys_dof(1, 3)


class TestBuildDofMap:

    def test_valid_mapping(self):
        master = np.array([10, 20, 30, 40, 50])
        lf, lo = build_dof_map(master, [20, 40], [20, 40])
        assert lf == [1, 3]
        assert lo == [1, 3]

    def test_missing_dof_raises(self):
        master = np.array([10, 20, 30])
        with pytest.raises(KeyError):
            build_dof_map(master, [99], [99])
