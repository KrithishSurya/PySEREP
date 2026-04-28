"""Unit tests for pyserep.io (matrix_loader and exporter)."""

import json
import os
import tempfile

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from pyserep.io.matrix_loader import load_matrix, load_matrices, enforce_symmetry
from pyserep.io.exporter import save_results, load_frf_npz, load_metrics
from pyserep.io.mesh_writer import (
    write_master_dofs_csv,
    write_master_dofs_vtk,
    write_ansys_node_list,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_sparse(n=30):
    """Small tridiagonal sparse matrix."""
    K = sp.diags([[-1.0]*(n-1), [2.0]*n, [-1.0]*(n-1)], [-1,0,1], format="csc")
    K[0,0] = 1.0
    return K


def _save_mtx(mat, path):
    scipy.io.mmwrite(path, mat)


# ─────────────────────────────────────────────────────────────────────────────
# load_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadMatrix:

    def test_mtx_roundtrip(self):
        K = _make_sparse(20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "K.mtx")
            _save_mtx(K, path)
            K2 = load_matrix(path)
            assert sp.issparse(K2)
            assert K2.shape == K.shape
            assert abs(K2 - K).max() < 1e-12

    def test_npz_roundtrip(self):
        K = _make_sparse(25)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "K.npz")
            sp.save_npz(path, K)
            K2 = load_matrix(path)
            assert K2.shape == K.shape
            assert abs(K2 - K).max() < 1e-12

    def test_npy_roundtrip(self):
        A = np.eye(10)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "A.npy")
            np.save(path, A)
            A2 = load_matrix(path)
            assert sp.issparse(A2)
            assert np.allclose(A2.toarray(), A)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_matrix("/nonexistent/path/K.mtx")

    def test_returns_csc(self):
        K = _make_sparse(15)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "K.mtx")
            _save_mtx(K, path)
            K2 = load_matrix(path)
            assert isinstance(K2, sp.csc_matrix)

    def test_severe_asymmetry_raises(self):
        """Matrices with >1% asymmetry raise ValueError."""
        with tempfile.TemporaryDirectory() as d:
            A = np.array([[2.0, 3.0], [1.0, 2.0]])   # ~67% asymmetry
            path = os.path.join(d, "A.mtx")
            scipy.io.mmwrite(path, sp.csc_matrix(A))
            with pytest.raises(ValueError, match="asymmetry"):
                load_matrix(path, symmetry_tol=1e-10)

    def test_mild_asymmetry_warns(self):
        """Matrices with small but non-trivial asymmetry issue a UserWarning."""
        with tempfile.TemporaryDirectory() as d:
            # Create matrix with ~0.1% asymmetry (> tol but < 1%)
            A = np.diag([2.0, 3.0, 4.0]).astype(float)
            A[0, 1] += 0.001   # tiny off-diagonal perturbation
            A[1, 0] -= 0.001
            path = os.path.join(d, "B.mtx")
            scipy.io.mmwrite(path, sp.csc_matrix(A))
            with pytest.warns(UserWarning, match="asymmetry"):
                load_matrix(path, symmetry_tol=1e-10)


class TestLoadMatrices:

    def test_pair_shapes_match(self):
        n = 20
        K = _make_sparse(n)
        M = sp.eye(n, format="csc")
        with tempfile.TemporaryDirectory() as d:
            kp = os.path.join(d, "K.mtx")
            mp = os.path.join(d, "M.mtx")
            _save_mtx(K, kp); _save_mtx(M, mp)
            K2, M2 = load_matrices(kp, mp, verbose=False)
            assert K2.shape == M2.shape == (n, n)

    def test_incompatible_shapes_raise(self):
        K = _make_sparse(10)
        M = sp.eye(15, format="csc")
        with tempfile.TemporaryDirectory() as d:
            kp = os.path.join(d, "K.mtx")
            mp = os.path.join(d, "M.mtx")
            _save_mtx(K, kp); _save_mtx(M, mp)
            with pytest.raises(ValueError, match="incompatible"):
                load_matrices(kp, mp, verbose=False)


class TestEnforceSymmetry:

    def test_already_symmetric_unchanged(self):
        A = sp.csc_matrix(np.diag([1.0, 2.0, 3.0]))
        B = enforce_symmetry(A)
        assert np.allclose(A.toarray(), B.toarray())

    def test_asymmetric_becomes_symmetric(self):
        data = np.array([[2.0, 3.0], [1.0, 4.0]])
        A = sp.csc_matrix(data)
        B = enforce_symmetry(A)
        assert np.allclose(B.toarray(), B.toarray().T)


# ─────────────────────────────────────────────────────────────────────────────
# Exporter
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveAndLoadFRF:

    def test_npz_roundtrip(self):
        freqs = np.linspace(1, 100, 50)
        H_rom = {"f0_o0": np.random.randn(50) + 1j * np.random.randn(50)}
        H_ref = {"f0_o0": np.random.randn(50) + 1j * np.random.randn(50)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "frf.npz")
            np.savez(path, freqs_hz=freqs,
                     **{f"rom_{k}": v for k, v in H_rom.items()},
                     **{f"ref_{k}": v for k, v in H_ref.items()})
            data = load_frf_npz(path)
            assert "freqs_hz" in data
            assert "rom_f0_o0" in data
            assert data["freqs_hz"].shape == (50,)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh writer
# ─────────────────────────────────────────────────────────────────────────────

class TestMeshWriter:

    def setup_method(self):
        self.dofs = np.array([0, 3, 6, 9, 3000])
        self.coords = np.random.randn(1001, 3)

    def test_csv_created(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dofs.csv")
            write_master_dofs_csv(self.dofs, path, verbose=False)
            assert os.path.isfile(path)
            with open(path) as f:
                lines = f.readlines()
            assert lines[0].startswith("dof_index")
            assert len(lines) == len(self.dofs) + 1  # header + rows

    def test_csv_with_coords(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dofs_coords.csv")
            write_master_dofs_csv(self.dofs, path,
                                   node_coords=self.coords, verbose=False)
            with open(path) as f:
                header = f.readline()
            assert "x" in header and "y" in header and "z" in header

    def test_vtk_created(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dofs.vtk")
            write_master_dofs_vtk(self.dofs, self.coords, path, verbose=False)
            assert os.path.isfile(path)
            with open(path) as f:
                content = f.read()
            assert "vtk DataFile" in content
            assert "UNSTRUCTURED_GRID" in content

    def test_ansys_apdl_created(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nodes.inp")
            write_ansys_node_list(self.dofs, path, verbose=False)
            assert os.path.isfile(path)
            with open(path) as f:
                content = f.read()
            assert "NSEL" in content
            assert "CM," in content

    def test_ansys_correct_node_numbers(self):
        """DOF 3000 = node 1001, direction UX."""
        dofs = np.array([3000])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nodes.inp")
            write_ansys_node_list(dofs, path, verbose=False)
            content = open(path).read()
            assert "1001" in content
