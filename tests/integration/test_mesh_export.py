"""Integration tests for mesh export functions."""

import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

from pyserep.models.synthetic import spring_chain
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.selection.mode_selector import select_modes_pipeline
from pyserep.selection.dof_selector import select_dofs_eid
from pyserep.core.rom_builder import build_serep_rom
from pyserep.io.mesh_writer import (
    write_master_dofs_csv,
    write_master_dofs_vtk,
    write_ansys_node_list,
    write_uff58_mode_shapes,
)


@pytest.fixture(scope="module")
def rom_results():
    """Build a small ROM once for all mesh export tests."""
    N = 120
    K, M = spring_chain(N, k=2e4)
    freqs, phi = solve_eigenproblem(K, M, n_modes=20, verbose=False)
    band = FrequencyBandSet([FrequencyBand(1.0, 50.0)], n_points_per_band=50)
    sel  = select_modes_pipeline(phi, freqs, [N//2], [N//2], band, verbose=False)
    req  = np.array([N//2])
    dofs, _ = select_dofs_eid(phi, sel, required_dofs=req, verbose=False)
    T, Ka, Ma = build_serep_rom(K, M, phi, sel, dofs, verbose=False)
    coords = np.column_stack([np.arange(N), np.zeros(N), np.zeros(N)])
    return {
        "phi": phi, "freqs": freqs, "sel": sel,
        "dofs": dofs, "coords": coords, "N": N,
    }


class TestCSVExport:

    def test_csv_file_created(self, rom_results, tmp_path):
        path = str(tmp_path / "master.csv")
        write_master_dofs_csv(rom_results["dofs"], path, verbose=False)
        assert os.path.isfile(path)

    def test_csv_row_count(self, rom_results, tmp_path):
        path = str(tmp_path / "master.csv")
        write_master_dofs_csv(rom_results["dofs"], path, verbose=False)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == len(rom_results["dofs"]) + 1  # +1 header

    def test_csv_with_coords(self, rom_results, tmp_path):
        path = str(tmp_path / "master_xyz.csv")
        write_master_dofs_csv(
            rom_results["dofs"], path,
            node_coords=rom_results["coords"], verbose=False,
        )
        with open(path) as f:
            header = f.readline()
        assert "x,y,z" in header

    def test_csv_node_numbers_correct(self, rom_results, tmp_path):
        """Each row's node number must equal dof//3 + 1."""
        path = str(tmp_path / "check.csv")
        write_master_dofs_csv(rom_results["dofs"], path, verbose=False)
        with open(path) as f:
            rows = f.readlines()[1:]   # skip header
        for row, dof in zip(rows, rom_results["dofs"]):
            parts = row.strip().split(",")
            assert int(parts[0]) == int(dof)           # dof_index
            assert int(parts[1]) == int(dof) // 3 + 1  # node_number


class TestVTKExport:

    def test_vtk_created(self, rom_results, tmp_path):
        path = str(tmp_path / "master.vtk")
        write_master_dofs_vtk(
            rom_results["dofs"], rom_results["coords"], path, verbose=False
        )
        assert os.path.isfile(path)

    def test_vtk_point_count(self, rom_results, tmp_path):
        path = str(tmp_path / "count.vtk")
        n = len(rom_results["dofs"])
        write_master_dofs_vtk(
            rom_results["dofs"], rom_results["coords"], path, verbose=False
        )
        with open(path) as f:
            content = f.read()
        assert f"POINTS {n} float" in content

    def test_vtk_with_scalar_data(self, rom_results, tmp_path):
        path = str(tmp_path / "scalar.vtk")
        scalars = np.random.randn(len(rom_results["dofs"]))
        write_master_dofs_vtk(
            rom_results["dofs"], rom_results["coords"], path,
            scalar_data=scalars, scalar_name="eid_score", verbose=False,
        )
        with open(path) as f:
            content = f.read()
        assert "eid_score" in content
        assert "POINT_DATA" in content


class TestAnsysExport:

    def test_apdl_created(self, rom_results, tmp_path):
        path = str(tmp_path / "nodes.inp")
        write_ansys_node_list(rom_results["dofs"], path, verbose=False)
        assert os.path.isfile(path)

    def test_apdl_contains_nsel(self, rom_results, tmp_path):
        path = str(tmp_path / "nsel.inp")
        write_ansys_node_list(rom_results["dofs"], path, verbose=False)
        content = open(path).read()
        assert "NSEL,A,NODE,," in content

    def test_apdl_custom_component_name(self, rom_results, tmp_path):
        path = str(tmp_path / "comp.inp")
        write_ansys_node_list(
            rom_results["dofs"], path,
            component_name="MY_SENSORS", verbose=False,
        )
        content = open(path).read()
        assert "MY_SENSORS" in content

    def test_apdl_all_nodes_present(self, rom_results, tmp_path):
        path = str(tmp_path / "all.inp")
        write_ansys_node_list(rom_results["dofs"], path, verbose=False)
        content = open(path).read()
        expected_nodes = sorted(set(int(d) // 3 + 1 for d in rom_results["dofs"]))
        for node in expected_nodes:
            assert str(node) in content


class TestUFF58Export:

    def test_uff58_created(self, rom_results, tmp_path):
        path = str(tmp_path / "modes.uff")
        write_uff58_mode_shapes(
            rom_results["phi"], rom_results["freqs"],
            rom_results["sel"], rom_results["dofs"],
            node_coords=rom_results["coords"],
            path=path, verbose=False,
        )
        assert os.path.isfile(path)

    def test_uff58_contains_all_modes(self, rom_results, tmp_path):
        path = str(tmp_path / "modes_check.uff")
        write_uff58_mode_shapes(
            rom_results["phi"], rom_results["freqs"],
            rom_results["sel"], rom_results["dofs"],
            node_coords=None, path=path, verbose=False,
        )
        content = open(path).read()
        n_modes = len(rom_results["sel"])
        # Each mode block header contains its mode number label
        # Check that each selected mode has an entry (one "    58" dataset marker)
        dataset_markers = content.count("    58")
        assert dataset_markers == n_modes
