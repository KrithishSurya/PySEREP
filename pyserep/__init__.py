"""
pyserep — System Equivalent Reduction Expansion Process
==========================================================

A production-grade, publication-ready Python library for SEREP-based
Reduced Order Modelling of large-scale structural dynamics problems.

Quick start (Pipeline API)
--------------------------
>>> from pyserep import SereпPipeline, ROMConfig, FrequencyBand
>>> cfg = ROMConfig(
...     stiffness_file="K.mtx",
...     mass_file="M.mtx",
...     force_dofs=[3000],
...     output_dofs=[3000],
...     bands=[FrequencyBand(0, 100), FrequencyBand(400, 500)],
... )
>>> results = SereпPipeline(cfg).run()

Quick start (Functional API)
-----------------------------
>>> from pyserep import load_matrices, solve_eigenproblem
>>> from pyserep import select_modes, select_dofs_eid
>>> from pyserep import build_serep_rom
>>> from pyserep import compute_frf_direct, compute_frf_modal
>>> K, M = load_matrices("K.mtx", "M.mtx")
>>> freqs, phi = solve_eigenproblem(K, M, n_modes=100)
>>> modes = select_modes(phi, freqs, force_dofs=[3000], output_dofs=[3000], f_max=500.0)
>>> dofs, kappa = select_dofs_eid(phi, modes)
>>> T, Ka, Ma = build_serep_rom(K, M, phi, modes, dofs)
>>> frf = compute_frf_direct(Ka, Ma, zeta=0.01, f_eval=range(1, 501))
"""

from importlib.metadata import version as _version

# Convergence analysis
from pyserep.analysis.convergence import (
    ConvergencePoint,
    ConvergenceStudy,
    dof_count_study,
    mode_count_study,
)
from pyserep.analysis.performance import (
    PerformanceMetrics,
    flop_count,
    reduction_metrics,
    summarise_performance,
)

# Analysis extras
from pyserep.analysis.sensitivity import (
    eigenvalue_sensitivity,
    frf_sensitivity,
    material_perturbation_study,
    monte_carlo_frf,
)

# Analysis
from pyserep.analysis.validation import (
    ValidationReport,
    eigenvalue_error,
    modal_assurance_criterion,
    orthogonality_check,
    validate_serep,
)

# ── Public re-exports ─────────────────────────────────────────────────────────
# Core
from pyserep.core.eigensolver import solve_eigenproblem
from pyserep.core.rom_builder import (
    build_rayleigh_damping,
    build_serep_rom,
    verify_eigenvalues,
)

# FRF
from pyserep.frf.direct_frf import (
    FRFResult,
    compute_frf_direct,
    compute_frf_direct_fullmodel,
    compute_frf_pair_direct,
)
from pyserep.frf.modal_frf import compute_frf_modal, compute_frf_modal_reference
from pyserep.io.exporter import load_frf_npz, load_metrics, load_reduced_matrices, save_results

# I/O
from pyserep.io.matrix_loader import enforce_symmetry, load_matrices, load_matrix

# I/O extras
from pyserep.io.mesh_writer import (
    write_ansys_node_list,
    write_master_dofs_csv,
    write_master_dofs_vtk,
    write_uff58_mode_shapes,
)

# Models (built-in synthetic test models)
from pyserep.models.synthetic import (
    euler_beam,
    model_info,
    plate_2d,
    random_symmetric_pd,
    spring_chain,
)

# Pipeline
from pyserep.pipeline.config import ROMConfig
from pyserep.pipeline.serep_pipeline import PipelineResults, SereпPipeline

# Selection
from pyserep.selection.band_selector import FrequencyBand, FrequencyBandSet
from pyserep.selection.dof_selector import (
    compare_dof_selectors,
    select_dofs_eid,  # DS4 — Effective Independence (recommended)
    select_dofs_kinetic,  # DS1 — Kinetic Energy
    select_dofs_modal_disp,  # DS2 — Peak modal displacement
    select_dofs_svd,  # DS3 — SVD-based
)
from pyserep.selection.mode_selector import (
    mac_filter,
    ms1_frequency_range,
    ms2_participation_factor,
    ms3_spatial_amplitude,
    ms4_conditioning_check,
    select_modes,
    select_modes_pipeline,
)

# Utils
from pyserep.utils.linalg import (
    condition_number_estimate,
    force_positive_definite,
    mass_normalise,
    modal_residues,
    modal_strain_energy,
    rank_revealing_qr,
    safe_pinv,
    sparse_submatrix,
    symmetrise,
)
from pyserep.utils.sparse_ops import (
    ansys_dof,
    apply_bcs,
    bandwidth,
    build_dof_map,
    diagonal_scaling,
    dof_to_ansys,
    is_diagonal,
    matrix_stats,
    memory_mb,
    reorder_rcm,
    sparsity,
    unscale_modes,
)

# Visualization
from pyserep.visualization.frf_plots import plot_frf_comparison, plot_frf_overlay
from pyserep.visualization.mode_plots import (
    plot_dof_selector_comparison,
    plot_mac_matrix,
    plot_mode_shapes,
)
from pyserep.visualization.summary_plots import plot_performance_dashboard

try:
    __version__ = _version("pyserep")
except Exception:
    __version__ = "3.0.0-dev"

__all__ = [
    # Core
    "solve_eigenproblem",
    "build_serep_rom",
    "verify_eigenvalues",
    "build_rayleigh_damping",
    # I/O
    "load_matrix",
    "load_matrices",
    "enforce_symmetry",
    "check_symmetric_pd",
    "save_results",
    "load_frf_npz",
    "load_metrics",
    "load_reduced_matrices",
    "write_master_dofs_csv",
    "write_master_dofs_vtk",
    "write_ansys_node_list",
    "write_uff58_mode_shapes",
    # Selection
    "FrequencyBand",
    "FrequencyBandSet",
    "select_modes",
    "select_modes_pipeline",
    "ms1_frequency_range",
    "ms2_participation_factor",
    "ms3_spatial_amplitude",
    "mac_filter",
    "ms4_conditioning_check",
    "select_dofs_eid",
    "select_dofs_kinetic",
    "select_dofs_modal_disp",
    "select_dofs_svd",
    "compare_dof_selectors",
    # FRF
    "compute_frf_direct",
    "compute_frf_direct_fullmodel",
    "compute_frf_pair_direct",
    "compute_frf_modal",
    "compute_frf_modal_reference",
    "FRFResult",
    # Analysis — validation
    "validate_serep",
    "ValidationReport",
    "eigenvalue_error",
    "modal_assurance_criterion",
    "orthogonality_check",
    # Analysis — performance
    "summarise_performance",
    "flop_count",
    "reduction_metrics",
    "PerformanceMetrics",
    # Analysis — convergence
    "mode_count_study",
    "dof_count_study",
    "ConvergenceStudy",
    "ConvergencePoint",
    # Analysis — sensitivity
    "eigenvalue_sensitivity",
    "frf_sensitivity",
    "material_perturbation_study",
    "monte_carlo_frf",
    # Visualization
    "plot_frf_comparison",
    "plot_frf_overlay",
    "plot_mode_shapes",
    "plot_mac_matrix",
    "plot_dof_selector_comparison",
    "plot_performance_dashboard",
    # Pipeline
    "ROMConfig",
    "SereпPipeline",
    "PipelineResults",
    # Models
    "spring_chain",
    "euler_beam",
    "plate_2d",
    "random_symmetric_pd",
    "model_info",
    # Utils — linalg
    "condition_number_estimate",
    "rank_revealing_qr",
    "safe_pinv",
    "mass_normalise",
    "symmetrise",
    "force_positive_definite",
    "modal_strain_energy",
    "sparse_submatrix",
    "modal_residues",
    # Utils — sparse ops
    "memory_mb",
    "sparsity",
    "is_diagonal",
    "bandwidth",
    "matrix_stats",
    "diagonal_scaling",
    "unscale_modes",
    "apply_bcs",
    "reorder_rcm",
    "ansys_dof",
    "dof_to_ansys",
    "build_dof_map",
]
