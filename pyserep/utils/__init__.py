"""pyserep.utils — timing, linear algebra, and sparse matrix utilities."""
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
from pyserep.utils.timers import Timer

__all__ = [
    "Timer",
    "condition_number_estimate","rank_revealing_qr","safe_pinv",
    "mass_normalise","symmetrise","force_positive_definite",
    "modal_strain_energy","sparse_submatrix","modal_residues",
    "memory_mb","sparsity","is_diagonal","bandwidth","matrix_stats",
    "diagonal_scaling","unscale_modes","apply_bcs","reorder_rcm",
    "ansys_dof","dof_to_ansys","build_dof_map",
]
