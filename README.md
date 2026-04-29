# pyserep

**Production-grade Python library for SEREP-based Reduced Order Modelling for structural dynamics problems.**

> **Symmetric matrices required.** Both K and M must be symmetric. K must be positive semi-definite, M must be positive definite. The matrix loader warns automatically if this is violated.

[![CI](https://github.com/KrithishSurya/pyserep/actions/workflows/ci.yml/badge.svg)](https://github.com/KrithishSurya/pyserep/actions)
[![codecov](https://codecov.io/gh/KrithishSurya/pyserep/graph/badge.svg)](https://codecov.io/gh/KrithishSurya/pyserep)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/pyserep)](https://pypi.org/project/pyserep/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://readthedocs.org/projects/pyserep/badge/?version=latest)](https://pyserep.readthedocs.io)
<!-- [![JOSS](https://joss.theoj.org/papers/placeholder/badge.svg)](https://joss.theoj.org) -->

---

## Scope and Requirements

### What problems does pyserep solve?

pyserep works with **any structural dynamics FE model** described by real symmetric K and M matrices:

- Mechanical structures (beams, plates, shells, machine components)
- Civil engineering structures (bridges, buildings, frames)
- Acoustic-structural coupled systems
- Rotating machinery (at operating points where K, M are symmetric)
- Aerospace structures (validated on the Garteur SM-AG19 benchmark)

The Garteur benchmark is used for validation only — it is not a scope limitation.

### Symmetric matrix requirement

> **pyserep requires real symmetric K and M matrices.**

| Matrix | Requirement |
|--------|-------------|
| K (stiffness) | Real, symmetric, positive **semi**-definite |
| M (mass) | Real, symmetric, positive **definite** |

The matrix loader validates this on every load:

```python
from pyserep import load_matrices, check_symmetric_pd

K, M = load_matrices("K.mtx", "M.mtx")
report = check_symmetric_pd(K, M)
print(report["message"])
# Symmetry & definiteness check:
#   K relative asymmetry : 2.31e-15  (PASS)
#   M relative asymmetry : 1.88e-15  (PASS)
#   M positive definite  : PASS
#   Overall              : PASS — safe to proceed
```

If your FE software produces slightly asymmetric matrices (numerical noise < 1e-6), symmetrise them first:

```python
from pyserep import enforce_symmetry
K = enforce_symmetry(K)   # K ← 0.5*(K + Kᵀ)
M = enforce_symmetry(M)
```

**Not supported in v3.x:** Non-symmetric systems (gyroscopic effects, aerodynamic stiffness, follower forces, fluid-structure with non-symmetric coupling). Planned for v4.0.

---

## What is SEREP?

The **System Equivalent Reduction Expansion Process (SEREP)** is a model order reduction technique for structural finite element models. It constructs a transformation matrix **T = Φ · Φₐ⁺** from the modal matrix and its partition at selected *master* degrees of freedom, then projects the full physical matrices:

```
Kₐ = Tᵀ K T        Mₐ = Tᵀ M T
```

**Key property**: SEREP preserves eigenvalues *exactly* (up to machine epsilon). The reduced model reproduces the same natural frequencies as the full model for the retained modes — unlike Guyan/IRS reduction methods which introduce approximation errors.

---

## Why this library?

| Library | SEREP | Direct FRF | EID DOF selection | Selective bands |
|---|:---:|:---:|:---:|:---:|
| **PySEREP** | ✅ | ✅ | ✅ | ✅ |
| pyMOR | ❌ | ❌ | ❌ | ❌ |
| pyFBS | ❌ | ❌ | ❌ | ❌ |
| SDyPy | ❌ | ❌ | ❌ | ❌ |
| AMfe | ❌ | ❌ | ❌ | ❌ |

No other Python library implements SEREP with Effective Independence DOF selection and direct physical-matrix FRF computation.

---

## Features

### Core Algorithm
- **Exact SEREP** — Moore–Penrose pseudoinverse formulation; `eig(Kₐ, Mₐ) = eig(K,M)|selected`
- **Four DOF selectors** (DS1–DS4) with side-by-side condition number comparison
- **Five-step mode selection pipeline**: MS1 + MAC filter + MS2 + MS3 + MS4 conditioning check

### NEW in v3: Direct FRF Method
The FRF is computed via **impedance inversion of the physical reduced matrices**:

```
Z(ω) = Kₐ − ω²Mₐ + jωCₐ
H(ω) = Z(ω)⁻¹
```

This is fundamentally more accurate than modal superposition because:
- No modal truncation error within the retained subspace
- Works correctly with non-proportional damping
- Consistent with the physics of the reduced model
- Four damping models: modal, Rayleigh, hysteretic, user-supplied Cₐ

### Analysis & Validation
- Eigenvalue preservation error (should be < 0.001% for well-conditioned SEREP)
- Mass orthogonality: `|ΦᵀMΦ − I|_max`
- Stiffness orthogonality: `|ΦᵀKΦ − Λ|_max`
- Modal Assurance Criterion (MAC) matrix
- Transformation expansion accuracy: `|TΦₐ − Φ|_F / |Φ|_F`
- Positive definiteness of Kₐ and Mₐ

### Two APIs
```python
# High-level: entire pipeline in one call
results = SereпPipeline(config).run()

# Low-level: call each function independently
freqs, phi = solve_eigenproblem(K, M, n_modes=100)
modes      = select_modes(phi, freqs, ...)
dofs, κ    = select_dofs_eid(phi, modes)
T, Ka, Ma  = build_pyserep(K, M, phi, modes, dofs)
freqs_eval, H = compute_frf_direct(Ka, Ma, ...)
```

---

## Installation

```bash
pip install pyserep
```

Or from source:
```bash
git clone https://github.com/YourOrg/pyserep.git
cd pyserep
pip install -e ".[dev]"
```

**Requirements**: Python 3.9+, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7

Optional: `h5py` for HDF5 matrix loading.

---

## Quick Start

### Pipeline API (recommended)

```python
from pyserep import SereпPipeline, ROMConfig, FrequencyBand

cfg = ROMConfig(
    stiffness_file    = "K.mtx",
    mass_file         = "M.mtx",
    force_dofs        = [3000],        # 0-based global DOF index
    output_dofs       = [3000],
    bands = [
        FrequencyBand(0.1, 100.0, label="LowBand"),
        FrequencyBand(400.0, 500.0, label="HighBand"),
    ],
    frf_method        = "direct",      # impedance inversion (recommended)
    damping_type      = "modal",
    zeta              = 0.005,
    dof_method        = "eid",         # Effective Independence (DS4)
    num_modes_eigsh   = 120,
)

results = SereпPipeline(cfg).run()
print(results.summary())
```

### Functional API

```python
from pyserep import (
    load_matrices, solve_eigenproblem, select_modes,
    select_dofs_eid, build_pyserep, verify_eigenvalues,
    compute_frf_direct, validate_serep,
    FrequencyBand, FrequencyBandSet,
)

# Load
K, M = load_matrices("K.mtx", "M.mtx")

# Eigenproblem
freqs, phi = solve_eigenproblem(K, M, n_modes=100)

# Mode selection (selective bands)
band_set = FrequencyBandSet([
    FrequencyBand(0.1, 100.0),
    FrequencyBand(400.0, 500.0),
])
modes = select_modes(phi, freqs, force_dofs=[3000], output_dofs=[3000],
                     band_set=band_set)

# DOF selection — DS4 Effective Independence
dofs, kappa = select_dofs_eid(phi, modes)
print(f"κ(Φₐ) = {kappa:.4e}")   # expect < 100

# SEREP ROM
T, Ka, Ma = build_pyserep(K, M, phi, modes, dofs)

# Eigenvalue verification (SEREP property)
errors, max_err = verify_eigenvalues(Ka, Ma, freqs, modes)
print(f"Max eigenvalue error: {max_err:.8f}%")   # expect < 0.001%

# Direct FRF
dof_map = {int(d): i for i, d in enumerate(dofs)}
lf = [dof_map[3000]]   # local index of force DOF
lo = [dof_map[3000]]   # local index of output DOF
freq_eval = band_set.frequency_grid()

_, H = compute_frf_direct(Ka, Ma, lf, lo, freq_eval, zeta=0.005)

# Validation suite
report = validate_serep(K, M, phi, freqs, modes, dofs, T, Ka, Ma)
print(report.summary())
```

---

## DOF Selector Comparison

```python
from pyserep.selection.dof_selector import compare_dof_selectors

comparison = compare_dof_selectors(phi, selected_modes)
```

Typical results for the Garteur SM-AG19 benchmark (37 retained modes):

| Method | Description | κ(Φₐ) | Notes |
|---|---|---|---|
| **DS1** | Kinetic Energy | ~10¹⁵ | Fast but unusable for SEREP |
| **DS2** | Peak Displacement | ~10¹² | Simple but poor conditioning |
| **DS3** | SVD / QR pivot | ~10⁵ | Better, still marginal |
| **DS4** | Effective Independence | **~23** | ✅ Recommended — excellent κ |

---

## Selective Frequency Bands

Analyse only the frequency ranges you care about. Gap regions are **never computed**.

```python
bands = [
    FrequencyBand(0.1, 100.0,  label="LowBand"),    # analyse
    # 100–400 Hz gap: completely ignored
    FrequencyBand(400.0, 500.0, label="HighBand"),   # analyse
]
```

**How it works:**
- **MS1**: Modes pass if `f_n ≤ α × max(band.f_max)` for *any* band
- **MS2**: Band-weighted MPF evaluated *within each band separately* — gap modes get near-zero scores and are excluded
- **FRF grid**: Union of band grids only; zero computation in gaps

---

## Matrix Formats

| Format | Extension | Notes |
|---|---|---|
| Matrix Market | `.mtx`, `.mm` | Standard FE export format (Ansys, Abaqus) |
| Harwell-Boeing | `.rua`, `.rb` | Legacy sparse format |
| NumPy sparse | `.npz` | `scipy.sparse.save_npz` |
| NumPy dense | `.npy` | Small models only |
| HDF5 | `.h5`, `.hdf5` | Requires `pip install h5py` |
| CSV | `.csv` | Dense; small models only |

---

## Ansys DOF Numbering

```
DOF_index = (node_number − 1) × 3 + direction
direction:  0 = UX,  1 = UY,  2 = UZ
```
Example: Node 1001, UX direction → `DOF = (1001−1)×3+0 = 3000`

---

## Output Files

After a pipeline run, `export_folder` contains:

| File | Contents |
|---|---|
| `SEREP_master_dofs.npy` | Master DOF indices (int array) |
| `SEREP_selected_modes.npy` | Selected mode indices |
| `SEREP_freqs_selected.npy` | Corresponding natural frequencies (Hz) |
| `SEREP_Ka.npy` | Reduced stiffness matrix (m × m) |
| `SEREP_Ma.npy` | Reduced mass matrix (m × m) |
| `SEREP_T.npy` | Transformation matrix (N × m) |
| `SEREP_frf.npz` | ROM and reference FRF (complex arrays) |
| `SEREP_metrics.json` | All scalar metrics (κ, errors, timing) |
| `SEREP_summary.txt` | Human-readable summary |
| `SEREP_frf.png` | 4-panel FRF comparison plot |
| `SEREP_dashboard.png` | 6-panel performance dashboard |

---

## CLI

```bash
# Full range, direct FRF
pyserep -k K.mtx -m M.mtx -f 3000 -o 3000 --freq-range 0.1 500 --frf-method direct

# Selective bands
pyserep -k K.mtx -m M.mtx -f 3000 -o 3000 \
    --bands "0.1,100,Low" "400,500,High" \
    --frf-method direct --zeta 0.005

# Multiple DOF pairs, quiet mode
pyserep -k K.mtx -m M.mtx -f 3000 5000 -o 3000 5000 \
    --bands "0.1,100" --quiet
```

---

## Project Structure

```
pyserep/
├── pyserep/
│   ├── __init__.py            ← Public API
│   ├── cli.py                 ← CLI entry point
│   ├── io/
│   │   ├── matrix_loader.py   ← Multi-format matrix loading
│   │   └── exporter.py        ← Save / reload results
│   ├── core/
│   │   ├── eigensolver.py     ← ARPACK shift-invert eigenproblem
│   │   └── rom_builder.py     ← T, Kₐ, Mₐ construction
│   ├── selection/
│   │   ├── band_selector.py   ← FrequencyBand + FrequencyBandSet
│   │   ├── mode_selector.py   ← MS1, MS2, MS3, MAC, pipeline
│   │   └── dof_selector.py    ← DS1 KE, DS2 Disp, DS3 SVD, DS4 EID
│   ├── frf/
│   │   ├── direct_frf.py      ← Impedance inversion (NEW — recommended)
│   │   └── modal_frf.py       ← Modal superposition (reference / legacy)
│   ├── analysis/
│   │   ├── validation.py      ← Full validation suite
│   │   └── performance.py     ← FLOP counts, timing, reduction metrics
│   ├── visualization/
│   │   ├── frf_plots.py       ← 4-panel FRF comparison
│   │   ├── mode_plots.py      ← Mode shapes, MAC matrix, DOF comparison
│   │   └── summary_plots.py   ← 6-panel performance dashboard
│   ├── pipeline/
│   │   ├── config.py          ← ROMConfig dataclass
│   │   └── serep_pipeline.py  ← 12-step orchestrator
│   └── utils/
│       └── timers.py
├── tests/
│   ├── unit/                  ← Per-module unit tests
│   └── integration/           ← Full pipeline end-to-end tests
├── examples/
│   ├── 01_quick_start.py      ← Functional API walkthrough
│   ├── 02_pipeline_garteur.py ← Garteur benchmark with pipeline
│   ├── 03_dof_selector_comparison.py  ← DS1–DS4 benchmark
│   └── 04_direct_vs_modal_frf.py      ← Direct vs modal FRF
├── benchmarks/
├── docs/
├── .github/workflows/ci.yml
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Comparison with Related Libraries

| | **PySEREP** | pyMOR | pyFBS | SDyPy |
|---|---|---|---|---|
| **SEREP ROM** | ✅ Full implementation | ❌ | ❌ | ❌ |
| **Direct FRF** (impedance) | ✅ | ❌ | ❌ | ❌ |
| **EID DOF selection** | ✅ DS4 | ❌ | ❌ | ❌ |
| **Selective freq bands** | ✅ | ❌ | ❌ | ❌ |
| **Eigenvalue preservation** | ✅ Exact | ≈ | ❌ | ❌ |
| **Multi-format I/O** | ✅ MTX/NPZ/H5/CSV | Partial | ❌ | Partial |
| **Full validation suite** | ✅ | Partial | ❌ | ❌ |
| **CLI** | ✅ | ❌ | ❌ | ❌ |
| **Primary application** | Structural ROM | PDE parametric | Exp. substructuring | Signal proc. |

---

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{pyserep_2025,
  author  = {Surya, Krithish},
  title   = {pyserep: SEREP-Based Reduced Order Modelling for Structural Dynamics Problems},
  year    = {2025},
  url     = {https://github.com/YourOrg/pyserep},
  version = {3.0.0}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
