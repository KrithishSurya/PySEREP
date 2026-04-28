# Changelog

## [3.0.0] — 2025

### Added
- **Direct FRF method** (`compute_frf_direct`): impedance inversion of Kₐ, Mₐ — no modal truncation error
- Four damping models: modal, Rayleigh, hysteretic, user-supplied Cₐ
- Full validation suite: eigenvalue, MAC, orthogonality, expansion, positive definiteness
- Three additional DOF selectors: DS1 (Kinetic Energy), DS2 (Modal Displacement), DS3 (SVD/QR pivot)
- `compare_dof_selectors()` — benchmarks all four methods simultaneously
- Multi-format I/O: HDF5 support, CSV support, symmetry check on load
- Performance dashboard (6-panel matplotlib figure)
- `PerformanceMetrics` dataclass with FLOP counts and speedup ratio
- `ValidationReport` dataclass
- CLI entry point: `pyserep` command
- `pyproject.toml` — modern packaging (replaces `setup.py`)
- GitHub Actions CI workflow (Python 3.9–3.12)

### Changed
- **Package renamed**: `pyserep` → namespace unchanged but internal structure fully reorganised
- `pyserep/core/` now contains only eigensolver and ROM builder (no FRF)
- FRF moved to dedicated `pyserep/frf/` subpackage
- Selection logic moved to `pyserep/selection/`
- I/O moved to `pyserep/io/`
- `SereпPipeline` now defaults to `frf_method="direct"`
- `PipelineResults` carries `validation` and `performance` fields

### Fixed
- Mass normalisation applied after ARPACK convergence (not before)
- Symmetry enforcement `0.5*(A + Aᵀ)` applied to Ka and Ma after construction
- Condition number labelling: EXCELLENT/GOOD/MARGINAL/POOR thresholds documented

## [2.0.0] — 2025 (Selective Band Edition)

### Added
- FrequencyBand and FrequencyBandSet
- Selective non-contiguous frequency band analysis
- Band-weighted MPF (MS2) with per-band evaluation
- Gap shading in FRF plots

## [1.0.0] — 2024 (Initial Release)

### Added
- Basic SEREP pipeline
- DS4 Effective Independence DOF selection
- Modal superposition FRF
- Matrix Market (.mtx) loading
