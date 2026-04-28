---
title: 'pyserep: A Python Library for SEREP-Based Reduced Order Modelling
        of Large-Scale Structural Dynamics Problems'
tags:
  - Python
  - structural dynamics
  - model order reduction
  - finite element analysis
  - structural engineering
  - vibration analysis
  - frequency response function
authors:
  - name: Krithish Surya
    orcid: 0009-0001-8980-9732
    affiliation: 1
affiliations:
  - name: Department of Mechanical Engineering, Vellore Institute of Technology,
          Vellore, Tamil Nadu, India
    index: 1
date: 2025
bibliography: paper.bib
---

# Summary

`pyserep` is a production-grade open-source Python library that implements the
**System Equivalent Reduction Expansion Process (SEREP)** [@OCallahan1989] for
reduced order modelling (ROM) of large-scale structural finite element (FE) models.
Given a structural stiffness matrix **K** and mass matrix **M** of arbitrary size,
`pyserep` automatically selects the physically significant vibration modes and
master degrees of freedom (DOFs), constructs the SEREP transformation matrix
**T = Φ Φₐ⁺**, and produces a reduced model (**Kₐ**, **Mₐ**) that is orders of
magnitude smaller than the original while preserving selected natural frequencies
to within machine precision.  The library computes Frequency Response Functions
(FRFs) via direct impedance inversion — `H(ω) = [Kₐ − ω²Mₐ + jωCₐ]⁻¹` — rather
than modal superposition, eliminating modal truncation errors.

`pyserep` has been validated on the Garteur SM-AG19 aircraft-like benchmark
structure [@Balmes1997] with 66,525 DOFs.  The pipeline reduces this to a
37-DOF system (0.056% DOF retention) with a maximum eigenvalue preservation
error of 7.06 × 10⁻⁷ % and zero measurable FRF error within the analysis
frequency band.

**Symmetric matrix requirement.** pyserep requires symmetric positive
semi-definite K and symmetric positive definite M.  The matrix loader
validates symmetry and issues a warning with the relative asymmetry measure
if this condition is not satisfied.

# Statement of Need

Structural FE models of mechanical, civil, and aerospace components routinely contain tens of
thousands to millions of DOFs.  Repeated dynamic analyses — frequency sweeps,
parametric studies, uncertainty quantification — on models of this size are
computationally prohibitive.  Model order reduction (ROM) addresses this by
projecting the full system onto a carefully chosen low-dimensional subspace,
producing a small equivalent model that preserves the dynamic behaviour in
the frequency range of interest.

Among ROM methods, SEREP holds a unique mathematical property: the eigenvalues
of the reduced model **exactly** equal those of the full model for the retained
modes.  This is in contrast to Guyan (static) condensation or the Improved
Reduced System (IRS) method, both of which introduce eigenvalue approximation
errors.  Despite this advantage, no dedicated Python library for SEREP existed
prior to `pyserep`.  Existing tools cover adjacent problems:

- **pyMOR** [@Milk2016] targets parametric ROM for PDE systems and does not
  implement SEREP or structural FE-oriented DOF selection.
- **pyFBS** [@Pogacar2021] implements frequency-based substructuring for
  experimental data but provides no analytical ROM capability.
- **SDyPy** targets structural dynamics signal processing and experimental
  modal analysis; it has no ROM pipeline.
- **AMfe** [@Gruber2019] is an academic FE research code with Craig-Bampton
  substructuring but no SEREP implementation.

`pyserep` fills this gap with a complete, tested, and documented SEREP
implementation including: a four-criterion mode selection pipeline, four DOF
selection methods with condition number benchmarking, direct FRF computation
with four damping models, a full validation suite, convergence and sensitivity
analysis tools, and mesh export to Ansys, ParaView, and UFF58 formats.

# Mathematical Background

## SEREP transformation

Given the generalised eigenproblem **K φ = λ M φ**, let **Φ** (N × m) be the
mass-normalised modal matrix for *m* retained modes and **Φₐ** (a × m) its
partition at *a* selected master DOFs.  The SEREP transformation is:

$$\mathbf{T} = \mathbf{\Phi} \mathbf{\Phi}_a^+ \quad (N \times a)$$

where **Φₐ⁺** is the Moore–Penrose pseudoinverse.  When a = m (exact SEREP),
**Φₐ** is square and **Φₐ⁺ = Φₐ⁻¹**.  The reduced matrices are:

$$\mathbf{K}_a = \mathbf{T}^\top \mathbf{K} \mathbf{T}, \qquad
  \mathbf{M}_a = \mathbf{T}^\top \mathbf{M} \mathbf{T}$$

**Exact preservation theorem** [@OCallahan1989]: `eig(Kₐ, Mₐ) = eig(K, M)` for
all retained modes — exactly, not approximately.

## Direct FRF computation

The dynamic stiffness matrix at angular frequency ω is:

$$\mathbf{Z}(\omega) = \mathbf{K}_a - \omega^2 \mathbf{M}_a + j\omega \mathbf{C}_a$$

The FRF between force DOF *f* and output DOF *o* is:

$$H_{of}(\omega) = \left[\mathbf{Z}(\omega)^{-1}\right]_{of}$$

This direct inversion approach has no modal truncation error, in contrast to the
classical modal superposition formula which sums over a finite set of modes.

## Effective Independence DOF selection

The master DOFs are selected using the Effective Independence (EID) method of
@Kammer1991, which maximises the determinant of the Fisher Information Matrix
det(**ΦₐᵀΦₐ**) by iteratively removing the DOF with the smallest diagonal entry
of the EI projection matrix **E = Φₛ(ΦₛᵀΦₛ)⁻¹Φₛᵀ**.  This yields
condition numbers κ(**Φₐ**) typically in the range 5–100, far below the 10¹⁰–10¹⁵
produced by simpler kinetic energy-based methods.

# Package Architecture

`pyserep` is organised into eight subpackages:

| Subpackage | Responsibility |
|---|---|
| `pyserep.io` | Matrix loading (MTX, NPZ, HDF5, CSV), result export, mesh writers |
| `pyserep.core` | Sparse eigensolver (ARPACK shift-invert), ROM construction |
| `pyserep.selection` | Frequency band management, mode selection pipeline, four DOF selectors |
| `pyserep.frf` | Direct FRF (impedance inversion), modal FRF (reference) |
| `pyserep.analysis` | Validation suite, performance metrics, convergence studies, sensitivity |
| `pyserep.visualization` | FRF comparison plots, mode shape plots, performance dashboard |
| `pyserep.models` | Built-in synthetic FE models for testing |
| `pyserep.utils` | Linear algebra utilities, sparse matrix operations, DOF index tools |

The library provides both a high-level **Pipeline API** and a low-level
**Functional API**, enabling use cases from one-command execution to component-level
algorithm research.

# Example

```python
from pyserep import SereпPipeline, ROMConfig, FrequencyBand

cfg = ROMConfig(
    stiffness_file    = "StiffMatrixmm.mtx",  # 66,525 × 66,525
    mass_file         = "MassMatrixmm.mtx",
    force_dofs        = [3000],               # Node 1001, UX direction
    output_dofs       = [3000],
    bands             = [FrequencyBand(0.1, 100.0),
                         FrequencyBand(400.0, 500.0)],
    frf_method        = "direct",
    dof_method        = "eid",
    num_modes_eigsh   = 120,
)

results = SereпPipeline(cfg).run()
# → 37 master DOFs, κ = 23, max eigenvalue error = 7×10⁻⁷ %
```

# Testing

`pyserep` includes 119 automated tests (unit and integration) covering all
subpackages.  Tests are run via `pytest` and executed automatically on every
commit through GitHub Actions across Python 3.9, 3.10, 3.11, and 3.12.
Integration tests build and validate the full SEREP ROM pipeline on synthetic
spring-mass chain models of varying sizes.

# Acknowledgements

The author thanks the VIT Vellore Department of Mechanical Engineering for
providing the computational resources and academic environment for this work.
The Garteur SM-AG19 benchmark structure used for validation is publicly
described in @Balmes1997.

# References
