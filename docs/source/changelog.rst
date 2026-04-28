Changelog
=========

All notable changes to `pyserep` are documented here.
This project follows `Semantic Versioning <https://semver.org>`_.

3.0.0 — 2025
-------------

Initial public release as **pyserep** (previously ``serep_rom`` internally).

**Added**

- ``pyserep.frf.direct_frf`` — Direct FRF via impedance inversion Z(ω)⁻¹
  with four damping models (modal, Rayleigh, hysteretic, user-supplied).
- ``pyserep.analysis.sensitivity`` — Eigenvalue sensitivity ∂λᵢ/∂p,
  FRF sensitivity ∂H/∂p, material perturbation sweep, Monte Carlo UQ.
- ``pyserep.analysis.convergence`` — Mode-count and DOF-count convergence
  studies with ``ConvergenceStudy`` result container and plots.
- ``pyserep.utils.linalg`` — Shared linear algebra: fast κ estimate,
  rank-revealing QR, safe pseudoinverse, force-positive-definite.
- ``pyserep.utils.sparse_ops`` — Sparse utilities: diagonal scaling, RCM
  reordering, BCs, bandwidth, ``ansys_dof`` / ``build_dof_map`` helpers.
- ``pyserep.io.mesh_writer`` — Export master DOFs to CSV, VTK, Ansys APDL,
  and UFF58 formats.
- ``pyserep.models.synthetic`` — Four built-in FE models: spring chain,
  Euler-Bernoulli beam, 2D Kirchhoff plate, random SPD pair.
- Full ``__all__`` export of 87 public symbols from the top-level namespace.
- ``py.typed`` PEP 561 marker for mypy compatibility.
- ``paper.md`` and ``paper.bib`` for JOSS submission.
- ``.readthedocs.yaml`` for automatic documentation builds.
- Complete CI/CD pipeline with auto-publish to PyPI on GitHub release.
- 119 automated tests (unit + integration) across Python 3.9–3.12.

**Changed**

- Package renamed from internal ``serep_rom`` to public **pyserep**.
- ``SereпPipeline`` now defaults to ``frf_method="direct"``.
- ``select_dofs_eid`` accepts ``required_dofs`` parameter, guaranteeing
  force/output DOFs are always present in the master set.
- All subpackage ``__init__.py`` files now expose complete public APIs.

**Fixed**

- Deleted ``build_serep_rom → build_pyserep`` mistranslation from rename.
- ``PerformanceMetrics`` serialisation via ``dataclasses.asdict``.
- Mass normalisation applied after ARPACK convergence check.
