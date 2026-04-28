pyserep Documentation
======================

.. toctree::
   :maxdepth: 3
   :caption: Contents

   installation
   quickstart
   api/index
   theory
   changelog

Introduction
------------

**pyserep** is a production-grade Python library for the System Equivalent
Reduction Expansion Process (SEREP), a model order reduction technique for
large-scale structural finite element models.

Key features:

- Exact SEREP eigenvalue preservation (Φₐ⁺ pseudoinverse formulation)
- **Direct FRF** via impedance inversion ``H(ω) = Z(ω)⁻¹`` — no modal truncation
- Four DOF selection methods (DS1–DS4) with condition number benchmarking
- Selective non-contiguous frequency band analysis
- Sensitivity analysis and Monte Carlo uncertainty quantification
- Full validation suite, CLI, and mesh export to Ansys/ParaView/UFF58

Install:

.. code-block:: bash

   pip install pyserep

.. note::
   For the Garteur SM-AG19 benchmark results, see the examples directory.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
