Installation
============

Requirements
------------

* Python 3.9, 3.10, 3.11, or 3.12
* NumPy ≥ 1.24
* SciPy ≥ 1.10
* Matplotlib ≥ 3.7

Standard install
----------------

.. code-block:: bash

   pip install pyserep

This installs the core library with all required dependencies.

Optional: HDF5 matrix loading
------------------------------

If your FE matrices are stored in HDF5 format (``h5``, ``hdf5``), install
the extra dependency:

.. code-block:: bash

   pip install pyserep[io-extra]

Development install
-------------------

To install from source with all development tools (test suite, linter,
documentation builder):

.. code-block:: bash

   git clone https://github.com/YourOrg/pyserep.git
   cd pyserep
   pip install -e ".[dev]"

Verify the installation
-----------------------

.. code-block:: bash

   make smoke

or:

.. code-block:: python

   import pyserep
   print(pyserep.__version__)

   from pyserep import spring_chain, solve_eigenproblem
   K, M = spring_chain(n=100)
   freqs, phi = solve_eigenproblem(K, M, n_modes=10, verbose=False)
   print(f"First 3 natural frequencies: {freqs[:3].round(2)} Hz")

Matrix file formats
-------------------

``pyserep`` reads structural matrices in the following formats without
any additional dependencies:

+------------------+--------------------+-------------------------------+
| Extension        | Format             | Source                        |
+==================+====================+===============================+
| ``.mtx``, ``.mm``| Matrix Market      | Ansys ``/AUX2``, Abaqus, etc. |
+------------------+--------------------+-------------------------------+
| ``.npz``         | SciPy sparse NPZ   | ``scipy.sparse.save_npz``     |
+------------------+--------------------+-------------------------------+
| ``.npy``         | NumPy dense array  | ``numpy.save``                |
+------------------+--------------------+-------------------------------+
| ``.csv``         | Dense CSV          | Any spreadsheet/text tool     |
+------------------+--------------------+-------------------------------+
| ``.h5``, ``.hdf5``| HDF5 sparse       | Requires ``pip install h5py`` |
+------------------+--------------------+-------------------------------+

Ansys DOF numbering
-------------------

Ansys exports matrices with DOFs numbered as:

.. code-block:: text

   DOF_index = (node_number − 1) × 3 + direction
   direction: 0=UX, 1=UY, 2=UZ

Use the built-in helper:

.. code-block:: python

   from pyserep import ansys_dof, dof_to_ansys

   dof = ansys_dof(1001, 0)   # Node 1001, UX → DOF 3000
   node, direction = dof_to_ansys(3000)   # → (1001, 0)

Symmetric matrix requirement
-----------------------------

.. warning::

   pyserep requires **real symmetric** K and M matrices.

   ========= ================================================
   Matrix    Requirement
   ========= ================================================
   K         Real, symmetric, positive **semi**-definite
   M         Real, symmetric, positive **definite**
   ========= ================================================

   Asymmetric matrices will not raise an immediate error in all code paths
   but will produce incorrect eigenvalues and FRF results.

Validate before running the pipeline:

.. code-block:: python

   from pyserep import load_matrices, check_symmetric_pd

   K, M = load_matrices("K.mtx", "M.mtx")
   report = check_symmetric_pd(K, M)
   print(report["message"])

Fix near-symmetric matrices (numerical noise from FE export):

.. code-block:: python

   from pyserep import enforce_symmetry
   K = enforce_symmetry(K)   # K ← 0.5*(K + Kᵀ)
   M = enforce_symmetry(M)

**Not supported:** Non-symmetric systems (gyroscopic, aerodynamic stiffness,
follower forces). These are planned for v4.0.

