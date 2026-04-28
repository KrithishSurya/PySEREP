Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install pyserep

For optional HDF5 matrix loading:

.. code-block:: bash

   pip install pyserep[io-extra]

Pipeline API (recommended)
--------------------------

The fastest way to run a complete SEREP ROM from matrix files to validated
FRF output:

.. code-block:: python

   from pyserep import SereпPipeline, ROMConfig, FrequencyBand

   cfg = ROMConfig(
       stiffness_file  = "K.mtx",
       mass_file       = "M.mtx",
       force_dofs      = [3000],
       output_dofs     = [3000],
       bands           = [FrequencyBand(0.1, 100.0),
                          FrequencyBand(400.0, 500.0)],
       frf_method      = "direct",
       dof_method      = "eid",
       num_modes_eigsh = 120,
   )

   results = SereпPipeline(cfg).run()
   print(results.summary())

Functional API
--------------

For research workflows where you need control over each step:

.. code-block:: python

   import numpy as np
   from pyserep import (
       load_matrices, solve_eigenproblem,
       select_modes, select_dofs_eid,
       build_serep_rom, verify_eigenvalues,
       compute_frf_direct, validate_serep,
       FrequencyBand, FrequencyBandSet,
   )
   from pyserep.utils.sparse_ops import build_dof_map

   # 1. Load
   K, M = load_matrices("K.mtx", "M.mtx")

   # 2. Eigenproblem
   freqs, phi = solve_eigenproblem(K, M, n_modes=100)

   # 3. Frequency bands
   band_set = FrequencyBandSet([
       FrequencyBand(0.1, 100.0, label="Low"),
       FrequencyBand(400.0, 500.0, label="High"),
   ])

   # 4. Mode selection
   modes = select_modes(phi, freqs, force_dofs=[3000],
                        output_dofs=[3000], band_set=band_set)

   # 5. DOF selection — DS4 Effective Independence
   dofs, kappa = select_dofs_eid(phi, modes,
                                  required_dofs=np.array([3000]))
   print(f"κ(Φₐ) = {kappa:.2e}")   # expect < 100

   # 6. Build SEREP ROM
   T, Ka, Ma = build_serep_rom(K, M, phi, modes, dofs)

   # 7. Verify eigenvalue preservation
   errors, max_err = verify_eigenvalues(Ka, Ma, freqs, modes)
   print(f"Max eigenvalue error: {max_err:.8f}%")   # expect < 0.001%

   # 8. Direct FRF
   local_f, local_o = build_dof_map(dofs, [3000], [3000])
   freq_eval = band_set.frequency_grid()
   _, H = compute_frf_direct(Ka, Ma, local_f, local_o, freq_eval, zeta=0.005)

   # 9. Full validation
   report = validate_serep(K, M, phi, freqs, modes, dofs, T, Ka, Ma)
   print(report.summary())

CLI
---

.. code-block:: bash

   # Full range
   pyserep -k K.mtx -m M.mtx -f 3000 -o 3000 --freq-range 0.1 500

   # Selective bands
   pyserep -k K.mtx -m M.mtx -f 3000 -o 3000 \
       --bands "0.1,100,Low" "400,500,High" \
       --frf-method direct --zeta 0.005

   # Multiple DOF pairs
   pyserep -k K.mtx -m M.mtx -f 3000 5000 -o 3000 5000 \
       --bands "0.1,100" --num-modes 150
