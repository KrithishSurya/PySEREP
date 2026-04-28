Mathematical Theory
===================

This page documents the mathematical foundations of SEREP and all algorithms
implemented in `pyserep`.

System Equivalent Reduction Expansion Process (SEREP)
------------------------------------------------------


Symmetric matrix assumption
-----------------------------

All algorithms in pyserep assume **real symmetric** (K, M):

.. math::

   \mathbf{K} = \mathbf{K}^\top \in \mathbb{R}^{N \times N}, \quad
   \mathbf{M} = \mathbf{M}^\top \in \mathbb{R}^{N \times N}

This guarantees:

* Real eigenvalues :math:`\lambda_i \in \mathbb{R}` (squared natural frequencies)
* Real orthogonal mode shapes :math:`\boldsymbol{\Phi} \in \mathbb{R}^{N \times m}`
* Positive-definite SEREP matrices Kₐ and Mₐ (when M > 0)
* Well-defined Moore–Penrose pseudoinverse :math:`\boldsymbol{\Phi}_a^+`

**Out of scope for v3.x:** Non-symmetric systems arise from gyroscopic effects
(:math:`\mathbf{G}` skew-symmetric), aerodynamic stiffness (:math:`\mathbf{K}_{aero}` non-symmetric),
and follower forces.  These require a complex eigenproblem solver and a
modified SEREP formulation.  Support is planned for v4.0.

Generalised eigenvalue problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a structural FE model with *N* degrees of freedom, the equation of motion
under harmonic excitation is:

.. math::

   \mathbf{K} \boldsymbol{\varphi} = \lambda \mathbf{M} \boldsymbol{\varphi}

where :math:`\mathbf{K}` (N × N) is the stiffness matrix, :math:`\mathbf{M}`
(N × N) is the mass matrix, :math:`\lambda_i = \omega_i^2` are the eigenvalues
(squared natural angular frequencies), and :math:`\boldsymbol{\varphi}_i` are
the corresponding mode shapes. For mass-normalised modes:

.. math::

   \boldsymbol{\Phi}^\top \mathbf{M} \boldsymbol{\Phi} = \mathbf{I}, \qquad
   \boldsymbol{\Phi}^\top \mathbf{K} \boldsymbol{\Phi} = \boldsymbol{\Lambda}

where :math:`\boldsymbol{\Phi}` (N × m) is the modal matrix retaining *m* modes
and :math:`\boldsymbol{\Lambda} = \text{diag}(\omega_1^2, \ldots, \omega_m^2)`.

SEREP transformation
^^^^^^^^^^^^^^^^^^^^^

Partition :math:`\boldsymbol{\Phi}` at *a* master DOFs:

.. math::

   \boldsymbol{\Phi}_a = \boldsymbol{\Phi}[\text{master\_dofs}, :] \quad (a \times m)

The SEREP transformation matrix is:

.. math::

   \mathbf{T} = \boldsymbol{\Phi} \, \boldsymbol{\Phi}_a^+ \quad (N \times a)

where :math:`\boldsymbol{\Phi}_a^+` is the Moore–Penrose pseudoinverse. When
:math:`a = m` (exact SEREP), :math:`\boldsymbol{\Phi}_a` is square and
:math:`\boldsymbol{\Phi}_a^+ = \boldsymbol{\Phi}_a^{-1}`.

The reduced matrices are:

.. math::

   \mathbf{K}_a = \mathbf{T}^\top \mathbf{K} \mathbf{T}, \qquad
   \mathbf{M}_a = \mathbf{T}^\top \mathbf{M} \mathbf{T}

**Exact preservation theorem** (O'Callahan, 1989):

.. math::

   \text{eig}(\mathbf{K}_a, \mathbf{M}_a) \equiv \text{eig}(\mathbf{K}, \mathbf{M})
   \Big|_{\text{retained modes}}

exactly, up to floating-point precision. This is in contrast to Guyan reduction
(static condensation), which approximates the dynamic condensation and introduces
eigenvalue errors proportional to the ratio of retained to omitted frequencies.

Mode Selection
--------------

MS1 — Frequency range filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mode *i* passes MS1 if:

.. math::

   f_{\text{rb}} < f_i \leq \alpha \cdot \max_b(f_{\text{max},b})

where :math:`\alpha` is a safety factor (default 1.5), :math:`f_{\text{rb}}`
is the rigid-body exclusion threshold (default 1 Hz), and the maximum is taken
over all analysis frequency bands *b*.

MS2 — Band-weighted Modal Participation Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each band *B* and DOF pair (*f*, *o*), define the band-weighted MPF:

.. math::

   C_{i,B} = |\varphi_i(f) \cdot \varphi_i(o)| \cdot
              \max_{\omega \in B} \frac{1}{|\omega_i^2 - \omega^2|}

Mode *i* passes MS2 if :math:`C_{i,B} / C_{\text{dom},B} \geq \epsilon_2`
for at least one band, where :math:`\epsilon_2` is the threshold (default 1%).

MS3 — Spatial amplitude at target DOFs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mode *i* passes MS3 if, at any target DOF *d*:

.. math::

   \frac{|\varphi_i(d)|}{\max_j |\varphi_i(j)|} \geq \epsilon_3

where :math:`\epsilon_3` is the amplitude threshold (default 5%).

MAC Filter
^^^^^^^^^^

The Modal Assurance Criterion between modes *i* and *j* is:

.. math::

   \text{MAC}(i, j) =
   \frac{|\boldsymbol{\varphi}_i^\top \boldsymbol{\varphi}_j|^2}
        {(\boldsymbol{\varphi}_i^\top \boldsymbol{\varphi}_i)
         (\boldsymbol{\varphi}_j^\top \boldsymbol{\varphi}_j)}
   \in [0, 1]

Modes with MAC > threshold (default 0.90) are considered spatially redundant;
the one with lower participation score is removed.

DOF Selection
-------------

DS4 — Effective Independence (Kammer, 1991)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Effective Independence matrix is:

.. math::

   \mathbf{E} = \boldsymbol{\Phi}_s
                (\boldsymbol{\Phi}_s^\top \boldsymbol{\Phi}_s)^{-1}
                \boldsymbol{\Phi}_s^\top

At each iteration, remove the DOF with the smallest diagonal entry
:math:`E_{dd}`, which contributes least to the linear independence of the
mode shape columns. This maximises:

.. math::

   \det\!\left(\boldsymbol{\Phi}_a^\top \boldsymbol{\Phi}_a\right)
   = \det\!\left(\mathbf{Q}^\top \mathbf{Q}\right) \quad
     \text{(Fisher Information Matrix)}

The condition number :math:`\kappa(\boldsymbol{\Phi}_a)` produced by DS4 is
typically 5–100, compared to :math:`10^{10}`–:math:`10^{15}` for kinetic
energy-based methods (DS1).

Direct FRF Computation
----------------------

Dynamic stiffness matrix
^^^^^^^^^^^^^^^^^^^^^^^^^

The equation of motion in the frequency domain:

.. math::

   \mathbf{Z}(\omega) \, \mathbf{q} = \mathbf{f}

where the dynamic stiffness (impedance) matrix is:

.. math::

   \mathbf{Z}(\omega) = \mathbf{K}_a - \omega^2 \mathbf{M}_a + j\omega \mathbf{C}_a

The FRF matrix is:

.. math::

   \mathbf{H}(\omega) = \mathbf{Z}(\omega)^{-1}

and the point or cross FRF is :math:`H_{of}(\omega) = [\mathbf{H}]_{of}`.

Damping models
^^^^^^^^^^^^^^

**Modal damping** — builds :math:`\mathbf{C}_a` from mass-orthonormal modes:

.. math::

   \mathbf{C}_a = \mathbf{M}_a \boldsymbol{\Psi}\,
                  \text{diag}(2\zeta_i\omega_i)\,
                  \boldsymbol{\Psi}^\top \mathbf{M}_a

**Rayleigh damping**:

.. math::

   \mathbf{C}_a = \alpha \mathbf{M}_a + \beta \mathbf{K}_a

where :math:`\alpha` and :math:`\beta` are chosen to match damping ratio
:math:`\zeta` at the first and last retained frequencies.

**Hysteretic (structural) damping**:

.. math::

   \mathbf{Z}(\omega) = \mathbf{K}_a(1 + j\eta) - \omega^2 \mathbf{M}_a

Eigenvalue Sensitivity
----------------------

Nelson's method for :math:`\partial\lambda_i / \partial p`:

For mass-normalised modes:

.. math::

   \frac{\partial \lambda_i}{\partial p}
   = \boldsymbol{\varphi}_i^\top
     \!\left(\frac{\partial \mathbf{K}}{\partial p}
             - \lambda_i \frac{\partial \mathbf{M}}{\partial p}\right)
     \boldsymbol{\varphi}_i

FRF Sensitivity
^^^^^^^^^^^^^^^

Differentiating :math:`\mathbf{Z}\mathbf{H} = \mathbf{I}`:

.. math::

   \frac{\partial \mathbf{H}}{\partial p}
   = -\mathbf{Z}^{-1}\,
     \frac{\partial \mathbf{Z}}{\partial p}\,
     \mathbf{Z}^{-1}

where :math:`\partial \mathbf{Z}/\partial p =
\partial \mathbf{K}_a/\partial p - \omega^2 \partial \mathbf{M}_a/\partial p`.

References
----------

- O'Callahan, J., Avitabile, P. & Riemer, R. (1989). *System Equivalent
  Reduction Expansion Process (SEREP)*. IMAC VII, Las Vegas.
- Kammer, D. C. (1991). Sensor placement for on-orbit modal identification
  and correlation of large space structures. *AIAA J. Guidance*, 14(2), 251–259.
- Guyan, R. J. (1965). Reduction of stiffness and mass matrices.
  *AIAA Journal*, 3(2), 380.
- Nelson, R. B. (1976). Simplified calculation of eigenvector derivatives.
  *AIAA Journal*, 14(9), 1201–1205.
