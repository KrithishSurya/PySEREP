# pyserep Examples

Eight self-contained examples demonstrating the full capability of `pyserep`.
Each script runs independently and saves its plots to the current directory.

## Running an example

```bash
cd examples/
python 01_quick_start.py
```

No FE matrix files are needed for examples 01–05, 07, and 08 — they use
the built-in synthetic models. Example 02 and 06 require your own Ansys-exported
matrix files; see the script header for path configuration.

---

## Example index

### 01 — Quick Start: Functional API  (`01_quick_start.py`)
Full walkthrough of the functional API on a 500-DOF spring-mass chain:
eigenproblem → mode selection → DS4 EID DOF selection → SEREP ROM →
direct FRF → validation.  Best starting point.

### 02 — Pipeline API: Garteur SM-AG19  (`02_pipeline_garteur.py`)
One-function `SereпPipeline(cfg).run()` call on the Garteur SM-AG19
aircraft-like benchmark structure (66,525 DOFs, selective frequency bands).
**Requires your K.mtx and M.mtx files** — update paths in the script.

### 03 — DOF Selector Comparison DS1–DS4  (`03_dof_selector_comparison.py`)
Benchmarks all four DOF selection methods on a 400-DOF chain.  Prints
a comparison table of κ(Φₐ), rank, and timing.  Saves a bar chart.

### 04 — Direct FRF vs Modal Superposition  (`04_direct_vs_modal_frf.py`)
Side-by-side comparison of the direct (impedance inversion) and modal
superposition FRF methods for the ROM, both measured against the
full-model reference.  Shows why the direct method is preferred.

### 05 — Mode Count Convergence Study  (`05_convergence_study.py`)
Sweeps the upper frequency cutoff from 20 Hz to 120 Hz and shows how
FRF accuracy and condition number improve as more modes are retained.

### 06 — HyAPA Parametric Material Study  (`06_hyapa_material.py`)
Runs the SEREP pipeline in parallel for two material configurations
(Standard and HyAPA) and prints a side-by-side comparison table.
**Requires two pairs of matrix files** — update paths in the script.

### 07 — Euler-Bernoulli Beam + Sensitivity  (`07_euler_beam.py`)
SEREP ROM on the built-in cantilever beam model.  Includes:
- Eigenvalue sensitivity to Young's modulus perturbation
- Monte Carlo FRF uncertainty quantification (±3% E, ±2% ρ)

### 08 — 2D Kirchhoff Plate  (`08_plate_2d.py`)
SEREP ROM on the built-in 2D thin plate model.  Shows 2D mode shape
visualisation, two-pair FRF comparison, and DOF selector bar chart.

---

## Increasing model complexity

| Example | N (DOFs) | Modes | Master DOFs | Time (approx.) |
|---------|----------|-------|-------------|----------------|
| 01      | 500      | ~15   | ~15         | < 5 s          |
| 03      | 400      | ~12   | ~12         | < 3 s          |
| 05      | 400      | ~8–25 | ~8–25       | < 10 s         |
| 07      | 82       | ~8    | ~8          | < 3 s          |
| 08      | 99       | ~12   | ~12         | < 5 s          |
| 02      | 66,525   | ~37   | ~37         | 5–15 min       |
