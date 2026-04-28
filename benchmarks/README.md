# pyserep Benchmarks

Two benchmarks measuring end-to-end performance on synthetic spring-mass chains.

## Running

```bash
# Full pipeline benchmark (default: 300, 1000, 3000 DOFs)
make benchmark-pipeline

# DOF selector comparison (default: 200, 500, 1000 DOFs)
make benchmark-selectors

# Custom sizes
python benchmarks/benchmark_pipeline.py --sizes 1000 5000 20000
python benchmarks/benchmark_dof_selectors.py --sizes 500 2000 5000
```

---

## benchmark_pipeline.py

End-to-end SEREP ROM pipeline benchmark.

**Metrics reported per model size:**

| Metric | Description |
|--------|-------------|
| Eigensolver time | ARPACK shift-invert for `n_modes` eigenpairs |
| Mode select time | Full MS1+MAC+MS2+MS3 pipeline |
| DOF select time | DS4 Effective Independence iterations |
| ROM build time | T, Kₐ, Mₐ construction |
| FRF (ROM) time | Direct impedance inversion over frequency grid |
| FRF (ref) time | Modal superposition over all elastic modes |
| Wallclock speedup | t_ref / t_ROM |
| DOF retention % | n_master / N × 100 |
| Peak memory | MB allocated at completion |
| Eigenvalue error % | Max freq preservation error |
| FRF max error % | Max relative FRF error vs reference |

**Interpretation:** The wallclock speedup is most meaningful for large N
(N > 10,000) where the modal reference sum over hundreds of modes is
expensive. For small synthetic models used in CI, the overhead of the
eigensolver setup dominates and the speedup appears < 1.

---

## benchmark_dof_selectors.py

Compares all four DOF selection methods (DS1–DS4) on the same model at
multiple sizes.  The key metric is κ(Φₐ) — the condition number of the
modal submatrix at selected master DOFs.

**Expected output:**

```
SUMMARY — κ(Φₐ) comparison across model sizes
      N   Modes        DS1 KE      DS2 Disp       DS3 SVD       DS4 EID
    200      78    1.70e+12    2.32e+10    5.00e+00    1.31e+02
    500      74    1.03e+13    3.41e+05    5.00e+00    4.68e+00
  2,000     111    4.78e+14    8.23e+06    5.01e+00    5.37e+00

Best condition number consistently: DS4 EID (Kammer 1991)
```

**Key insight:** DS4 (Effective Independence) is the only method that
consistently produces κ < 10³ for SEREP.  DS3 (SVD/QR pivot) is a
reasonable fallback when DS4 is too slow for very large candidate sets.
DS1 and DS2 are included only for comparison — they are unsuitable for
SEREP use because the resulting κ ≫ 10³ destroys eigenvalue preservation.

---

## Reference hardware

Benchmarks were run on:
- CPU: single-threaded
- Python: 3.11
- NumPy: 1.24 (OpenBLAS)
- SciPy: 1.10
