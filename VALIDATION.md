# Validation: R vs Python Implementation

Parity is enforced by the test suite, not asserted. `tests/r_reference.py` is an independent transcription of celda's `src/DecontX.cpp`. It uses dense numpy. It shares no code with `decontx/`. The test `test_r_parity_gate` runs the current implementation against it on every test run.

| | max absolute error vs R reference | correlation |
|---|---|---|
| float64 (default) | 7.2e-16 | 1.000000000 |
| float32 | 2.4e-08 | 1.000000000 |

The remaining float64 difference is a documented one-E-step offset. R reports contamination from the E-step at the start of its final iteration. Its reported contamination corresponds to a different E-step than its reported counts. DecontX Python runs one more E-step with the converged parameters. The extra E-step keeps contamination consistent with the returned counts. The agreement is 5e-16. If you align that step, the two implementations match to 7.2e-16. The test `test_r_parity_is_exact_modulo_final_estep` pins this at 1e-10.

Reproduce it yourself:

```bash
pytest tests/test_regression.py -k parity -v
```

## Real-data comparison

<img width="100%" alt="decontx_comparison_real_data" src="https://github.com/user-attachments/assets/fc1358fa-1f54-42d9-953d-d2281d90d2d5" />

**PBMC 3K dataset results:**
- **Correlation: 0.999** between the R and Python implementations
- **Mean absolute error: <1%** across all parameter settings
- **Identical statistical properties** (mean, std, range)
- **Per-cluster consistency** across cell types

The figure above predates the R-parity corrections described above. We have not regenerated it since. The measurements in the table are the current, reproducible numbers. For reference, the pre-correction implementation scored 0.9815 against the transcription above. That score was below the >0.999 bar the project sets for itself. That gap prompted the fixes.
