# Validation: R vs Python Implementation

Parity is enforced by the test suite, not asserted. `tests/r_reference.py` is an
independent transcription of celda's `src/DecontX.cpp` — dense numpy, no Numba, no code
shared with `decontx/` — and `test_r_parity_gate` runs the current implementation
against it on every test run.

| | max absolute error vs R reference | correlation |
|---|---|---|
| float64 (default) | 7.2e-16 | 1.000000000 |
| float32 | 2.4e-08 | 1.000000000 |

The remaining float64 difference is a documented one-E-step offset: R reports
contamination from the E-step at the start of its final iteration, so its reported
contamination corresponds to a different E-step than its reported counts. DecontX Python
runs one more E-step with the converged parameters, keeping contamination consistent
with the counts it returns (agreement: 5e-16). Align that step and the two
implementations match to 7.2e-16, which `test_r_parity_is_exact_modulo_final_estep`
pins at 1e-10.

Reproduce it yourself:

```bash
pytest tests/test_regression.py -k parity -v
```

## Real-data comparison

<img width="100%" alt="decontx_comparison_real_data" src="https://github.com/user-attachments/assets/fc1358fa-1f54-42d9-953d-d2281d90d2d5" />

**PBMC 3K Dataset Results:**
- **Correlation: 0.999** between R and Python implementations
- **Mean Absolute Error: <1%** across all parameter settings
- **Identical statistical properties** (mean, std, range)
- **Per-cluster consistency** maintained across cell types

Note that this figure predates the R-parity corrections described above and has not been
regenerated since. The measurements in the table are the current, reproducible numbers.
For reference, the pre-correction implementation scored 0.9815 against the transcription
above — below the >0.999 bar the project sets for itself — which is what prompted those
fixes.
