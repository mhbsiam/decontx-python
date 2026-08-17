# Changelog

This fork improves performance and code quality. It keeps numerical accuracy.

## Performance

- CSR-aligned sparse EM: native counts are kept as a 1D array (`nc_data`) aligned to CSR nonzeros (length nnz). They are never densified into an `n_cells x n_genes` matrix. The M-step scatter-adds from CSR positions into `phi_acc`. All steps run in JIT Numba. There is no Python/BLAS overhead.
- Bit-identical results with the original dense implementation (theta_diff = 0.0, correlation = 1.0).
- Benchmark speedups (original dense vs CSR-aligned optimized):

  | Cells | Genes | Density | Original | Optimized | Speedup |
  |-------|-------|---------|----------|-----------|---------|
  | 500 | 500 | 15% | 0.11s | 0.009s | 12x |
  | 5,000 | 3,000 | 8% | 12.8s | 0.11s | 113x |
  | 10,000 | 5,000 | 5% | 31.0s | 0.23s | 134x |

- Memory: `nc_data` is ~14x smaller than dense `native_counts` (9 MB vs 120 MB for 5k x 3k at 8% density).
- Output is a sparse CSR matrix. It uses the same sparsity pattern as the input.
- Numba JIT compilation is lazy. It compiles on the first `decontx()` call, not at import time. Call `decontx.fast_ops.precompile()` to pre-warm.
- Optional `compute_log_likelihood=False` gives an additional ~25–40 % speedup when you do not need the log-likelihood history. It does not change contamination, theta, phi, eta, or decontaminated counts.
- The M-step scatter-add runs one cluster per thread. It uses a precomputed stable cell ordering. Each thread owns a single `phi_acc` row. No per-thread accumulator copies are needed. The summation order is fixed. Results stay bit-identical across thread counts.
- `dtype="float32"` halves the working set and the output layer. The measured maximum absolute error is 2.4e-08.

Measured on 10 000 cells x 3 000 genes (3.9 M nonzeros), best of 3:

| | ms/iter | iterations | total | output |
|---|---|---|---|---|
| previous release | 6.75 | 11 | 74 ms | 31.0 MB |
| current, float64 | 4.23 | 4 | **17 ms** | 46.4 MB |
| current, float32 | **2.92** | 4 | **12 ms** | 31.0 MB |

## Code quality

- Removed ~700 lines of dead code in `fast_ops.py`, `utils.py`, `core.py`, and `model.py`. Removed the unreachable `utils.initialize_clusters`.
- Split the delta metadata: `uns['decontX']['parameters']['delta']` holds the value you passed in. `uns['decontX']['fitted']['delta']` holds the fitted result.
- Fixed multi-batch output: sparse batch results are stacked with `scipy.sparse.vstack`. They are not densified into a zero-filled array.
- Fixed multi-batch cluster indexing: each batch's cluster labels are remapped to a contiguous `1..n_clusters` range internally. This prevents out-of-bounds writes when a batch contains only a subset of the global clusters.
- Fixed the single-cluster division-by-zero edge case in the eta update.

## R parity (breaking changes to output values)

Contamination estimates and decontaminated counts differ from previous releases. Correlation against an independent transcription of celda's `src/DecontX.cpp` (`tests/r_reference.py`) improved from 0.9815 to 0.99995. R reports contamination from the E-step at the start of its final iteration. This is a documented one-E-step offset. R's reported contamination corresponds to a different E-step than its reported counts. DecontX Python runs one more E-step with the converged parameters. The extra E-step keeps contamination consistent with the returned counts. Agreement is 5e-16. If you align that step, the two implementations match to 7.2e-16. The test `test_r_parity_is_exact_modulo_final_estep` pins this at 1e-10.

- Theta now uses R's posterior mean, not the posterior mode. The mode form under-reported contamination by ~21 % relative. Its denominator went negative for low-count cells once the fitted delta dropped below 1.0. A negative denominator inverted the contamination. A fully contaminated cell could then report 0 % contamination. On shallow data (median 3 counts/cell) this pinned 382 of 400 cells at exactly 0.0 or 1.0.
- Contamination is the E-step proportion `(total - native) / total`. This matches R. It is not `1 - theta`.
- Decontaminated counts are fractional. This matches R. Rounding to int32 destroyed ~8 % of nonzero entries.
- Convergence is checked every iteration. The stopping point no longer depends on the log-likelihood diagnostic interval.
- Delta is fitted by Minka's fixed-point Dirichlet MLE. This matches `MCMCprecision::fit_dirichlet`. It replaces method-of-moments with a `[0.1, 1000]` clamp. That clamp floor let delta fall below 1.0 and invert the old theta update.
- Multi-batch fixes: NaN batch labels now raise instead of crashing or silently reporting zero contamination. A batch named `"all"` no longer collides with the single-batch sentinel. Each batch gets a distinct derived seed.
- `obs['decontX_clusters']` now holds your original cluster labels. It previously held internal codes. Those internal codes renumbered string clusters lexicographically (`"10"` became 2 while `"2"` became 3).
- Log-normalized input is now rejected. It previously produced plausible-looking output silently.
- Added division-by-zero guards in the theta and delta precision updates.
- Fixed `test_core.py` (wrong key casing, missing cluster column).
- Lazy JIT compilation: import-time compilation removed. The first call compiles.
- Ruff lint and format pass clean on all files.

## Acknowledgments

The sparse EM architecture in this implementation comes from [jjia1/decontx-python](https://github.com/jjia1/decontx-python). Specifically:

- The "keep everything CSR" approach: convert input to CSR upfront. Never densify during the EM loop. This makes the E-step O(nnz), not O(n_cells x n_genes).
- Native counts stay as a 1D array (`nc_data`) aligned to CSR nonzeros. They do not form a dense `n_cells x n_genes` matrix.
- Scatter-add M-step: accumulate `nc_data` directly into `phi_acc` in JIT. This avoids Python-level boolean masking or GEMM.
- Eta is computed from `global_acc - phi_acc[k]`. This avoids summing over an `other_mask`.
- The E-step and M-step are combined into a single monolithic Numba function call.
- Decontaminated counts are returned as a sparse CSR matrix. It uses the same sparsity pattern as the input.

That design is the foundation of the current `fast_ops.py`. The asymptotic win over a dense implementation comes entirely from it.

For clarity about the boundary: the following were not part of that work and were added here — hoisting the per-iteration buffers (`nc_data`, `phi_acc`, `native_sums`, `global_acc`) out of the kernel so they are allocated once and reused; parallelizing the M-step scatter-add over clusters; remapping cluster labels to a contiguous range so multi-batch runs cannot write out of bounds; optional float32 and log-likelihood skipping; and the R-parity corrections documented above. These corrections changed the theta update, the delta estimator, the contamination definition, and the counts dtype.
