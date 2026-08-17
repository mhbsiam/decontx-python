# DecontX Python
[![PyPI version](https://badge.fury.io/py/decontx-python.svg)](https://badge.fury.io/py/decontx-python)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

A Python implementation of DecontX for removing ambient RNA contamination from single-cell RNA-seq data, designed for seamless integration with scanpy workflows.

## Overview

DecontX is a Bayesian method to estimate and remove cross-contamination from ambient RNA in droplet-based single-cell RNA-seq data. This Python implementation provides near-perfect parity with the original R version (correlation > 0.999) while enabling pure Python workflows without R dependencies.

**Key Features:**
- Pure Python implementation (no R required)
- Seamless scanpy integration
- Numba-accelerated performance
- Bayesian contamination estimation per cell
- Validated against original R implementation

## What's New

This github fork substantially improves performance and code quality while
preserving numerical accuracy.

### Performance

- **CSR-aligned sparse EM**: native counts are kept as a 1D array (`nc_data`)
  aligned to CSR nonzeros (length nnz), never densified into an
  `n_cells x n_genes` matrix. The M-step scatter-adds from CSR positions
  into `phi_acc` — all in JIT Numba, no Python/BLAS overhead.
- **Bit-identical results** to the original dense implementation
  (theta_diff = 0.0, correlation = 1.0).
- **Benchmark speedups** (original dense vs CSR-aligned optimized):

  | Cells | Genes | Density | Original | Optimized | Speedup |
  |-------|-------|---------|----------|-----------|---------|
  | 500 | 500 | 15% | 0.11s | 0.009s | 12x |
  | 5,000 | 3,000 | 8% | 12.8s | 0.11s | 113x |
  | 10,000 | 5,000 | 5% | 31.0s | 0.23s | 134x |

- Memory: `nc_data` is ~14x smaller than dense `native_counts`
  (9 MB vs 120 MB for 5k x 3k at 8% density).
- Output is a sparse CSR matrix (same sparsity pattern as input).
- Numba JIT compilation is lazy (compiles on first `decontx()` call,
  not at import time). Call `decontx.fast_ops.precompile()` to pre-warm.
- Optional `compute_log_likelihood=False` gives an additional ~25–40 %
  speedup when the log-likelihood history is not needed, with no change to
  contamination, theta, phi, eta, or decontaminated counts.
- The M-step scatter-add runs one cluster per thread, using a precomputed
  stable cell ordering so each thread owns a single `phi_acc` row. No
  per-thread accumulator copies, and the summation order is fixed, so results
  stay bit-identical across thread counts.
- `dtype="float32"` halves the working set and the output layer for a measured
  2.4e-08 max absolute error.

Measured on 10 000 cells x 3 000 genes (3.9 M nonzeros), best of 3:

| | ms/iter | iterations | total | output |
|---|---|---|---|---|
| previous release | 6.75 | 11 | 74 ms | 31.0 MB |
| current, float64 | 4.23 | 4 | **17 ms** | 46.4 MB |
| current, float32 | **2.92** | 4 | **12 ms** | 31.0 MB |

### Code quality

- Removed ~700 lines of dead code across `fast_ops.py`, `utils.py`,
  `core.py`, and `model.py`, plus the unreachable `utils.initialize_clusters`.
- Split the delta metadata: `uns['decontX']['parameters']['delta']` is the value
  you passed in, `uns['decontX']['fitted']['delta']` is the fitted result.
- Fixed multi-batch output: sparse batch results are stacked with
  `scipy.sparse.vstack` instead of densified into a zero-filled array.
- Fixed multi-batch cluster indexing: each batch's cluster labels are now
  remapped to a contiguous `1..n_clusters` range internally, preventing
  out-of-bounds writes when a batch contains only a subset of the global
  clusters.
- Fixed the single-cluster division-by-zero edge case in the eta update.

### R parity (breaking changes to output values)

Contamination estimates and decontaminated counts differ from previous releases.
Correlation against an independent transcription of celda's `src/DecontX.cpp`
(`tests/r_reference.py`) improved from **0.9815 to 0.99995**. Accounting for a
documented one-E-step offset in when R reports contamination, the two
implementations now agree to **7e-16**.

- **Theta now uses R's posterior mean**, not the posterior mode. The mode form
  under-reported contamination by ~21% relative, and its denominator went
  negative for low-count cells once the fitted delta dropped below 1.0 —
  inverting them, so a fully contaminated cell could report 0% contamination.
  On shallow data (median 3 counts/cell) this pinned 382 of 400 cells at exactly
  0.0 or 1.0.
- **Contamination is the E-step proportion** `(total - native) / total`, matching
  R, rather than `1 - theta`.
- **Decontaminated counts are fractional**, matching R. Rounding to int32
  destroyed ~8% of nonzero entries.
- **Convergence is checked every iteration**, so the stopping point no longer
  depends on the log-likelihood diagnostic interval.
- **Delta is fitted by Minka's fixed-point Dirichlet MLE**, matching
  `MCMCprecision::fit_dirichlet`, replacing method-of-moments with a
  `[0.1, 1000]` clamp. That clamp floor was what let delta fall below 1.0 and
  invert the old theta update.
- Multi-batch fixes: NaN batch labels now raise instead of crashing or silently
  reporting zero contamination; a batch named `"all"` no longer collides with the
  single-batch sentinel; each batch gets a distinct derived seed.
- `obs['decontX_clusters']` now holds your original cluster labels. It previously
  held internal codes, which renumbered string clusters lexicographically
  (`"10"` became 2 while `"2"` became 3).
- Log-normalized input is now rejected rather than silently producing
  plausible-looking output.
- Added division-by-zero guards in the theta and delta precision updates.
- Fixed `test_core.py` (wrong key casing, missing cluster column).
- Lazy JIT compilation: import-time compilation removed; first call compiles.
- Ruff lint and format pass clean on all files.

### Acknowledgments

The sparse EM architecture that makes this implementation fast originates with
[jjia1/decontx-python](https://github.com/jjia1/decontx-python). Specifically:

- The "keep everything CSR" approach: convert input to CSR upfront and never
  densify during the EM loop, giving an E-step that is O(nnz) rather than
  O(n_cells x n_genes).
- Keeping native counts as a 1D array (`nc_data`) aligned to CSR nonzeros
  instead of a dense `n_cells x n_genes` matrix.
- Scatter-add M-step: accumulating `nc_data` directly into `phi_acc` in JIT
  instead of Python-level boolean masking or GEMM.
- Computing eta from `global_acc - phi_acc[k]` instead of summing over an
  `other_mask`.
- Combining the E-step and M-step into a single monolithic Numba function call.
- Returning decontaminated counts as a sparse CSR matrix with the same
  sparsity pattern as the input.

That design is the foundation of the current `fast_ops.py`, and the asymptotic
win over a dense implementation is entirely attributable to it.

For clarity about the boundary: the following were **not** part of that work and
were added here — hoisting the per-iteration buffers (`nc_data`, `phi_acc`,
`native_sums`, `global_acc`) out of the kernel so they are allocated once and
reused; parallelizing the M-step scatter-add over clusters; remapping cluster
labels to a contiguous range so multi-batch runs cannot write out of bounds;
optional float32 and log-likelihood skipping; and the R-parity corrections
documented above, which changed the theta update, the delta estimator, the
contamination definition, and the counts dtype.


## Installation

### Latest version from GitHub

```bash
pip install git+https://github.com/NiRuff/decontx-python.git
```

This installs the current `main` branch directly. Use this if you need the newest changes (for example, the performance improvements or `compute_log_likelihood` described below).

### From a local checkout

```bash
git clone https://github.com/NiRuff/decontx-python.git
cd decontx-python
pip install .
```

This installs a snapshot into site-packages. Re-run it after any source change.

For development, `pip install -e .` makes Python import the local source directly, so
edits take effect without reinstalling.

**Verify which copy you are importing**, especially if the package has ever been
installed from another path:

```bash
python -c "import decontx; print(decontx.__file__)"
```

A stale editable install (`.pth` file) pointing at an old checkout will shadow your
working copy from every directory except the repo root, and because the Numba kernels
have no bounds checking, running an outdated copy can surface as heap corruption rather
than a clean error. If the path is not what you expect, run `pip uninstall
decontx-python` and reinstall.

## Where DecontX goes in your workflow

DecontX is a **count model**. It needs two things: **raw integer counts** in `adata.X`,
and **cluster labels** in `adata.obs`. That combination determines where it slots in:

```
load  →  QC filter  →  cluster (on normalized data)  →  DecontX (on raw counts)  →  normalize → analyze
                                                        ▲
                                        raw counts in .X, labels in .obs
```

The subtlety is that clustering needs normalized data while DecontX needs raw counts, so
the normalization must not clobber `.X` before DecontX runs. Two ways to handle that.

### Recommended: cluster on a copy

`adata.X` never stops being raw counts, so there is nothing to restore and nothing to
get wrong.

```python
import scanpy as sc
import decontx

adata = sc.read_h5ad("pbmc.h5ad")          # adata.X = raw counts

# 1. QC filtering (operates on counts, safe to do first)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 2. Cluster on a normalized copy; adata.X stays raw
clust = adata.copy()
sc.pp.normalize_total(clust)
sc.pp.log1p(clust)
sc.pp.highly_variable_genes(clust, n_top_genes=2000)
sc.pp.pca(clust)
sc.pp.neighbors(clust)
sc.tl.leiden(clust)
adata.obs["leiden"] = clust.obs["leiden"]
del clust

# 3. DecontX, on raw counts
decontx.decontx(adata, cluster_key="leiden")

contamination = adata.obs["decontX_contamination"]
clean_counts = adata.layers["decontX_counts"]
print(f"Mean contamination: {contamination.mean():.1%}")
print(f"Highly contaminated cells (>50%): {(contamination > 0.5).sum()}")

# 4. Continue downstream from the decontaminated counts
adata.X = adata.layers["decontX_counts"]
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
```

### Memory-conscious alternative: stash counts in a layer

Avoids the full copy. Preprocess in place, then put the raw counts back before running
DecontX.

```python
adata = sc.read_h5ad("pbmc.h5ad")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.layers["counts"] = adata.X.copy()    # stash BEFORE normalizing

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

adata.X = adata.layers["counts"].copy()    # restore raw counts
decontx.decontx(adata, cluster_key="leiden")
```

### If you already have cluster labels

Skip straight to it — any `obs` column works, including `cell_type` annotations:

```python
decontx.decontx(adata, cluster_key="cell_type")
```

### Common mistakes

DecontX validates its input and will tell you if something is off, but the two easy
errors are worth stating outright:

- **Running it after `sc.pp.log1p`.** Raises `ValueError`. DecontX models counts; log
  values are meaningless to it. Run it earlier, or restore the counts layer first.
- **Running it after `sc.pp.normalize_total` only.** Emits a warning about non-integer
  values rather than raising, since normalized-but-not-logged data is harder to detect
  with certainty. The results are still unreliable — fix the ordering.
- **Subsetting to highly variable genes first.** DecontX estimates the ambient profile
  from the full transcriptome. Run it on all genes, then subset.

Filtering cells and genes *before* DecontX is fine and recommended — empty droplets and
never-detected genes only add noise to the ambient estimate.

## Usage and performance tips

### Basic run

```python
import decontx

decontx.decontx(adata, cluster_key="leiden")
```

The function modifies `adata` in place unless `copy=True`.

### Multi-batch processing

If your dataset has multiple batches, pass the batch column so each batch is processed separately:

```python
decontx.decontx(adata, cluster_key="leiden", batch_key="batch")
```

### Speed mode: skip log-likelihood history

When you only need the decontaminated counts and contamination estimates, you can skip the log-likelihood computation that runs every 10 iterations. This is typically **25–40 % faster** and does not change the contamination, theta, phi, eta, or decontaminated counts.

```python
decontx.decontx(adata, cluster_key="leiden", compute_log_likelihood=False)
```

The default is `True` to preserve existing behavior.

### Pre-warm the JIT compiler

Numba compiles the sparse kernels on the first call. To avoid paying that cost during the first real run, call `precompile()` once after importing:

```python
from decontx.fast_ops import precompile

precompile()

decontx.decontx(adata, cluster_key="leiden")
```

### Run tests and benchmarks (from the repo root)

```bash
# Tests
python -m pytest tests/test_core.py tests/test_regression.py -v

# Standalone benchmark
PYTHONPATH=. python tests/benchmark.py

# Head-to-head comparison with the jjia1 implementation
# (requires a local checkout of jjia1/decontx-python)
DECONTX_JJIA1_PATH=/path/to/jjia1/decontx-python \
PYTHONPATH=. python tests/benchmark_compare.py
```

If the package is installed in editable mode, `PYTHONPATH=.` is not needed.

## Why DecontX?

Ambient RNA contamination occurs when mRNA from lysed/stressed cells gets captured in droplets with other cells, causing:
- Cross-contamination between cell types
- Blurred cell type boundaries  
- False positive marker gene expression
- Reduced clustering quality

DecontX models each cell as a mixture of:
1. **Native transcripts** from the cell's true type
2. **Contaminating transcripts** from other cell types in the sample

## Validation: R vs Python Implementation

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

### Real-data comparison

<img width="2250" height="1500" alt="decontx_comparison_real_data" src="https://github.com/user-attachments/assets/fc1358fa-1f54-42d9-953d-d2281d90d2d5" />

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

## Method Comparison

Based on our benchmarking study:

| Method | Ambient RNA Removed | Precision | Conservativeness |
|--------|-------------------|-----------|------------------|
| **SoupX** | ~65% | High | Very conservative |
| **DecontX** | ~90% | Medium-High | Balanced |
| **CellBender** | ~90% | Medium | More aggressive |

**Recommendation**: 
- Use **SoupX** for maximum safety and minimal false positives
- Use **DecontX** for balanced contamination removal in standard workflows  
- Use **CellBender** when you can replace your entire preprocessing pipeline

## API Reference

### Main Function

```python
decontx.decontx(
    adata,
    cluster_key="leiden",
    batch_key=None,
    max_iter=500,
    delta=(10.0, 10.0),
    estimate_delta=True,
    convergence=0.001,
    seed=12345,
    copy=False,
    verbose=True,
    compute_log_likelihood=True,
)
```

**Parameters:**
- `adata`: AnnData object with raw counts in `.X`
- `cluster_key`: Column in `.obs` containing cluster labels
- `batch_key`: Optional column in `.obs` containing batch labels; batches are processed separately (default: `None`)
- `max_iter`: Maximum EM iterations (default: 500)
- `delta`: Beta prior parameters for contamination (default: `(10, 10)`)
- `estimate_delta`: Whether to estimate delta parameters (default: `True`)
- `convergence`: Convergence threshold (default: 0.001)
- `seed`: Random seed for theta initialization (default: `12345`)
- `copy`: Return copy or modify in place (default: `False`)
- `verbose`: Print progress to stdout (default: `True`)
- `compute_log_likelihood`: Compute the log-likelihood every 10 iterations for the
  progress output; set to `False` for faster runs. It is a diagnostic only — it does
  not affect convergence or any result (default: `True`)
- `dtype`: Working precision, `"float64"` or `"float32"`. float32 halves the largest
  allocations and the output layer, and is ~30 % faster per iteration. Measured cost:
  max absolute error 2.4e-08 against the R reference, versus 7.2e-16 at float64
  (default: `"float64"`)
- `round_counts`: Round the decontaminated counts to `int32`. Reduces the output layer
  by ~1.5x in RAM and on disk, at the cost of zeroing every native count below 0.5.
  Contamination estimates are unaffected (default: `False`)

**Returns:**
`None` if `copy=False` (default), otherwise a copy of `adata`. Results stored in `adata`:
- `adata.obs['decontX_contamination']`: Per-cell contamination estimates, computed
  as `(total counts - native counts) / total counts` to match the R implementation
- `adata.obs['decontX_clusters']`: The cluster labels used, as supplied by the caller
- `adata.layers['decontX_counts']`: Decontaminated count matrix. **Fractional**, as
  in R — round it yourself if you need integers, bearing in mind that rounding
  zeroes every entry below 0.5
- `adata.uns['decontX']`: Run metadata
  - `'parameters'`: the arguments you passed (`delta`, `estimate_delta`, `max_iter`,
    `convergence`, `seed`)
  - `'fitted'`: `{'delta': ...}`, the fitted beta prior (equal to the input when
    `estimate_delta=False`; the mean across batches when `batch_key` is used)
  - `'convergence'`: whether the EM actually converged — see below
  - `'cluster_map'`: your cluster labels mapped to the internal `1..K` codes
  - `'runtime'`: start/end timestamps and duration
  - `'version'`: package version

### Checking convergence

`uns['decontX']['convergence']` records why the EM stopped, per batch:

```python
conv = adata.uns["decontX"]["convergence"]
conv["all_converged"]            # False if any batch hit max_iter
conv["n_batches_not_converged"]
conv["per_batch"]                # parallel lists: batch, n_iter,
                                 # converged, final_theta_change
```

If any batch fails to converge, DecontX emits a `UserWarning` naming it. That
warning fires even under `verbose=False`, so non-convergence cannot pass
unnoticed. Estimates for cells in a non-converged batch are unreliable — raise
`max_iter` (or loosen `convergence`) and rerun.

Iteration counts are workload-dependent: heterogeneous real data routinely needs
several hundred iterations, so do not treat the default `max_iter=500` as
generous. Check `conv["per_batch"]["n_iter"]` against it rather than assuming.

```python
import pandas as pd
pd.DataFrame(conv["per_batch"]).sort_values("n_iter", ascending=False).head()
```

## After DecontX

See [Where DecontX goes in your workflow](#where-decontx-goes-in-your-workflow) for
where it slots in. Once it has run, `adata.layers["decontX_counts"]` holds
decontaminated counts and you re-normalize from those:

```python
adata.layers["raw_counts"] = adata.X.copy()      # keep the originals
adata.X = adata.layers["decontX_counts"]

sc.pp.normalize_total(adata)                      # normalize BEFORE log1p
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)                               # re-cluster on clean counts
sc.tl.rank_genes_groups(adata, "leiden")
```

Re-clustering after decontamination is optional. The clusters DecontX consumed only
need to be good enough to estimate each population's expression profile, but marker
detection and differential expression generally benefit from clusters derived from the
cleaned counts.

Counts are fractional by default, matching the R implementation. Most scanpy functions
accept that. If a downstream tool requires integers, either run with
`round_counts=True` or round explicitly — but note that rounding zeroes every native
count below 0.5, roughly 8 % of nonzero entries on typical data.

To inspect what was removed:

```python
removed = adata.layers["raw_counts"] - adata.layers["decontX_counts"]
```

## Citation

If you use DecontX in your research, please cite:

> Yang, S., Corbett, S.E., Koga, Y. et al. Decontamination of ambient RNA in single-cell RNA-seq with DecontX. Genome Biol 21, 57 (2020). https://doi.org/10.1186/s13059-020-1950-6

## License

MIT License
