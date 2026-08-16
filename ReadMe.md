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

### Code quality

- Removed ~700 lines of dead code across `fast_ops.py`, `utils.py`,
  `core.py`, and `model.py`.
- Fixed the delta metadata bug: `adata.uns['decontX']['parameters']['delta']`
  now stores the fitted delta, not the input prior.
- Fixed multi-batch output: sparse batch results are stacked with
  `scipy.sparse.vstack` instead of densified into a zero-filled array.
- Fixed multi-batch cluster indexing: each batch's cluster labels are now
  remapped to a contiguous `1..n_clusters` range internally, preventing
  out-of-bounds writes when a batch contains only a subset of the global
  clusters.
- Fixed the single-cluster division-by-zero edge case in the eta update.
- Added division-by-zero guards in the theta and delta precision updates.
- Fixed `test_core.py` (wrong key casing, missing cluster column).
- Lazy JIT compilation: import-time compilation removed; first call compiles.
- Ruff lint and format pass clean on all files.

### Acknowledgments

The following architectural improvements were adopted from
[jjia1/decontx-python](https://github.com/jjia1/decontx-python):

- Keeping native counts as a 1D array (`nc_data`) aligned to CSR nonzeros
  instead of a dense `n_cells x n_genes` matrix.
- Scatter-add M-step: accumulating `nc_data` directly into `phi_acc` in JIT
  instead of Python-level boolean masking or GEMM.
- Computing eta from `global_acc - phi_acc[k]` instead of summing over an
  `other_mask`.
- Combining the E-step and M-step into a single monolithic Numba function call.
- Returning decontaminated counts as a sparse CSR matrix with the same
  sparsity pattern as the input.
- The overall "keep everything CSR" approach: convert all input to CSR
  upfront and never densify during the EM loop.

We thank the author for these improvement ideas, which made the
optimization possible.


## Installation

### Latest version from GitHub

```bash
pip install git+https://github.com/NiRuff/decontx-python.git
```

This installs the current `main` branch directly. Use this if you need the newest changes (for example, the performance improvements or `compute_log_likelihood` described below).

### Editable local install (for development)

```bash
git clone https://github.com/NiRuff/decontx-python.git
cd decontx-python
pip install -e .
```

An editable install makes Python import the local source. If the package was previously installed in the active environment from a different path, uninstall it first with `pip uninstall decontx-python` so `import decontx` resolves to the local copy.

## Quick Start

```python
import scanpy as sc
import decontx

# Load and preprocess data with scanpy
adata = sc.read_h5ad("pbmc.h5ad")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)

# Remove ambient RNA contamination
decontx.decontx(adata, cluster_key="leiden")

# Access results
contamination = adata.obs["decontX_contamination"]
clean_counts = adata.layers["decontX_counts"]

print(f"Mean contamination: {contamination.mean():.1%}")
print(f"Highly contaminated cells (>50%): {(contamination > 0.5).sum()}")
```

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

Our Python implementation achieves near-perfect concordance with the original R version:

<img width="2250" height="1500" alt="decontx_comparison_real_data" src="https://github.com/user-attachments/assets/fc1358fa-1f54-42d9-953d-d2281d90d2d5" />

**PBMC 3K Dataset Results:**
- **Correlation: 0.999** between R and Python implementations
- **Mean Absolute Error: <1%** across all parameter settings
- **Identical statistical properties** (mean, std, range) 
- **Per-cluster consistency** maintained across cell types

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
- `compute_log_likelihood`: Append log-likelihood to `uns['decontX']['log_likelihood_history']` every 10 iterations; set to `False` for faster runs when the history is not needed (default: `True`)

**Returns:**
`None` if `copy=False` (default), otherwise a copy of `adata`. Results stored in `adata`:
- `adata.obs['decontX_contamination']`: Per-cell contamination estimates
- `adata.layers['decontX_counts']`: Decontaminated count matrix
- `adata.uns['decontX']`: Model parameters and metadata
  - `'contamination'`: same as `adata.obs['decontX_contamination']`
  - `'delta'`: fitted beta prior (or input prior if `estimate_delta=False`)
  - `'log_likelihood_history'`: list of log-likelihood values every 10 iterations
    (only if `compute_log_likelihood=True`)
  - `'n_iter'`: number of EM iterations actually run

## Integration with Existing Workflows

DecontX fits naturally into scanpy workflows:

```python
# Standard scanpy analysis
sc.tl.leiden(adata, resolution=0.5)
sc.tl.rank_genes_groups(adata, "leiden")

# Add decontamination
decontx.decontx(adata, cluster_key="leiden")

# Continue with decontaminated data
adata.X = adata.layers["decontX_counts"]
sc.pp.log1p(adata)  # Re-log transform clean counts
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=50)
```

## Citation

If you use DecontX in your research, please cite:

> Yang, S., Corbett, S.E., Koga, Y. et al. Decontamination of ambient RNA in single-cell RNA-seq with DecontX. Genome Biol 21, 57 (2020). https://doi.org/10.1186/s13059-020-1950-6

## License

MIT License
