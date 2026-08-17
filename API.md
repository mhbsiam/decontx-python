# API Reference

## Main Function

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

## Checking convergence

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
