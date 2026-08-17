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
- `adata`: AnnData object. It must hold raw counts in `.X`.
- `cluster_key`: Column in `.obs` with cluster labels.
- `batch_key`: Optional column in `.obs` with batch labels. Batches are processed separately. Default: `None`.
- `max_iter`: Maximum number of EM iterations. Default: 500.
- `delta`: Beta prior parameters for contamination. Default: `(10.0, 10.0)`.
- `estimate_delta`: If `True`, estimate the delta parameters. Default: `True`.
- `convergence`: Convergence threshold. Default: 0.001.
- `seed`: Random seed for theta initialization. Default: 12345.
- `copy`: If `True`, return a copy. If `False`, modify `adata` in place. Default: `False`.
- `verbose`: If `True`, print progress to `stdout`. Default: `True`.
- `compute_log_likelihood`: If `True`, compute the log-likelihood every 10 iterations for the progress output. Set `compute_log_likelihood` to `False` to run faster. The log-likelihood is a diagnostic only. It does not affect convergence or the results. Default: `True`.
- `dtype`: Working precision. Use `"float64"` or `"float32"`. `float32` halves the largest allocations and the output layer. It is approximately 30 % faster per iteration. The measured cost: maximum absolute error is 2.4e-08 against the R reference. `float64` has a maximum absolute error of 7.2e-16. Default: `"float64"`.
- `round_counts`: If `True`, round decontaminated counts to `int32`. This reduces the output layer by approximately 1.5x in RAM and on disk. It zeroes every native count below 0.5. Contamination estimates do not change. Default: `False`.

**Returns:**
`None` if `copy=False` (default). Otherwise a copy of `adata`. The function stores results in `adata`:
- `adata.obs['decontX_contamination']`: Per-cell contamination estimates. The function computes these as `(total counts - native counts) / total counts`. This matches the R implementation.
- `adata.obs['decontX_clusters']`: The cluster labels from the caller.
- `adata.layers['decontX_counts']`: Decontaminated count matrix. These counts are fractional, as in R. Round them yourself if you need integers. Rounding zeroes every entry below 0.5.
- `adata.uns['decontX']`: Run metadata.
  - `'parameters'`: The arguments you passed (`delta`, `estimate_delta`, `max_iter`, `convergence`, `seed`).
  - `'fitted'`: `{'delta': ...}`. This is the fitted beta prior. It equals the input when `estimate_delta=False`. When `batch_key` is set, the function returns the mean across batches.
  - `'convergence'`: Whether the EM actually converged. See below.
  - `'cluster_map'`: Your cluster labels mapped to the internal `1..K` codes.
  - `'runtime'`: Start and end timestamps and the duration.
  - `'version'`: Package version.

## Checking convergence

`uns['decontX']['convergence']` records why the EM stopped. This record is reported per batch.

```python
conv = adata.uns["decontX"]["convergence"]
conv["all_converged"]            # False if any batch hit max_iter
conv["n_batches_not_converged"]
conv["per_batch"]                # parallel lists: batch, n_iter,
                                 # converged, final_theta_change
```

If any batch fails to converge, DecontX emits a `UserWarning`. The warning names the batch. It fires even when `verbose=False`. Non-convergence cannot pass unnoticed.

Estimates for cells in a non-converged batch are unreliable. Raise `max_iter` or loosen `convergence` and rerun.

Iteration counts depend on the workload. Heterogeneous real data often needs several hundred iterations. Do not treat the default `max_iter=500` as generous. Check `conv["per_batch"]["n_iter"]` against it. Do not assume convergence.

```python
import pandas as pd
pd.DataFrame(conv["per_batch"]).sort_values("n_iter", ascending=False).head()
```
