# Performance tips

## Process multiple batches

If your dataset has multiple batches, pass the batch column. Each batch is processed separately.

```python
decontx.decontx(adata, cluster_key="leiden", batch_key="batch")
```

## Skip log-likelihood history for speed

If you only need the decontaminated counts and contamination estimates, skip the log-likelihood computation. The log-likelihood computation runs every 10 iterations. Skipping the log-likelihood computation is typically **25–40 % faster**. The contamination, theta, phi, eta, and decontaminated counts do not change.

```python
decontx.decontx(adata, cluster_key="leiden", compute_log_likelihood=False)
```

The default is `True`. The default `True` preserves existing behavior.

## Pre-warm the JIT compiler

Numba compiles the sparse kernels on the first call. Call `precompile()` once after importing. This avoids the compilation delay during the first real run.

```python
from decontx.fast_ops import precompile

precompile()

decontx.decontx(adata, cluster_key="leiden")
```
