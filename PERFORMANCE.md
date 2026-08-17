# Performance tips

## Multi-batch processing

If your dataset has multiple batches, pass the batch column so each batch is processed separately:

```python
decontx.decontx(adata, cluster_key="leiden", batch_key="batch")
```

## Speed mode: skip log-likelihood history

When you only need the decontaminated counts and contamination estimates, you can skip the log-likelihood computation that runs every 10 iterations. This is typically **25–40 % faster** and does not change the contamination, theta, phi, eta, or decontaminated counts.

```python
decontx.decontx(adata, cluster_key="leiden", compute_log_likelihood=False)
```

The default is `True` to preserve existing behavior.

## Pre-warm the JIT compiler

Numba compiles the sparse kernels on the first call. To avoid paying that cost during the first real run, call `precompile()` once after importing:

```python
from decontx.fast_ops import precompile

precompile()

decontx.decontx(adata, cluster_key="leiden")
```
