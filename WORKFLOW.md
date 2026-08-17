# Workflow

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
