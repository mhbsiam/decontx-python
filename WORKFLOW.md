# Workflow

## Where DecontX fits in your workflow

DecontX is a count model. It needs two things:
- raw integer counts in `adata.X`
- cluster labels in `adata.obs`

The diagram below shows where DecontX fits:

```
load  →  QC filter  →  cluster (on normalized data)  →  DecontX (on raw counts)  →  normalize → analyze
                                                        ▲
                                        raw counts in .X, labels in .obs
```

You must cluster on normalized data. DecontX needs raw counts. So normalization must not overwrite `.X` before DecontX runs. Use one of the two methods below.

## Option 1: cluster on a copy

`adata.X` always stays as raw counts. You do not need to restore anything. You cannot get the order wrong.

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

# 3. Run DecontX on raw counts
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

## Option 2: stash counts in a layer

This option avoids the full copy. Preprocess in place. Then put the raw counts back before you run DecontX.

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

## Use existing cluster labels

If you already have cluster labels, skip straight to DecontX. Any `obs` column works, including `cell_type` annotations:

```python
decontx.decontx(adata, cluster_key="cell_type")
```

## Common errors

DecontX validates its input. It reports problems with the input. But three easy errors are worth stating outright:

- **Running DecontX after `sc.pp.log1p`.** This raises `ValueError`. DecontX models counts. Log values have no meaning to it. Run DecontX earlier, or restore the counts layer first.
- **Running DecontX after `sc.pp.normalize_total` only.** This emits a warning about non-integer values. DecontX does not raise because normalized-but-not-logged data is harder to detect with certainty. The results are still unreliable. Fix the ordering.
- **Subsetting to highly variable genes first.** DecontX estimates the ambient profile from the full transcriptome. Run it on all genes, then subset.

You can filter cells and genes before DecontX. This is recommended. Empty droplets and never-detected genes only add noise to the ambient estimate.

## After you run DecontX

`adata.layers["decontX_counts"]` holds the decontaminated counts. Re-normalize from those counts:

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

Re-clustering after decontamination is optional. DecontX only needs clusters that are good enough to estimate each population's expression profile. Marker detection and differential expression usually benefit from clusters derived from the cleaned counts.

Counts are fractional by default. This matches the R implementation. Most scanpy functions accept fractional counts. If a downstream tool needs integers, either run with `round_counts=True` or round explicitly. Rounding zeroes every native count below 0.5. This removes roughly 8 % of nonzero entries on typical data.

To inspect what was removed:

```python
removed = adata.layers["raw_counts"] - adata.layers["decontX_counts"]
```
