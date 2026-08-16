"""Utility functions for DecontX."""

import numpy as np
from anndata import AnnData


def initialize_clusters(
    adata: AnnData, var_genes: int = 5000, seed: int = 12345
) -> np.ndarray:
    """Initialize cell clusters with scanpy preprocessing and Leiden clustering.

    Call this function when no cluster labels are available.
    """
    import scanpy as sc

    adata_temp = adata.copy()

    sc.pp.highly_variable_genes(adata_temp, n_top_genes=var_genes)
    adata_temp = adata_temp[:, adata_temp.var.highly_variable]

    sc.pp.normalize_total(adata_temp)
    sc.pp.log1p(adata_temp)

    sc.pp.pca(adata_temp, random_state=seed)
    sc.pp.neighbors(adata_temp, random_state=seed)
    sc.tl.umap(adata_temp, random_state=seed)

    sc.tl.leiden(adata_temp, random_state=seed)

    return adata_temp.obs["leiden"].astype(int).values
