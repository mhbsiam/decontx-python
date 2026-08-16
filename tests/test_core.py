"""Basic tests for decontx functionality."""

import numpy as np
from anndata import AnnData

from decontx import decontx


def test_decontx_basic():
    """Test basic decontx functionality with cluster labels."""
    np.random.seed(42)
    n_cells, n_genes = 100, 50
    X = np.random.poisson(5, size=(n_cells, n_genes))
    adata = AnnData(X)
    adata.obs["leiden"] = np.repeat([1, 2, 3, 4], 25)

    result = decontx(adata, cluster_key="leiden", copy=True, verbose=False)

    assert "decontX_counts" in result.layers
    assert "decontX_contamination" in result.obs
    assert result.layers["decontX_counts"].shape == X.shape
    assert len(result.obs["decontX_contamination"]) == n_cells
