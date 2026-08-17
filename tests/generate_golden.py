"""Generate golden reference outputs from the current DecontX implementation.

Run once to capture theta, phi, eta, contamination, and decontX_counts
on a fixed synthetic dataset. These arrays are the regression baseline.

Usage:
    python tests/generate_golden.py
"""

import os

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, issparse

import decontx


def make_dataset(
    seed: int = 12345,
    n_cells: int = 400,
    n_genes: int = 300,
    n_clusters: int = 4,
    density: float = 0.15,
):
    """Build a deterministic synthetic scRNA-seq dataset with cluster structure.

    Each cluster has distinct marker genes. Contamination is added from
    other clusters' markers to give the EM real signal to separate.
    """
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(1, n_clusters + 1), n_cells // n_clusters)
    z = z[:n_cells]

    X = np.zeros((n_cells, n_genes), dtype=np.float64)
    markers_per_cluster = max(5, n_genes // (2 * n_clusters))

    for k in range(1, n_clusters + 1):
        cells_k = np.where(z == k)[0]
        marker_start = (k - 1) * markers_per_cluster
        marker_end = marker_start + markers_per_cluster
        for c in cells_k:
            X[c, marker_start:marker_end] = rng.poisson(8, size=markers_per_cluster)
            n_bg = int(density * n_genes) - markers_per_cluster
            if n_bg > 0:
                bg_idx = rng.choice(
                    [g for g in range(n_genes) if not (marker_start <= g < marker_end)],
                    size=n_bg,
                    replace=False,
                )
                X[c, bg_idx] = rng.poisson(1, size=n_bg)
        for c in cells_k:
            other_clusters = [kk for kk in range(1, n_clusters + 1) if kk != k]
            kk = rng.choice(other_clusters)
            om_start = (kk - 1) * markers_per_cluster
            om_end = om_start + markers_per_cluster
            contam_idx = rng.choice(
                np.arange(om_start, om_end),
                size=max(1, markers_per_cluster // 3),
                replace=False,
            )
            X[c, contam_idx] += rng.poisson(2, size=len(contam_idx))

    X = csr_matrix(X)
    adata = ad.AnnData(X=X)
    adata.obs["leiden"] = pd_Categorical(z)
    return adata, z


def pd_Categorical(z):
    import pandas as pd

    return pd.Categorical(z)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "golden")
    os.makedirs(out_dir, exist_ok=True)

    adata, z = make_dataset()
    print(
        f"Dataset: {adata.n_obs} cells x {adata.n_vars} genes, "
        f"{adata.X.nnz} non-zeros ({adata.X.nnz / (adata.n_obs * adata.n_vars):.1%} density)"
    )
    print(f"Clusters: {np.unique(z)}")

    print("Running DecontX (golden run)...")
    result = decontx.decontx(
        adata, cluster_key="leiden", copy=True, verbose=True, seed=12345, max_iter=200
    )

    contamination = result.obs["decontX_contamination"].values
    decontx_counts = result.layers["decontX_counts"]
    if issparse(decontx_counts):
        decontx_counts = decontx_counts.toarray()

    from decontx.model import DecontXModel

    X = adata.X
    z_int = np.ascontiguousarray(z, dtype=np.int32)

    model = DecontXModel(max_iter=200, convergence=0.001, seed=12345, verbose=False)
    res = model.fit_transform(X, z_int)

    np.save(os.path.join(out_dir, "theta.npy"), res["theta"])
    np.save(os.path.join(out_dir, "phi.npy"), res["phi"])
    np.save(os.path.join(out_dir, "eta.npy"), res["eta"])
    np.save(os.path.join(out_dir, "contamination.npy"), contamination)
    np.save(os.path.join(out_dir, "decontx_counts.npy"), np.asarray(decontx_counts))
    np.save(os.path.join(out_dir, "z.npy"), z_int)
    np.save(os.path.join(out_dir, "delta.npy"), np.asarray(res["delta"]))
    np.save(os.path.join(out_dir, "X_dense.npy"), X.toarray())
    from scipy.sparse import save_npz

    save_npz(os.path.join(out_dir, "X_sparse.npz"), adata.X)

    print("\nGolden reference saved to", out_dir)
    print(f"  theta:         shape={res['theta'].shape} mean={res['theta'].mean():.4f}")
    print(f"  phi:           shape={res['phi'].shape}")
    print(f"  eta:           shape={res['eta'].shape}")
    print(
        f"  contamination: shape={contamination.shape} mean={contamination.mean():.4f}"
    )
    print(f"  decontx_counts: shape={np.asarray(decontx_counts).shape}")
    print(f"  delta:         {res['delta']}")


if __name__ == "__main__":
    main()
