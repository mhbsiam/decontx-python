"""DecontX core functionality for scanpy integration."""

import warnings
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, vstack

from .model import DecontXModel


def decontx(
    adata: AnnData,
    cluster_key: str = "leiden",
    batch_key: Optional[str] = None,
    max_iter: int = 500,
    delta: Tuple[float, float] = (10.0, 10.0),
    estimate_delta: bool = True,
    convergence: float = 0.001,
    seed: int = 12345,
    copy: bool = False,
    verbose: bool = True,
    compute_log_likelihood: bool = True,
) -> Optional[AnnData]:
    """Remove ambient RNA contamination from single-cell RNA-seq data.

    Store results in the AnnData object:
    - adata.obs['decontX_contamination']: per-cell contamination estimates.
    - adata.layers['decontX_counts']: decontaminated count matrix.
    - adata.uns['decontX']: model parameters and run information.
    """
    if copy:
        adata = adata.copy()

    start_time = datetime.now()

    if verbose:
        print("=" * 50)
        print("Starting DecontX")
        print("=" * 50)

    _validate_inputs(adata, cluster_key, batch_key)

    if cluster_key not in adata.obs:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")

    z_labels = adata.obs[cluster_key].values
    z_labels = _process_cluster_labels(z_labels)

    if verbose:
        n_clusters = len(np.unique(z_labels))
        print(f"Processing {adata.n_obs} cells, {adata.n_vars} genes")
        print(f"Using {n_clusters} clusters from '{cluster_key}'")

    if batch_key is not None:
        if batch_key not in adata.obs:
            raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")

        batch_labels = adata.obs[batch_key].values
        unique_batches = np.unique(batch_labels)

        if verbose:
            print(f"Processing {len(unique_batches)} batches separately")

        results = _process_batches(
            adata,
            z_labels,
            batch_labels,
            unique_batches,
            max_iter,
            delta,
            estimate_delta,
            convergence,
            seed,
            verbose,
            compute_log_likelihood,
        )
    else:
        if verbose:
            print("Processing as single batch")

        result = _run_decontx_single(
            adata.X,
            z_labels,
            max_iter,
            delta,
            estimate_delta,
            convergence,
            seed,
            verbose,
            compute_log_likelihood,
        )
        results = {"all": result}

    _store_results(adata, results, z_labels, cluster_key, batch_key)

    if "all" in results:
        fitted_delta = results["all"]["delta"]
    else:
        fitted_delta = np.mean(
            [np.asarray(r["delta"]) for r in results.values()], axis=0
        )
    if not isinstance(fitted_delta, np.ndarray):
        fitted_delta = np.asarray(fitted_delta)

    _store_metadata(
        adata, fitted_delta, estimate_delta, max_iter, convergence, seed, start_time
    )

    if verbose:
        contamination = adata.obs["decontX_contamination"]
        print(f"Mean contamination: {contamination.mean():.1%}")
        print(f"Highly contaminated cells (>50%): {(contamination > 0.5).sum()}")

        end_time = datetime.now()
        print("=" * 50)
        print(f"Completed DecontX in {end_time - start_time}")
        print("=" * 50)

    if copy:
        return adata
    return None


def _validate_inputs(adata: AnnData, cluster_key: str, batch_key: Optional[str]):
    """Validate input data and parameters."""
    if adata.X.min() < 0:
        raise ValueError("Count matrix contains negative values")

    if issparse(adata.X):
        if np.any(np.isnan(adata.X.data)):
            raise ValueError("Count matrix contains NaN values")
    else:
        if np.any(np.isnan(adata.X)):
            raise ValueError("Count matrix contains NaN values")

    if adata.n_obs < 10:
        warnings.warn(
            "Very few cells (<10) detected. Results may be unreliable.",
            stacklevel=2,
        )

    if adata.n_vars < 100:
        warnings.warn(
            "Very few genes (<100) detected. Results may be unreliable.",
            stacklevel=2,
        )


def _process_cluster_labels(z: np.ndarray) -> np.ndarray:
    """Convert cluster labels to sequential integers starting from 1."""
    z = np.asarray(z)

    unique_labels = np.unique(z)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 clusters for decontamination")

    if not np.issubdtype(z.dtype, np.integer):
        label_map = {label: i + 1 for i, label in enumerate(unique_labels)}
        z = np.array([label_map[x] for x in z])
    else:
        min_label = np.min(z)
        if min_label <= 0:
            z = z - min_label + 1
        elif min_label > 1:
            label_map = {label: i + 1 for i, label in enumerate(np.sort(unique_labels))}
            z = np.array([label_map[x] for x in z])

    return z.astype(int)


def _process_batches(
    adata: AnnData,
    z_labels: np.ndarray,
    batch_labels: np.ndarray,
    unique_batches: np.ndarray,
    max_iter: int,
    delta: Tuple[float, float],
    estimate_delta: bool,
    convergence: float,
    seed: int,
    verbose: bool,
    compute_log_likelihood: bool,
) -> dict:
    """Process each batch separately."""
    batch_results = {}

    for batch in unique_batches:
        if verbose:
            print(f"  Processing batch '{batch}'...")

        batch_mask = batch_labels == batch
        batch_indices = np.where(batch_mask)[0]

        if issparse(adata.X):
            X_batch = adata.X[batch_mask].tocsr()
        else:
            X_batch = adata.X[batch_mask]

        z_batch = z_labels[batch_mask]

        result = _run_decontx_single(
            X_batch,
            z_batch,
            max_iter,
            delta,
            estimate_delta,
            convergence,
            seed,
            verbose=False,
            compute_log_likelihood=compute_log_likelihood,
        )

        result["batch_indices"] = batch_indices
        result["batch_name"] = batch
        batch_results[batch] = result

        if verbose:
            contamination = result["contamination"]
            print(f"    Mean contamination: {contamination.mean():.1%}")

    return batch_results


def _run_decontx_single(
    X: Union[np.ndarray, "csr_matrix"],
    z_labels: np.ndarray,
    max_iter: int,
    delta: Tuple[float, float],
    estimate_delta: bool,
    convergence: float,
    seed: int,
    verbose: bool = True,
    compute_log_likelihood: bool = True,
) -> dict:
    """Run DecontX on a single batch."""
    model = DecontXModel(
        max_iter=max_iter,
        delta=delta,
        estimate_delta=estimate_delta,
        convergence=convergence,
        seed=seed,
        verbose=verbose,
        compute_log_likelihood=compute_log_likelihood,
    )

    result = model.fit_transform(X, z_labels)
    return result


def _store_results(
    adata: AnnData,
    results: dict,
    z_labels: np.ndarray,
    cluster_key: str,
    batch_key: Optional[str],
):
    """Store decontX results in the AnnData object."""
    n_cells = adata.n_obs
    n_genes = adata.n_vars

    if len(results) == 1 and "all" in results:
        result = results["all"]
        adata.layers["decontX_counts"] = result["decontaminated_counts"]
        adata.obs["decontX_contamination"] = result["contamination"]
    else:
        # Multiple batches: stack sparse results in original cell order.
        contamination = np.zeros(n_cells)
        batch_matrices = {}

        for batch_name, result in results.items():
            if "batch_indices" in result:
                batch_indices = result["batch_indices"]
                batch_matrices[batch_name] = (
                    batch_indices,
                    result["decontaminated_counts"],
                )
                contamination[batch_indices] = result["contamination"]

        ordered = sorted(batch_matrices.values(), key=lambda x: x[0][0])
        if ordered:
            stacked = vstack([m for _, m in ordered], format="csr")
            all_indices = np.concatenate([idx for idx, _ in ordered])
            inv_perm = np.argsort(all_indices)
            decontx_counts = stacked[inv_perm]
        else:
            decontx_counts = csr_matrix((n_cells, n_genes))

        adata.layers["decontX_counts"] = decontx_counts
        adata.obs["decontX_contamination"] = contamination

    adata.obs["decontX_clusters"] = pd.Categorical(z_labels)


def _store_metadata(
    adata: AnnData,
    delta: np.ndarray,
    estimate_delta: bool,
    max_iter: int,
    convergence: float,
    seed: int,
    start_time: datetime,
):
    """Store run parameters and metadata."""
    end_time = datetime.now()

    metadata = {
        "parameters": {
            "delta": list(delta),
            "estimate_delta": estimate_delta,
            "max_iter": max_iter,
            "convergence": convergence,
            "seed": seed,
        },
        "runtime": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
        },
        "version": "0.2.0",
    }

    adata.uns["decontX"] = metadata
