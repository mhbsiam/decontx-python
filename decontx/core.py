"""DecontX core functionality for scanpy integration."""

import warnings
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, vstack

from .fast_ops import all_integral
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
    dtype: str = "float64",
    round_counts: bool = False,
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

    original_labels = adata.obs[cluster_key].values
    z_labels, label_map = _process_cluster_labels(original_labels)

    if verbose:
        n_clusters = len(np.unique(z_labels))
        print(f"Processing {adata.n_obs} cells, {adata.n_vars} genes")
        print(f"Using {n_clusters} clusters from '{cluster_key}'")

    if batch_key is not None:
        if batch_key not in adata.obs:
            raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")

        # Cast to object and use pd.unique: np.unique raises TypeError on
        # mixed str/float (a categorical or object column carrying NaN), and
        # silently produces an unmatchable NaN "batch" on a float column.
        batch_series = pd.Series(np.asarray(adata.obs[batch_key].values, dtype=object))
        if batch_series.isna().any():
            n_missing = int(batch_series.isna().sum())
            raise ValueError(
                f"Batch key '{batch_key}' has {n_missing} missing value(s). "
                "Assign every cell to a batch or subset them out before running."
            )
        batch_labels = batch_series.to_numpy()
        unique_batches = pd.unique(batch_labels)

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
            dtype,
            round_counts,
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
            dtype,
            round_counts,
        )
        results = {"all": result}

    _store_results(adata, results, original_labels, cluster_key, batch_key)

    # Discriminate on batch_key, not on the "all" key: a real batch can be
    # named "all" and would otherwise be mistaken for the single-batch sentinel.
    if batch_key is None:
        fitted_delta = results["all"]["delta"]
    else:
        fitted_delta = np.mean(
            [np.asarray(r["delta"]) for r in results.values()], axis=0
        )
    if not isinstance(fitted_delta, np.ndarray):
        fitted_delta = np.asarray(fitted_delta)

    _store_metadata(
        adata,
        delta,
        fitted_delta,
        estimate_delta,
        max_iter,
        convergence,
        seed,
        start_time,
        label_map,
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

    # DecontX is a count model. Log-normalized input produces plausible-looking
    # but meaningless output.
    #
    # Integrality is the real evidence, not uns['log1p']: restoring raw counts
    # into .X after clustering leaves that key behind, and rejecting on it alone
    # would break the standard workflow. Raise only when both signals agree.
    data = adata.X.data if issparse(adata.X) else np.asarray(adata.X).ravel()
    is_integral = data.size == 0 or all_integral(
        np.ascontiguousarray(data, dtype=np.float64)
    )

    if not is_integral:
        if "log1p" in adata.uns:
            raise ValueError(
                "adata.X has non-integer values and adata.uns['log1p'] is set, so "
                "it appears to be log-transformed. DecontX requires raw counts: "
                "pass the counts layer (e.g. adata.X = adata.layers['counts']) "
                "or run DecontX before normalizing."
            )
        warnings.warn(
            "adata.X contains non-integer values. DecontX expects raw counts; "
            "normalized or transformed data will give unreliable results.",
            stacklevel=2,
        )

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


def _process_cluster_labels(z: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Convert cluster labels to sequential integers starting from 1.

    Return the remapped labels and the original->internal mapping. The integer
    path used to shift by the minimum rather than compact, which left gaps
    (e.g. [0, 5, 9] -> [1, 6, 10]); np.unique is already sorted, so one
    compaction path is correct for every dtype.
    """
    z = np.asarray(z)

    unique_labels = np.unique(z)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 clusters for decontamination")

    label_map = {label: i + 1 for i, label in enumerate(unique_labels)}
    return np.array([label_map[x] for x in z]).astype(int), label_map


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
    dtype: str = "float64",
    round_counts: bool = False,
) -> dict:
    """Process each batch separately."""
    batch_results = {}

    # Distinct child seed per batch. Passing the same seed to every batch makes
    # beta.rvs emit an identical theta init prefix, correlating fits that are
    # meant to be independent.
    child_seeds = [
        int(s.generate_state(1)[0] % (2**31 - 1))
        for s in np.random.SeedSequence(seed).spawn(len(unique_batches))
    ]

    for batch, batch_seed in zip(unique_batches, child_seeds):
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
            batch_seed,
            verbose=False,
            compute_log_likelihood=compute_log_likelihood,
            dtype=dtype,
            round_counts=round_counts,
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
    dtype: str = "float64",
    round_counts: bool = False,
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
        dtype=dtype,
        round_counts=round_counts,
    )

    result = model.fit_transform(X, z_labels)
    return result


def _store_results(
    adata: AnnData,
    results: dict,
    original_labels: np.ndarray,
    cluster_key: str,
    batch_key: Optional[str],
):
    """Store decontX results in the AnnData object."""
    n_cells = adata.n_obs
    n_genes = adata.n_vars

    if batch_key is None:
        result = results["all"]
        adata.layers["decontX_counts"] = result["decontaminated_counts"]
        adata.obs["decontX_contamination"] = result["contamination"]
    else:
        # Multiple batches: stack sparse results in original cell order.
        contamination = np.zeros(n_cells)
        covered = np.zeros(n_cells, dtype=bool)
        batch_matrices = {}

        for batch_name, result in results.items():
            if "batch_indices" in result:
                batch_indices = result["batch_indices"]
                batch_matrices[batch_name] = (
                    batch_indices,
                    result["decontaminated_counts"],
                )
                contamination[batch_indices] = result["contamination"]
                covered[batch_indices] = True

        # Without this, uncovered cells would silently keep contamination 0.0.
        if not covered.all():
            raise RuntimeError(
                f"{int((~covered).sum())} cell(s) were not assigned to any batch; "
                "cannot assemble decontaminated counts."
            )

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

    # Store the user's own labels. Writing the internal 1..K codes here silently
    # renumbered string clusters lexicographically ("10" -> 2, "2" -> 3), so the
    # column could not be joined against the source cluster column.
    adata.obs["decontX_clusters"] = pd.Categorical(original_labels)


def _store_metadata(
    adata: AnnData,
    input_delta: Tuple[float, float],
    fitted_delta: np.ndarray,
    estimate_delta: bool,
    max_iter: int,
    convergence: float,
    seed: int,
    start_time: datetime,
    label_map: dict,
):
    """Store run parameters and metadata.

    Merge into any existing uns['decontX'] rather than replacing it: this runs
    after _store_results, so a wholesale assignment would discard whatever that
    wrote.
    """
    end_time = datetime.now()

    metadata = dict(adata.uns.get("decontX", {}))
    metadata.update(
        {
            "parameters": {
                # The value the caller passed in, not the fitted one.
                "delta": list(input_delta),
                "estimate_delta": estimate_delta,
                "max_iter": max_iter,
                "convergence": convergence,
                "seed": seed,
            },
            "fitted": {
                "delta": list(np.asarray(fitted_delta)),
            },
            "cluster_map": {str(k): int(v) for k, v in label_map.items()},
            "runtime": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
            },
            "version": "0.2.0",
        }
    )

    adata.uns["decontX"] = metadata
