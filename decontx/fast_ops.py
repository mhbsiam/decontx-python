"""Numba JIT kernels for the DecontX EM algorithm."""

from typing import Tuple

import numpy as np
from numba import jit, prange


def _precompile_functions():
    """Compile all JIT functions with dummy data. Call precompile() instead."""
    dummy_counts = np.random.rand(10, 20).astype(np.float64)
    dummy_z = np.array([1, 1, 2, 2, 3, 3, 1, 2, 3, 1], dtype=np.int32)
    dummy_theta = np.random.rand(10).astype(np.float64)
    dummy_phi = np.random.rand(3, 20).astype(np.float64)
    dummy_eta = np.random.rand(3, 20).astype(np.float64)
    dummy_delta = np.array([10.0, 10.0])
    dummy_colsums = dummy_counts.sum(axis=1)

    from scipy.sparse import csr_matrix as _csr

    dummy_sparse = _csr(dummy_counts)
    dummy_nc = np.zeros(dummy_sparse.nnz, dtype=np.float64)
    dummy_phi_acc = np.zeros((3, 20), dtype=np.float64)
    dummy_indptr = dummy_sparse.indptr.astype(np.int64)
    dummy_indices = dummy_sparse.indices.astype(np.int64)
    dummy_data = dummy_sparse.data.astype(np.float64)

    decontx_initialize_exact_sparse(
        dummy_indptr, dummy_indices, dummy_data, 10, 20, dummy_theta, dummy_z, 1e-20
    )
    decontx_em_exact_sparse(
        dummy_data,
        dummy_indices,
        dummy_indptr,
        10,
        20,
        dummy_colsums,
        dummy_theta,
        True,
        dummy_eta,
        dummy_phi,
        dummy_z,
        True,
        dummy_delta,
        dummy_nc,
        dummy_phi_acc,
        1e-20,
    )
    calculate_native_matrix_fast_sparse(
        dummy_data,
        dummy_indices,
        dummy_indptr,
        10,
        20,
        dummy_theta,
        dummy_phi,
        dummy_eta,
        dummy_z,
    )
    decontx_log_likelihood_exact_sparse(
        dummy_data,
        dummy_indices,
        dummy_indptr,
        10,
        20,
        dummy_theta,
        dummy_eta,
        dummy_phi,
        dummy_z,
        1e-20,
    )


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def decontx_em_exact_sparse(
    counts_data,
    counts_indices,
    counts_indptr,
    n_cells,
    n_genes,
    counts_colsums,
    theta,
    estimate_eta,
    eta,
    phi,
    z,
    estimate_delta,
    delta,
    nc_data,
    phi_acc,
    pseudocount=1e-20,
):
    """Run one EM step. Fill nc_data and update theta, phi, eta in place.

    nc_data holds native count estimates at CSR nonzero positions.
    phi_acc holds per-cluster gene sums. Both buffers are reused across iterations.
    """
    n_clusters = phi.shape[0]

    # E-step: compute native counts at each CSR nonzero position.
    # Each cell writes to its own indptr slice. Race-free with prange.
    for j in prange(n_cells):
        cluster_idx = z[j] - 1
        theta_j = theta[j]
        one_minus_theta = 1.0 - theta_j

        for idx in range(counts_indptr[j], counts_indptr[j + 1]):
            g = counts_indices[idx]
            count = counts_data[idx]
            p_native = theta_j * phi[cluster_idx, g]
            p_contam = one_minus_theta * eta[cluster_idx, g]
            total = p_native + p_contam + pseudocount
            nc_data[idx] = count * (p_native + pseudocount) / total

    # M-step: compute native row sums for theta update. Serial for determinism.
    native_sums = np.zeros(n_cells, dtype=np.float64)
    for j in range(n_cells):
        s = 0.0
        for idx in range(counts_indptr[j], counts_indptr[j + 1]):
            s += nc_data[idx]
        native_sums[j] = s

    if estimate_delta:
        proportions = native_sums / (counts_colsums + pseudocount)
        mean_prop = 0.0
        for j in range(n_cells):
            mean_prop += proportions[j]
        mean_prop /= n_cells

        var_prop = 0.0
        for j in range(n_cells):
            d = proportions[j] - mean_prop
            var_prop += d * d
        var_prop /= n_cells

        if var_prop > 0.0 and var_prop < mean_prop * (1.0 - mean_prop):
            precision = mean_prop * (1.0 - mean_prop) / var_prop - 1.0
            delta[0] = max(0.1, min(1000.0, mean_prop * precision))
            delta[1] = max(0.1, min(1000.0, (1.0 - mean_prop) * precision))

    for j in range(n_cells):
        t = (native_sums[j] + delta[0] - 1.0) / (
            counts_colsums[j] + delta[0] + delta[1] - 2.0
        )
        theta[j] = max(pseudocount, min(1.0 - pseudocount, t))

    # Scatter-add nc_data into phi_acc. Serial to prevent race conditions.
    phi_acc[:] = 0.0
    for j in range(n_cells):
        k = z[j] - 1
        for idx in range(counts_indptr[j], counts_indptr[j + 1]):
            g = counts_indices[idx]
            phi_acc[k, g] += nc_data[idx]

    for k in range(n_clusters):
        total = pseudocount * n_genes
        for g in range(n_genes):
            total += phi_acc[k, g]
        for g in range(n_genes):
            phi[k, g] = (phi_acc[k, g] + pseudocount) / total

    if estimate_eta:
        global_acc = np.zeros(n_genes, dtype=np.float64)
        for k in range(n_clusters):
            for g in range(n_genes):
                global_acc[g] += phi_acc[k, g]

        for k in range(n_clusters):
            total = pseudocount * n_genes
            for g in range(n_genes):
                other_native = global_acc[g] - phi_acc[k, g]
                total += other_native
            # Prevent division by zero when only one cluster exists.
            if total <= 0.0:
                for g in range(n_genes):
                    eta[k, g] = 1.0 / n_genes
            else:
                for g in range(n_genes):
                    other_native = global_acc[g] - phi_acc[k, g]
                    eta[k, g] = (other_native + pseudocount) / total

    contamination = 1.0 - theta
    return theta, phi, eta, delta, contamination


@jit(nopython=True, cache=True)
def decontx_initialize_exact_sparse(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_cells: int,
    n_genes: int,
    theta: np.ndarray,
    z: np.ndarray,
    pseudocount: float = 1e-20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize phi and eta from CSR nonzeros.

    phi holds theta-weighted native expression per cluster.
    eta holds theta-weighted native expression from all other clusters.
    """
    n_clusters = 0
    for j in range(n_cells):
        if z[j] > n_clusters:
            n_clusters = z[j]

    phi_acc = np.zeros((n_clusters, n_genes), dtype=np.float64)
    global_acc = np.zeros(n_genes, dtype=np.float64)

    for j in range(n_cells):
        k = z[j] - 1
        w = theta[j]
        for idx in range(indptr[j], indptr[j + 1]):
            g = indices[idx]
            wv = data[idx] * w
            phi_acc[k, g] += wv
            global_acc[g] += wv

    phi = np.zeros((n_clusters, n_genes), dtype=np.float64)
    eta = np.zeros((n_clusters, n_genes), dtype=np.float64)

    for k in range(n_clusters):
        phi_total = pseudocount * n_genes
        eta_total = pseudocount * n_genes
        for g in range(n_genes):
            phi_total += phi_acc[k, g]
            eta_total += global_acc[g] - phi_acc[k, g]

        if phi_total > 0:
            for g in range(n_genes):
                phi[k, g] = (phi_acc[k, g] + pseudocount) / phi_total
        else:
            for g in range(n_genes):
                phi[k, g] = 1.0 / n_genes

        if eta_total > 0:
            for g in range(n_genes):
                eta[k, g] = (global_acc[g] - phi_acc[k, g] + pseudocount) / eta_total
        else:
            for g in range(n_genes):
                eta[k, g] = 1.0 / n_genes

    return phi, eta


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calculate_native_matrix_fast_sparse(
    counts_data, counts_indices, counts_indptr, n_cells, n_genes, theta, phi, eta, z
) -> np.ndarray:
    """Compute final native counts at CSR nonzero positions.

    Return nc_data aligned to the input CSR nonzeros.
    The caller assembles the sparse output matrix from indptr and indices.
    """
    nc_data = np.zeros(len(counts_data), dtype=np.float64)

    for j in prange(n_cells):
        cluster = z[j] - 1
        theta_j = theta[j]
        one_minus_theta = 1.0 - theta_j

        for idx in range(counts_indptr[j], counts_indptr[j + 1]):
            g = counts_indices[idx]
            p_native = theta_j * phi[cluster, g] + 1e-20
            p_contam = one_minus_theta * eta[cluster, g] + 1e-20
            nc_data[idx] = counts_data[idx] * p_native / (p_native + p_contam)

    return nc_data


@jit(nopython=True, cache=True)
def decontx_log_likelihood_exact_sparse(
    counts_data,
    counts_indices,
    counts_indptr,
    n_cells,
    n_genes,
    theta,
    eta,
    phi,
    z,
    pseudocount=1e-20,
) -> float:
    """Compute log-likelihood over CSR nonzeros.

    The log-likelihood is used for convergence display only.
    Parameter updates use theta_change, not the log-likelihood value.
    """
    log_likelihood = 0.0

    for j in range(n_cells):
        cluster_idx = z[j] - 1
        for idx in range(counts_indptr[j], counts_indptr[j + 1]):
            g = counts_indices[idx]
            c = counts_data[idx]
            if c > 0:
                mixture = (
                    theta[j] * phi[cluster_idx, g]
                    + (1.0 - theta[j]) * eta[cluster_idx, g]
                    + pseudocount
                )
                log_likelihood += c * np.log(mixture)

    return log_likelihood


def precompile():
    """Compile all Numba JIT functions. Call this before benchmarking."""
    _precompile_functions()
