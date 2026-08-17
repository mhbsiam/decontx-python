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
    dummy_native_sums = np.zeros(10, dtype=np.float64)
    dummy_global_acc = np.zeros(20, dtype=np.float64)
    dummy_cell_order = np.ascontiguousarray(
        np.argsort(dummy_z, kind="stable"), dtype=np.int64
    )
    dummy_cluster_starts = np.zeros(4, dtype=np.int64)
    np.cumsum(np.bincount(dummy_z - 1, minlength=3), out=dummy_cluster_starts[1:])
    all_integral(dummy_data)
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
        dummy_cell_order,
        dummy_cluster_starts,
        True,
        dummy_delta,
        dummy_nc,
        dummy_phi_acc,
        dummy_native_sums,
        dummy_global_acc,
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
        dummy_nc,
    )
    decontx_log_likelihood_exact_sparse(
        dummy_data,
        dummy_indices,
        dummy_indptr,
        10,
        dummy_theta,
        dummy_eta,
        dummy_phi,
        dummy_z,
        1e-20,
    )


@jit(nopython=True, cache=True)
def all_integral(data):
    """Whether every value is a whole number, allocation-free.

    np.allclose(data, np.round(data)) would materialize two nnz-sized temporaries
    just to validate input.
    """
    for i in range(len(data)):
        v = data[i]
        if v != np.floor(v):
            return False
    return True


@jit(nopython=True, cache=True)
def _digamma(x):
    """Digamma. scipy.special is not callable from Numba nopython mode.

    Recurrence up to x >= 6, then the standard asymptotic series.
    """
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += np.log(x) - 0.5 * inv
    result -= inv2 * (
        1.0 / 12.0
        - inv2
        * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0)))
    )
    return result


@jit(nopython=True, cache=True)
def _trigamma(x):
    """Trigamma (first derivative of digamma), same approach as _digamma."""
    result = 0.0
    while x < 6.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += inv * (
        1.0
        + 0.5 * inv
        + inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)))
    )
    return result


@jit(nopython=True, cache=True)
def _inv_digamma(y):
    """Invert digamma by Newton iteration (Minka 2000, appendix C)."""
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        x = -1.0 / (y + 0.5772156649015329)
    for _ in range(6):
        x -= (_digamma(x) - y) / _trigamma(x)
    return x


@jit(nopython=True, cache=True)
def fit_dirichlet_2d(native_prop, contam_prop, alpha_init, max_iter=1000, tol=1e-10):
    """Minka's fixed-point MLE for a 2-component Dirichlet.

    Mirrors MCMCprecision::fit_dirichlet, which celda calls at DecontX.cpp:138
    on cbind(native_prop, contamination_prop). Returns the previous alpha
    unchanged if the data are degenerate, rather than clamping to a floor --
    the old [0.1, 1000] clamp is what let delta drop below 1.0 and invert the
    theta update.
    """
    n = len(native_prop)
    floor = 1e-12

    log_p0 = 0.0
    log_p1 = 0.0
    mean0 = 0.0
    for i in range(n):
        p0 = min(max(native_prop[i], floor), 1.0 - floor)
        p1 = min(max(contam_prop[i], floor), 1.0 - floor)
        log_p0 += np.log(p0)
        log_p1 += np.log(p1)
        mean0 += p0
    log_p0 /= n
    log_p1 /= n
    mean0 /= n

    var0 = 0.0
    for i in range(n):
        p0 = min(max(native_prop[i], floor), 1.0 - floor)
        d = p0 - mean0
        var0 += d * d
    var0 /= n

    if var0 <= 0.0 or mean0 <= 0.0 or mean0 >= 1.0:
        return alpha_init

    precision = mean0 * (1.0 - mean0) / var0 - 1.0
    if precision <= 0.0:
        return alpha_init

    a0 = mean0 * precision
    a1 = (1.0 - mean0) * precision

    for _ in range(max_iter):
        psi_sum = _digamma(a0 + a1)
        new_a0 = _inv_digamma(psi_sum + log_p0)
        new_a1 = _inv_digamma(psi_sum + log_p1)
        if not (np.isfinite(new_a0) and np.isfinite(new_a1)):
            return alpha_init
        if new_a0 <= 0.0 or new_a1 <= 0.0:
            return alpha_init
        delta_max = max(abs(new_a0 - a0), abs(new_a1 - a1))
        a0 = new_a0
        a1 = new_a1
        if delta_max < tol:
            break

    out = np.empty(2, dtype=np.float64)
    out[0] = a0
    out[1] = a1
    return out


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
    cell_order,
    cluster_starts,
    estimate_delta,
    delta,
    nc_data,
    phi_acc,
    native_sums,
    global_acc,
    pseudocount=1e-20,
):
    """Run one EM step. Fill nc_data and update theta, phi, eta in place."""
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

    # M-step native row sums. Each cell sums its own CSR slice in a fixed order,
    # so prange is both race-free and deterministic.
    for j in prange(n_cells):
        s = 0.0
        for idx in range(counts_indptr[j], counts_indptr[j + 1]):
            s += nc_data[idx]
        native_sums[j] = s

    if estimate_delta:
        # Dirichlet MLE on (native_prop, contamination_prop), matching
        # DecontX.cpp:131-139. R fits the pair jointly rather than doing
        # method-of-moments on the native proportion alone.
        contam_prop = np.empty(n_cells, dtype=np.float64)
        native_prop = np.empty(n_cells, dtype=np.float64)
        for j in range(n_cells):
            total = counts_colsums[j]
            if total > 0.0:
                c = (total - native_sums[j]) / total
            else:
                c = 0.0
            contam_prop[j] = c
            native_prop[j] = 1.0 - c

        delta = fit_dirichlet_2d(native_prop, contam_prop, delta)

    # Beta posterior mean, matching celda src/DecontX.cpp:119. The denominator
    # is positive for any non-negative counts and positive delta, so no guard
    # or clamp is needed (R has neither).
    for j in range(n_cells):
        theta[j] = (native_sums[j] + delta[0]) / (
            counts_colsums[j] + delta[0] + delta[1]
        )

    # Scatter-add nc_data into phi_acc, parallel over clusters.
    #
    # cell_order groups cells by cluster (stable sort) and cluster_starts indexes
    # into it, so thread k touches only phi_acc[k] -- race-free without per-thread
    # accumulator copies, which would cost n_threads * n_clusters * n_genes.
    # Because each cluster's cells are visited in a fixed order, the summation
    # order is identical on every run regardless of thread count, so this stays
    # bit-reproducible.
    phi_acc[:] = 0.0
    for k in prange(n_clusters):
        for ci in range(cluster_starts[k], cluster_starts[k + 1]):
            j = cell_order[ci]
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
        for g in range(n_genes):
            global_acc[g] = 0.0
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

    phi_acc = np.zeros((n_clusters, n_genes), dtype=data.dtype)
    global_acc = np.zeros(n_genes, dtype=data.dtype)

    for j in range(n_cells):
        k = z[j] - 1
        w = theta[j]
        for idx in range(indptr[j], indptr[j + 1]):
            g = indices[idx]
            wv = data[idx] * w
            phi_acc[k, g] += wv
            global_acc[g] += wv

    phi = np.zeros((n_clusters, n_genes), dtype=data.dtype)
    eta = np.zeros((n_clusters, n_genes), dtype=data.dtype)

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
    counts_data,
    counts_indices,
    counts_indptr,
    n_cells,
    n_genes,
    theta,
    phi,
    eta,
    z,
    nc_data,
) -> np.ndarray:
    """Compute final native counts at CSR nonzero positions.

    Write into the caller's nc_data buffer (the EM already holds one of exactly
    this size) and return it, rather than allocating a second nnz-sized array.
    The caller assembles the sparse output matrix from indptr and indices.
    """
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
