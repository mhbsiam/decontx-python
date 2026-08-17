"""DecontX Bayesian mixture model. EM inference per Yang et al. (2020)."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import beta

from .fast_ops import (
    calculate_native_matrix_fast_sparse,
    decontx_em_exact_sparse,
    decontx_initialize_exact_sparse,
    decontx_log_likelihood_exact_sparse,
)


class DecontXModel:
    """DecontX model. Theta is the native transcript proportion per cell."""

    def __init__(self, **kwargs):
        self.max_iter = kwargs.get("max_iter", 500)
        self.convergence_threshold = kwargs.get("convergence", 0.001)
        self.delta = np.array(kwargs.get("delta", [10.0, 10.0]))
        self.estimate_delta = kwargs.get("estimate_delta", True)
        self.iter_loglik = kwargs.get("iter_loglik", 10)
        self.compute_log_likelihood = kwargs.get("compute_log_likelihood", True)
        self.seed = kwargs.get("seed", 12345)
        self.verbose = kwargs.get("verbose", True)
        # float32 halves counts_data, nc_data and the CSR copy -- the three
        # largest allocations. Default stays float64.
        self.dtype = np.dtype(kwargs.get("dtype", np.float64))
        # Opt back into int32 output (~1.5x smaller in RAM and on disk) at the
        # cost of zeroing every native count below 0.5.
        self.round_counts = kwargs.get("round_counts", False)

    def fit_transform(self, X, z, X_background=None):
        """Run the EM loop. Keep X as CSR. Do not densify during EM.

        Native counts stay as a 1D array aligned to CSR nonzeros.
        The M-step scatter-adds from CSR positions into phi_acc.
        """
        np.random.seed(self.seed)

        if issparse(X):
            X_csr = X.tocsr()
            if X_csr.dtype != self.dtype:
                X_csr = X_csr.astype(self.dtype)
        else:
            X_csr = csr_matrix(np.ascontiguousarray(X, dtype=self.dtype))

        n_cells, n_genes = X_csr.shape
        z = np.ascontiguousarray(z, dtype=np.int32)
        n_clusters = len(np.unique(z))

        # Remap z to 1..n_clusters so sparse indexing is always in range.
        unique = np.unique(z)
        if unique[0] != 1 or unique[-1] != n_clusters or len(unique) != n_clusters:
            label_map = {label: i + 1 for i, label in enumerate(unique)}
            z = np.array([label_map[x] for x in z], dtype=np.int32)

        counts_indptr = np.ascontiguousarray(X_csr.indptr, dtype=np.int64)
        counts_indices = np.ascontiguousarray(X_csr.indices, dtype=np.int64)
        counts_data = np.ascontiguousarray(X_csr.data, dtype=self.dtype)
        nnz = len(counts_data)

        counts_colsums = np.ascontiguousarray(
            np.asarray(X_csr.sum(axis=1)).ravel(), dtype=self.dtype
        )

        # Group cells by cluster so the M-step scatter-add can run one cluster
        # per thread. Stable sort keeps the ordering (and thus the float
        # summation order) reproducible across runs.
        cell_order = np.ascontiguousarray(np.argsort(z, kind="stable"), dtype=np.int64)
        cluster_starts = np.zeros(n_clusters + 1, dtype=np.int64)
        np.cumsum(np.bincount(z - 1, minlength=n_clusters), out=cluster_starts[1:])

        # Theta is the native proportion. Prior is Beta(delta[0], delta[1]).
        theta = beta.rvs(
            self.delta[0], self.delta[1], size=n_cells, random_state=self.seed
        )
        theta = np.ascontiguousarray(theta, dtype=self.dtype)

        phi, eta = decontx_initialize_exact_sparse(
            counts_indptr,
            counts_indices,
            counts_data,
            n_cells,
            n_genes,
            theta,
            z,
            1e-20,
        )

        if X_background is not None:
            if issparse(X_background):
                bg_total = np.asarray(X_background.sum(axis=0)).ravel()
            else:
                bg_total = X_background.sum(axis=0)
            bg_sum = bg_total.sum()
            if bg_sum > 0:
                eta_bg = (bg_total + 1e-20) / (bg_sum + n_genes * 1e-20)
                eta = np.tile(eta_bg, (n_clusters, 1))

        log_likelihood_history = []
        n_iter = 0
        # Tracked explicitly: n_iter == max_iter is ambiguous on its own, since
        # a run can converge exactly on its final allowed iteration. Callers
        # need to tell "stopped because it converged" from "ran out of budget".
        converged = False
        theta_change = np.inf
        estimate_eta = X_background is None
        pseudocount = 1e-20

        # Pre-allocate buffers. Reuse across all EM iterations.
        nc_data = np.zeros(nnz, dtype=self.dtype)
        phi_acc = np.zeros((n_clusters, n_genes), dtype=self.dtype)
        native_sums = np.zeros(n_cells, dtype=self.dtype)
        global_acc = np.zeros(n_genes, dtype=self.dtype)

        for iteration in range(self.max_iter):
            n_iter = iteration + 1
            theta_old = theta.copy()

            theta, phi, eta, delta_new, contamination = decontx_em_exact_sparse(
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
                self.estimate_delta,
                self.delta.copy(),
                nc_data,
                phi_acc,
                native_sums,
                global_acc,
                pseudocount,
            )

            if self.estimate_delta:
                self.delta = delta_new

            # Test convergence every iteration, as R does. Sampling it only
            # every iter_loglik-th iteration overshot by up to 9 iterations and
            # made the stopping point depend on the diagnostic interval.
            theta_change = np.max(np.abs(theta - theta_old))

            if iteration % self.iter_loglik == 0:
                if self.compute_log_likelihood:
                    log_lik = decontx_log_likelihood_exact_sparse(
                        counts_data,
                        counts_indices,
                        counts_indptr,
                        n_cells,
                        theta,
                        eta,
                        phi,
                        z,
                        pseudocount,
                    )
                    log_likelihood_history.append(log_lik)

                if self.verbose and iteration % 10 == 0:
                    if self.compute_log_likelihood and log_likelihood_history:
                        print(
                            f"Iter {iteration}: LL={log_likelihood_history[-1]:.1f}, "
                            f"change={theta_change:.4f}, "
                            f"mean_contam={(1 - theta.mean()):.3f}"
                        )
                    else:
                        print(
                            f"Iter {iteration}: "
                            f"change={theta_change:.4f}, "
                            f"mean_contam={(1 - theta.mean()):.3f}"
                        )

            if theta_change < self.convergence_threshold:
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {n_iter}")
                break

        nc_data_final = calculate_native_matrix_fast_sparse(
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
        )
        # Counts stay fractional. R returns real-valued native counts and leaves
        # rounding to the caller; rounding here would zero out every entry below
        # 0.5, which is ~8% of nonzeros on typical data.
        if self.round_counts:
            out_data = np.round(nc_data_final).astype(np.int32)
        else:
            out_data = nc_data_final.copy()

        decontaminated = csr_matrix(
            (out_data, X_csr.indices.copy(), X_csr.indptr.copy()),
            shape=(n_cells, n_genes),
        )

        # Contamination is the E-step proportion (celda src/DecontX.cpp:115),
        # not 1 - theta. Theta is the prior-shrunk posterior mean, so 1 - theta
        # is pulled toward the delta prior. Empty cells have no measurable
        # contamination and are reported as 0.
        # Always from the unrounded native counts, and in float64 even when the
        # EM ran at float32 -- rounding is an output-format choice and must not
        # change the contamination estimate.
        # Blocked so this never materializes a float64 copy of the whole nnz
        # array (2.1 GB on a 260M-nonzero matrix). bincount accumulates in
        # float64 and handles empty rows without special-casing.
        native_totals = np.zeros(n_cells, dtype=np.float64)
        block = 8192
        for start in range(0, n_cells, block):
            stop = min(start + block, n_cells)
            lo = int(counts_indptr[start])
            hi = int(counts_indptr[stop])
            if hi <= lo:
                continue
            seg = np.asarray(nc_data_final[lo:hi], dtype=np.float64)
            rows = np.repeat(
                np.arange(stop - start), np.diff(counts_indptr[start : stop + 1])
            )
            native_totals[start:stop] = np.bincount(
                rows, weights=seg, minlength=stop - start
            )

        colsums64 = counts_colsums.astype(np.float64)
        contamination = (colsums64 - native_totals) / np.maximum(colsums64, 1.0)

        return {
            "contamination": contamination,
            "decontaminated_counts": decontaminated,
            "theta": theta,
            "phi": phi,
            "eta": eta,
            "delta": self.delta,
            "z": z,
            "log_likelihood_history": log_likelihood_history,
            "n_iter": n_iter,
            "converged": converged,
            "final_theta_change": float(theta_change),
        }
