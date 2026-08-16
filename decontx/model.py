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

    def fit_transform(self, X, z, X_background=None):
        """Run the EM loop. Keep X as CSR. Do not densify during EM.

        Native counts stay as a 1D array aligned to CSR nonzeros.
        The M-step scatter-adds from CSR positions into phi_acc.
        """
        np.random.seed(self.seed)

        if issparse(X):
            X_csr = X.tocsr().astype(np.float64)
        else:
            X_csr = csr_matrix(np.ascontiguousarray(X, dtype=np.float64))

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
        counts_data = np.ascontiguousarray(X_csr.data, dtype=np.float64)
        nnz = len(counts_data)

        counts_colsums = np.ascontiguousarray(
            np.asarray(X_csr.sum(axis=1)).ravel(), dtype=np.float64
        )

        # Theta is the native proportion. Prior is Beta(delta[0], delta[1]).
        theta = beta.rvs(
            self.delta[0], self.delta[1], size=n_cells, random_state=self.seed
        )
        theta = np.ascontiguousarray(theta, dtype=np.float64)

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
        estimate_eta = X_background is None
        pseudocount = 1e-20

        # Pre-allocate buffers. Reuse across all EM iterations.
        nc_data = np.zeros(nnz, dtype=np.float64)
        phi_acc = np.zeros((n_clusters, n_genes), dtype=np.float64)
        native_sums = np.zeros(n_cells, dtype=np.float64)
        global_acc = np.zeros(n_genes, dtype=np.float64)

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

                theta_change = np.max(np.abs(theta - theta_old))

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
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
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
        )
        nc_data_rounded = np.round(nc_data_final).astype(np.int32)

        decontaminated = csr_matrix(
            (nc_data_rounded, X_csr.indices.copy(), X_csr.indptr.copy()),
            shape=(n_cells, n_genes),
        )

        # Contamination is 1 - theta.
        contamination = 1.0 - theta

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
        }
