"""Benchmark DecontX on synthetic datasets of increasing size.

All inputs go through the CSR-aligned sparse EM path.

Usage:
    python tests/benchmark.py
"""

import time

import numpy as np
from scipy.sparse import csr_matrix

from decontx.fast_ops import precompile
from decontx.model import DecontXModel


def make_dataset(seed=12345, n_cells=2000, n_genes=2000, n_clusters=5, density=0.10):
    """Build a deterministic synthetic scRNA-seq dataset with cluster structure."""
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(1, n_clusters + 1), n_cells // n_clusters)
    z = z[:n_cells]

    X = np.zeros((n_cells, n_genes), dtype=np.float64)
    markers_per_cluster = max(10, n_genes // (2 * n_clusters))

    for k in range(1, n_clusters + 1):
        cells_k = np.where(z == k)[0]
        ms = (k - 1) * markers_per_cluster
        me = ms + markers_per_cluster
        for c in cells_k:
            X[c, ms:me] = rng.poisson(8, size=markers_per_cluster)
            n_bg = int(density * n_genes) - markers_per_cluster
            if n_bg > 0:
                bg_idx = rng.choice(
                    [g for g in range(n_genes) if not (ms <= g < me)],
                    size=n_bg,
                    replace=False,
                )
                X[c, bg_idx] = rng.poisson(1, size=n_bg)
        for c in cells_k:
            others = [kk for kk in range(1, n_clusters + 1) if kk != k]
            kk = rng.choice(others)
            oms = (kk - 1) * markers_per_cluster
            ome = oms + markers_per_cluster
            ci = rng.choice(
                np.arange(oms, ome),
                size=max(1, markers_per_cluster // 3),
                replace=False,
            )
            X[c, ci] += rng.poisson(2, size=len(ci))

    return csr_matrix(X), z


def benchmark_one(n_cells, n_genes, n_clusters=5, density=0.10, max_iter=100):
    """Benchmark one dataset size."""
    X_sparse, z = make_dataset(
        n_cells=n_cells, n_genes=n_genes, n_clusters=n_clusters, density=density
    )
    z_int = np.ascontiguousarray(z, dtype=np.int32)

    nnz = X_sparse.nnz
    density_actual = nnz / (n_cells * n_genes)

    warmup = DecontXModel(max_iter=5, convergence=0.1, seed=999, verbose=False)
    warmup_idx = []
    for k in range(1, n_clusters + 1):
        warmup_idx.extend(np.where(z == k)[0][:10])
    warmup_idx = np.array(warmup_idx[:40])
    _ = warmup.fit_transform(
        X_sparse[warmup_idx].tocsr(),
        np.ascontiguousarray(z[warmup_idx], dtype=np.int32),
    )

    model = DecontXModel(max_iter=max_iter, convergence=1e-6, seed=12345, verbose=False)
    t0 = time.perf_counter()
    res = model.fit_transform(X_sparse, z_int)
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print(f"Dataset: {n_cells} cells x {n_genes} genes, {n_clusters} clusters")
    print(f"Non-zeros: {nnz:,} ({density_actual:.1%} density)")
    print(f"EM iterations: {res['n_iter']}")
    print(f"{'-' * 60}")
    print(f"Time:    {elapsed:.3f}s")
    print(f"Mean contamination: {res['contamination'].mean():.3f}")

    return {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "nnz": nnz,
        "density": density_actual,
        "time": elapsed,
        "n_iter": res["n_iter"],
    }


def main():
    print("Precompiling Numba JIT functions...")
    t0 = time.perf_counter()
    precompile()
    print(f"Precompilation done in {time.perf_counter() - t0:.1f}s")

    configs = [
        (500, 500, 4, 0.15, 100),
        (2000, 2000, 5, 0.10, 100),
        (5000, 3000, 8, 0.08, 50),
        (10000, 5000, 10, 0.05, 30),
    ]

    results = []
    for n_cells, n_genes, n_clusters, density, max_iter in configs:
        try:
            r = benchmark_one(n_cells, n_genes, n_clusters, density, max_iter)
            results.append(r)
        except Exception as e:
            print(f"\nFAILED for {n_cells}x{n_genes}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Cells':>8} {'Genes':>8} {'Density':>8} {'Time(s)':>10} {'Iters':>6}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['n_cells']:>8} {r['n_genes']:>8} {r['density']:>7.1%} "
            f"{r['time']:>10.3f} {r['n_iter']:>6}"
        )


if __name__ == "__main__":
    main()
