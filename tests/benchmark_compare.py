"""Head-to-head benchmark: this implementation vs jjia1's.

Both packages are named decontx. Run them in subprocesses to avoid conflicts.
"""

import json
import os
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix


def make_dataset(seed=12345, n_cells=5000, n_genes=3000, n_clusters=8, density=0.08):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(1, n_clusters + 1), n_cells // n_clusters)
    z = z[:n_cells]
    X = np.zeros((n_cells, n_genes), dtype=np.float64)
    mpc = max(10, n_genes // (2 * n_clusters))
    for k in range(1, n_clusters + 1):
        cells_k = np.where(z == k)[0]
        ms = (k - 1) * mpc
        me = ms + mpc
        for c in cells_k:
            X[c, ms:me] = rng.poisson(8, size=mpc)
            n_bg = int(density * n_genes) - mpc
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
            oms = (kk - 1) * mpc
            ome = oms + mpc
            ci = rng.choice(np.arange(oms, ome), size=max(1, mpc // 3), replace=False)
            X[c, ci] += rng.poisson(2, size=len(ci))
    return csr_matrix(X), z


def run_in_subprocess(pkg_path, script):
    """Run a script in a subprocess with the given package path prepended."""
    env = os.environ.copy()
    env["PYTHONPATH"] = pkg_path + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Subprocess failed: {result.returncode}")
    return result.stdout


def benchmark_subprocess(
    pkg_path, n_cells, n_genes, n_clusters, density, max_iter, label
):
    """Run benchmark in a subprocess with the given decontx package."""
    script = f"""
import time, json, numpy as np
from scipy.sparse import csr_matrix
import sys
sys.path.insert(0, {repr(pkg_path)})
# Force reimport
if 'decontx' in sys.modules:
    del sys.modules['decontx']
if 'decontx.model' in sys.modules:
    del sys.modules['decontx.model']
if 'decontx.fast_ops' in sys.modules:
    del sys.modules['decontx.fast_ops']
if 'decontx.core' in sys.modules:
    del sys.modules['decontx.core']

from decontx.model import DecontXModel

# Make dataset (deterministic)
import numpy as np
from scipy.sparse import csr_matrix
rng = np.random.default_rng(12345)
n_cells, n_genes, n_clusters, density = {n_cells}, {n_genes}, {n_clusters}, {density}
z = np.repeat(np.arange(1, n_clusters + 1), n_cells // n_clusters)[:n_cells]
X = np.zeros((n_cells, n_genes), dtype=np.float64)
mpc = max(10, n_genes // (2 * n_clusters))
for k in range(1, n_clusters + 1):
    cells_k = np.where(z == k)[0]
    ms = (k - 1) * mpc; me = ms + mpc
    for c in cells_k:
        X[c, ms:me] = rng.poisson(8, size=mpc)
        n_bg = int(density * n_genes) - mpc
        if n_bg > 0:
            bg_idx = rng.choice([g for g in range(n_genes) if not (ms <= g < me)], size=n_bg, replace=False)
            X[c, bg_idx] = rng.poisson(1, size=n_bg)
    for c in cells_k:
        others = [kk for kk in range(1, n_clusters + 1) if kk != k]
        kk = rng.choice(others)
        oms = (kk - 1) * mpc; ome = oms + mpc
        ci = rng.choice(np.arange(oms, ome), size=max(1, mpc // 3), replace=False)
        X[c, ci] += rng.poisson(2, size=len(ci))
X_sparse = csr_matrix(X)
z_int = np.ascontiguousarray(z, dtype=np.int32)

# Precompile
try:
    from decontx.fast_ops import precompile
    precompile()
except ImportError:
    pass  # jjia1 has no precompile() function

# Warm up with balanced subset (need >=2 clusters to avoid div-by-zero in eta)
# Take first 10 cells from each cluster
warmup_idx = []
for k in range(1, n_clusters + 1):
    warmup_idx.extend(np.where(z == k)[0][:10])
warmup_idx = np.array(warmup_idx[:40])
X_warm = X_sparse[warmup_idx].tocsr()
z_warm = np.ascontiguousarray(z[warmup_idx], dtype=np.int32)
warmup = DecontXModel(max_iter=5, convergence=0.1, seed=999, verbose=False)
_ = warmup.fit_transform(X_warm, z_warm)

# Real run
model = DecontXModel(max_iter={max_iter}, convergence=1e-6, seed=12345, verbose=False)
t0 = time.perf_counter()
res = model.fit_transform(X_sparse, z_int)
elapsed = time.perf_counter() - t0

out = {{
    'elapsed': elapsed,
    'theta_mean': float(res['theta'].mean()),
    'contam_mean': float(res['contamination'].mean()),
    'n_iter': res.get('n_iter', len(res.get('log_likelihood_history', res.get('log_likelihood', [])))),
    'theta_first5': res['theta'][:5].tolist(),
    'contam_first5': res['contamination'][:5].tolist(),
}}
print(json.dumps(out))
"""
    out = run_in_subprocess(pkg_path, script)
    return json.loads(out.strip().split("\n")[-1])


def main():
    # Both paths were hardcoded to one machine. Default to this checkout and let
    # the reference implementation be pointed at via the environment.
    mine_path = os.environ.get(
        "DECONTX_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    jjia1_path = os.environ.get("DECONTX_JJIA1_PATH")
    if not jjia1_path or not os.path.isdir(jjia1_path):
        raise SystemExit(
            "Set DECONTX_JJIA1_PATH to a checkout of jjia1/decontx-python "
            "(the directory containing the 'decontx' package).\n"
            "  DECONTX_JJIA1_PATH=/path/to/decontx-python python tests/benchmark_compare.py"
        )

    configs = [
        (500, 500, 4, 0.15, 100),
        (2000, 2000, 5, 0.10, 100),
        (5000, 3000, 8, 0.08, 50),
        (10000, 5000, 10, 0.05, 30),
    ]

    print(
        f"{'Cells':>8} {'Genes':>8} {'Dens':>6} {'Mine(s)':>10} {'jjia1(s)':>10} "
        f"{'Speedup':>8} {'theta_match':>12}"
    )
    print("-" * 75)

    for n_cells, n_genes, n_clusters, density, max_iter in configs:
        print(f"\n--- {n_cells}x{n_genes}, density={density}, max_iter={max_iter} ---")
        print("  Running mine...", flush=True)
        r_mine = benchmark_subprocess(
            mine_path, n_cells, n_genes, n_clusters, density, max_iter, "mine"
        )
        print(f"  mine: {r_mine['elapsed']:.3f}s, {r_mine['n_iter']} iters", flush=True)

        print("  Running jjia1...", flush=True)
        r_jjia1 = benchmark_subprocess(
            jjia1_path, n_cells, n_genes, n_clusters, density, max_iter, "jjia1"
        )
        print(
            f"  jjia1: {r_jjia1['elapsed']:.3f}s, {r_jjia1['n_iter']} iters", flush=True
        )

        speedup = r_jjia1["elapsed"] / r_mine["elapsed"] if r_mine["elapsed"] > 0 else 0
        # Compare theta
        t_mine = np.array(r_mine["theta_first5"])
        t_jjia1 = np.array(r_jjia1["theta_first5"])
        theta_diff = np.max(np.abs(t_mine - t_jjia1))

        print(
            f"  {'Cells':>8} {'Genes':>8} {'Dens':>6} {'Mine(s)':>10} {'jjia1(s)':>10} "
            f"{'Mine/jjia1':>10} {'theta_diff':>12}"
        )
        print(
            f"  {n_cells:>8} {n_genes:>8} {density:>5.0%} {r_mine['elapsed']:>10.3f} "
            f"{r_jjia1['elapsed']:>10.3f} {speedup:>9.2f}x {theta_diff:>12.2e}"
        )


if __name__ == "__main__":
    main()
