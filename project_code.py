"""HPC Optimization: Pairwise Distance Matrix
Baseline (nested loops) vs Optimized (NumPy vectorization with Gram matrix)
"""
import numpy as np
import time
import math
from typing import List, Dict, Tuple

def baseline_pairwise_distances_loops(X: np.ndarray) -> np.ndarray:
    N, D = X.shape
    dist = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        xi = X[i]
        for j in range(N):
            s = 0.0
            xj = X[j]
            for k in range(D):
                diff = xi[k] - xj[k]
                s += diff * diff
            dist[i, j] = math.sqrt(s)
    return dist

def optimized_pairwise_distances_vectorized(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    norms = np.sum(X * X, axis=1)
    G = X @ X.T
    d2 = norms[:, None] + norms[None, :] - 2.0 * G
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, dtype=np.float64)

def run_benchmark(sizes: List[int], dim: int = 32, seed: int = 42, loop_cap_seconds: float = 40.0) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    rng = np.random.default_rng(seed)
    records = []
    last_baseline = None
    last_optimized = None

    # calibrate baseline per-pair cost
    calib_N = 60
    X_calib = rng.random((calib_N, dim), dtype=np.float64)
    t0 = time.perf_counter()
    _ = baseline_pairwise_distances_loops(X_calib)
    t1 = time.perf_counter()
    baseline_per_pair = (t1 - t0) / (calib_N * calib_N)

    for N in sizes:
        X = rng.random((N, dim), dtype=np.float64)

        # optimized
        t0 = time.perf_counter()
        D_opt = optimized_pairwise_distances_vectorized(X)
        t1 = time.perf_counter()
        opt_time = t1 - t0

        predicted = baseline_per_pair * (N * N)
        do_baseline = predicted <= loop_cap_seconds

        if do_baseline:
            t0 = time.perf_counter()
            D_base = baseline_pairwise_distances_loops(X)
            t1 = time.perf_counter()
            base_time = t1 - t0
            max_abs_err = float(np.max(np.abs(D_base - D_opt)))
            last_baseline = D_base
        else:
            base_time = float('nan')
            max_abs_err = float('nan')
            D_base = None

        last_optimized = D_opt

        records.append({
            "N": N,
            "Dim": dim,
            "Baseline_time_s": base_time,
            "Optimized_time_s": opt_time,
            "Speedup_x": (base_time / opt_time) if (not np.isnan(base_time) and opt_time > 0) else float('nan'),
            "Predicted_baseline_time_s": predicted,
            "Max_abs_error": max_abs_err
        })

    return (last_baseline if last_baseline is not None else np.array([])), last_optimized, records

if __name__ == "__main__":
    sizes = [200, 400, 600, 800, 1000]
    baseline_last, optimized_last, records = run_benchmark(sizes, dim=32, seed=7, loop_cap_seconds=35.0)
    import pandas as pd
    df = pd.DataFrame(records)
    print(df.to_string(index=False))
