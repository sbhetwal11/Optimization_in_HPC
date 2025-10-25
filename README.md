# HPC Optimization Project – Vectorized Pairwise Distance

This project demonstrates how vectorization and memory locality optimization can improve computational performance in **High-Performance Computing (HPC)** tasks.  
It compares a slow, loop-based implementation of Euclidean distance with a fully vectorized NumPy version leveraging BLAS and SIMD.

---

## Project Overview

### Objective
- Compare **baseline (loop)** vs **optimized (vectorized)** approaches for computing the pairwise Euclidean distance matrix.
- Quantify speedup and validate correctness.
- Illustrate how vectorization and contiguous memory access lead to HPC-grade performance.

---

## How It Works

### 1. Baseline Function
`baseline_pairwise_distances_loops()`  
Computes distances using **three nested Python loops**, manually iterating over every pair of points and every feature.

```python
for i in range(N):
    xi = X[i]
    for j in range(N):
        s = 0.0
        xj = X[j]
        for k in range(D):
            diff = xi[k] - xj[k]
            s += diff * diff
        dist[i, j] = math.sqrt(s)
```

 This approach behaves like looping through lists — high Python overhead, no caching, and no parallelization.

---

### 2. Optimized Function
`optimized_pairwise_distances_vectorized()`  
Rewrites the math using linear algebra:

\[
||x - y||^2 = ||x||^2 + ||y||^2 - 2x·y
\]

```python
norms = np.sum(X * X, axis=1)
G = X @ X.T
d2 = norms[:, None] + norms[None, :] - 2.0 * G
np.maximum(d2, 0.0, out=d2)
return np.sqrt(d2)
```

 NumPy delegates this to highly optimized **C/BLAS routines**, using SIMD and cache-efficient memory access.  
Result: the same output, **hundreds of times faster**.

---

##  Benchmark Setup

### Parameters
| Parameter | Description |
|------------|-------------|
| N | Number of data points (200, 400, 600, 800, 1000) |
| D | Number of features (32) |
| Seed | Random seed for reproducibility |
| Cap | Skip baseline runs predicted to exceed ~35 seconds |

### Data Generation
The script creates synthetic data on the fly:
```python
X = rng.random((N, dim), dtype=np.float64)
```
This ensures reproducible, purely CPU-bound tests (no I/O bottlenecks).

---

## How To Run

1. Clone or copy this repository.
2. Ensure Python ≥ 3.8 and install dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. Run the benchmark:
   ```bash
   python project_code.py
   ```
4. Results are saved to:
   - `results_pairwise_distances.csv`


---

## Findings

### Key Observations
- Vectorized operations achieve **20×–400× speedup** depending on dataset size.
- Accuracy difference between both methods is negligible (`< 1e-12`).
- Baseline runtime grows **quadratically (O(N²))**, while vectorized version scales almost linearly.
- Performance improvements arise from:
  - Eliminating Python interpreter overhead.
  - Using contiguous memory and SIMD vector units.
  - Delegating computation to compiled BLAS libraries.



##  Lessons Learned

- Even when using NumPy, **explicit Python loops kill performance**.
- Leveraging **matrix algebra** unlocks full CPU vectorization and caching.
- Small algorithmic changes can translate to **massive real-world gains** in HPC workloads.

---

##  Output Files

| File | Description |
|------|-------------|
| `python_code.py` | Full Python source code (benchmark + functions) |
| `results_pairwise_distances.csv` | Timing results for each test size |

---

## License
MIT License © 2025  
Developed for the **Data Structure and Algorithm** course project.

