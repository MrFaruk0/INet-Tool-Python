# Test Plan — INet-Tool Python Migration

## Test Philosophy

Every test must verify that Python output matches R output exactly for a given input. Tests are organized in tiers: unit tests (individual functions), integration tests (function chains), numerical equivalence tests (R vs Python), graph equivalence tests, edge-case tests, and reproducibility tests.

All test inputs should be deterministic (fixed seeds, synthetic data, or exported R data).

---

## Tier 0: Data Export from R

Before any Python tests can run, export reference data from R:

### Test 0.1: Export `adjL_data`
```r
library(INetTool)
data("adjL_data")
saveRDS(adjL_data, "test_data/adjL_data.rds")
write.csv(adjL_data[[1]], "test_data/adjL_data_1.csv")
write.csv(adjL_data[[2]], "test_data/adjL_data_2.csv")
```

### Test 0.2: Export `graphL_data`
```r
data("graphL_data")
saveRDS(graphL_data, "test_data/graphL_data.rds")
```

### Test 0.3: Export `tryL_data`
```r
data("tryL_data")
saveRDS(tryL_data, "test_data/tryL_data.rds")
```

### Test 0.4: Generate and export synthetic test data
```r
set.seed(42)
# Simple 5×5 adjacency matrices with distinct weights
adj_A <- matrix(c(0, 0.3, 0.7, 0, 0,
                  0.3, 0, 0.5, 0.9, 0,
                  0.7, 0.5, 0, 0.2, 0,
                  0, 0.9, 0.2, 0, 0.4,
                  0, 0, 0, 0.4, 0), nrow=5, byrow=TRUE)
adj_B <- matrix(c(0, 0.4, 0.6, 0, 0,
                  0.4, 0, 0.8, 0.3, 0,
                  0.6, 0.8, 0, 0.1, 0.1,
                  0, 0.3, 0.1, 0, 0.5,
                  0, 0, 0.1, 0.5, 0), nrow=5, byrow=TRUE)
adj_C <- matrix(c(0, 0.5, 0.3, 0.7, 0,
                  0.5, 0, 0.4, 0.2, 0,
                  0.3, 0.4, 0, 0.6, 0,
                  0.7, 0.2, 0.6, 0, 0.3,
                  0, 0, 0, 0.3, 0), nrow=5, byrow=TRUE)
rownames(adj_A) <- colnames(adj_A) <- c("A","B","C","D","E")
rownames(adj_B) <- colnames(adj_B) <- c("A","B","C","D","E")
rownames(adj_C) <- colnames(adj_C) <- c("A","B","C","D","E")
adjL_3x5 <- list(A=adj_A, B=adj_B, C=adj_C)
saveRDS(adjL_3x5, "test_data/adjL_3x5.rds")

# Run consensusNet on this data and save all outputs
result_3x5 <- consensusNet(adjL_3x5, threshold=0.5, tolerance=0.1, theta=0.04, ncores=1)
saveRDS(result_3x5, "test_data/result_3x5.rds")
# Also save individual components
write.csv(as.matrix(as_adjacency_matrix(result_3x5$graphConsensus, attr="weight")),
          "test_data/consensus_3x5.csv")
write.csv(result_3x5$Comparison, "test_data/comparison_3x5.csv")

# 2 identical graphs
adj_ident <- adj_A
adjL_identical <- list(adj_A, adj_A)
result_identical <- consensusNet(adjL_identical, threshold=0.5, tolerance=0.1, ncores=1)
saveRDS(result_identical, "test_data/result_identical.rds")

# 2 completely disjoint graphs (no shared edges)
adj_disjoint_1 <- matrix(c(0,0.8,0,0,0,0.8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), nrow=5)
adj_disjoint_2 <- matrix(c(0,0,0,0,0,0,0,0,0,0,0,0,0,0.9,0,0,0,0.9,0,0,0,0,0,0,0), nrow=5)
rownames(adj_disjoint_1) <- colnames(adj_disjoint_1) <- c("A","B","C","D","E")
rownames(adj_disjoint_2) <- colnames(adj_disjoint_2) <- c("A","B","C","D","E")
adjL_disjoint <- list(adj_disjoint_1, adj_disjoint_2)
result_disjoint <- consensusNet(adjL_disjoint, nitermax=20, ncores=1)
saveRDS(result_disjoint, "test_data/result_disjoint.rds")

# 1 graph only (degenerate case)
adjL_single <- list(adj_A)
result_single <- consensusNet(adjL_single, ncores=1)
saveRDS(result_single, "test_data/result_single.rds")

# 3 identical graphs
adjL_3ident <- list(adj_A, adj_A, adj_A)
result_3ident <- consensusNet(adjL_3ident, ncores=1)
saveRDS(result_3ident, "test_data/result_3ident.rds")

# JWmatrix and JWmean outputs
jw_mat <- JWmatrix(graphL_from(adjL_3x5))
jw_mean_val <- JWmean(graphL_from(adjL_3x5))
saveRDS(list(jwmat=jw_mat, jwmean=jw_mean_val), "test_data/jaccard_outputs.rds")
```

---

## Tier 1: Unit Tests — Internal Functions

### Test 1.1: `get_lower_tri_noDiag`
**File:** `tests/test_internal.py`

| Test ID | Input | Expected Behavior | R Verification |
|---------|-------|-------------------|----------------|
| T1.1.1 | 3×3 matrix with values 1-9 (row-major) | Upper triangle (row<col) and diag are NaN. Lower triangle (row>col) values: 4,7,8 preserved | `get_lower_tri_noDiag(matrix(1:9,3))` |
| T1.1.2 | 1×1 matrix [5] | Result is [NaN] (diag=NaN) | Verify diag only |
| T1.1.3 | 2×2 symmetric matrix [[0,0.5],[0.5,0]] | diag=NaN, lower=[0.5], upper=NaN | Check symmetry handling |
| T1.1.4 | All zeros 4×4 | All NaN except lower triangle of zeros | Values in lower tri should be 0.0 |

### Test 1.2: `code` (Cantor pairing)
**File:** `tests/test_internal.py`

| Test ID | Input | Expected Behavior | R Verification |
|---------|-------|-------------------|----------------|
| T1.2.1 | code(0,1) | Result: (0+1)*(0+1+1)//2 + 1 = 1*2//2 + 1 = 2 | Direct formula |
| T1.2.2 | code(1,0) | Same as code(0,1) = 2 (order invariant) | — |
| T1.2.3 | code(0,0) | (0+0)*(0+0+1)//2 + 0 = 0 | Self-loop key (unused but should work) |
| T1.2.4 | code(2,5) | (2+5)*(2+5+1)//2 + 5 = 7*8//2 + 5 = 28 + 5 = 33 | Verify |
| T1.2.5 | code(5,2) | Same as code(2,5) = 33 | Order invariance |
| T1.2.6 | Uniqueness: check all pairs for 0..99 | No collisions in 5050 pairs | Cantor pairing guarantee |
| T1.2.7 | code(0,0) through code(0,99) | Strictly increasing sequence | Verify monotonic |

### Test 1.3: `weight_mat`
**File:** `tests/test_internal.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T1.3.1 | New edge: graph has no edge (0,1); call weight_mat(0.7, g, (0,1)) | Edge (0,1) added with weight=0.7 |
| T1.3.2 | Existing edge: edge (0,1) has weight 0.3; call weight_mat(0.7, g, (0,1)) | Edge weight updated from 0.3 to 0.7, no duplicate |
| T1.3.3 | Self-loop: weight_mat(0.5, g, (0,0)) | Edge added with weight=0.5 |
| T1.3.4 | Multiple edges: add (0,1), (1,2), (0,3) | All three edges exist with correct weights |

---

## Tier 2: Unit Tests — Distance Functions

### Test 2.1: `jaccard_all`
**File:** `tests/test_distance.py`

| Test ID | Input | Expected Behavior | R Verification |
|---------|-------|-------------------|----------------|
| T2.1.1 | 2 identical 5-node graphs with edges (0-1:0.5, 1-2:0.7) | Jaccard distance = 0.0 (identical) | `JaccardAll(list(g,g))` |
| T2.1.2 | 2 disjoint graphs (no shared edge positions with weight) | Jaccard distance = 1.0 (completely different) | Verify |
| T2.1.3 | 2 partially overlapping graphs | Distance in (0,1) | Compare with R output |
| T2.1.4 | 3 graphs (validade all pairwise) | Mean distance scalar | Compare with R `JWmean` output |
| T2.1.5 | Graphs with different vertex names | Raises ValueError / error message | R: `stop("Check: Not same nodes...")` |
| T2.1.6 | Empty graphs (no edges, N vertices) | Distance = NaN → converted to 0. Check: all-zeros produce sim.jac=0/0=NaN → 0, so distance=1? Actually 1-sim where sim is set 0 for NaN. dist=1. But if ALL pairs are 0/0, all distances are 1. Mean = 1. | Verify edge-case behavior exactly |

**CRITICAL NOTE for T2.1.6:** The algorithm sets `sim.jac[is.na(sim.jac)] = 0` BEFORE converting to distance. So NaN similarity → 0 similarity → distance = 1. If all graph pairs have NO non-zero edges in common, all pairwise distances = 1.0, mean = 1.0.

### Test 2.2: `jw_matrix`
**File:** `tests/test_distance.py`

| Test ID | Input | Expected Behavior | R Verification |
|---------|-------|-------------------|----------------|
| T2.2.1 | 2 identical graphs | Distance matrix: [[0,0],[0,0]] | `JWmatrix(list(g,g))` |
| T2.2.2 | 2 different graphs | Symmetric 2×2, diag=0, off-diag between 0 and 1 | Compare |
| T2.2.3 | 3 graphs | Symmetric 3×3, diag=0 | Compare with R |
| T2.2.4 | Named graphs | Row/col names preserved | Check names propagate |

### Test 2.3: `jw_mean`
**File:** `tests/test_distance.py`

| Test ID | Input | Expected Behavior | R Verification |
|---------|-------|-------------------|----------------|
| T2.3.1 | 2 identical graphs | Mean = 0.0 | `JWmean(list(g,g))` |
| T2.3.2 | adjL_3x5 input | Match R output exactly | Compare float |
| T2.3.3 | Named check verification | Error if names differ | Stop message match |

---

## Tier 3: Unit Tests — Pre-Processing

### Test 3.1: `adj_rename`
**File:** `tests/test_preprocess.py`

| Test ID | Input | Expected Behavior | R Verification |
|---------|-------|-------------------|----------------|
| T3.1.1 | 2 matrices with SAME names, size 2×2 | Output: same matrices, same names | Identity transformation |
| T3.1.2 | 2 matrices with DIFFERENT names: mat1 has {A,B}, mat2 has {B,C} | Output: 2 matrices of size 3×3 with names {A,B,C}. mat1 has row/col C=0. mat2 has row/col A=0. | `adj_rename(tryL_data)` |
| T3.1.3 | 3 matrices with completely disjoint names {A,B}, {C,D}, {E,F} | Output: 3 matrices of size 6×6, each with zeros for nodes not in original | Verify |
| T3.1.4 | Matrices with duplicate names within a matrix | R uses `which(...)` — first match only. Duplicate names → only first occurrence populated. | Document as edge-case limitation |

### Test 3.2: `construction_graph`
**File:** `tests/test_preprocess.py`

| Test ID | Input | Expected Behavior | R Verification |
|---------|-------|-------------------|----------------|
| T3.2.1 | 10×5 data matrix (10 samples, 5 nodes) | Output: 5×5 correlation matrix, thresholded, graph built | Compare correlation values, threshold, edge count |
| T3.2.2 | perc=0.99 (top 1%) | Sparser graph than default | Verify edge count |
| T3.2.3 | perc=0.0 (keep all) | Full graph (complete, except zero correlations) | Verify |
| T3.2.4 | Identical columns in data | Correlation = 1.0 between those columns | Verify |
| T3.2.5 | Check Louvain modularity | Modularity between -0.5 and 1.0 | Range check |

---

## Tier 4: Unit Tests — Core Algorithm

### Test 4.1: `consensus_net` — Basic Operation
**File:** `tests/test_consensus.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T4.1.1 | adjL_3x5 (3 matrices, 5×5) | Correct output shape: list of graphConsensus (igraph), Comparison (array), similarGraphs (list of 3 graphs) |
| T4.1.2 | adjL_3x5 — Match consensus adjacency matrix | Element-wise comparison with R's output. Tolerance: 1e-12 for weights. Exact match for zero/nonzero pattern. |
| T4.1.3 | adjL_3x5 — Match Comparison vector | Exact distances per iteration, tolerance 1e-12 |
| T4.1.4 | Default parameters | threshold=0.5, tolerance=0.1, theta=0.04, nitermax=50 |

### Test 4.2: `consensus_net` — Parameter Variation
**File:** `tests/test_consensus.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T4.2.1 | threshold=0.2 | Sparser consensus (more edges kept) |
| T4.2.2 | threshold=0.8 | Denser consensus (fewer edges kept) |
| T4.2.3 | tolerance=0.01 | More iterations, potentially stricter convergence |
| T4.2.4 | tolerance=1000.0 | Immediate convergence (distance < tol on first check) |
| T4.2.5 | theta=0.0 | Neighborhood component disabled. Weight = (w_own + mean(w_others))/2 only |
| T4.2.6 | theta=1.0 | Strong neighborhood influence |
| T4.2.7 | nitermax=2 | Stopped at iteration 2 if not converged earlier |
| T4.2.8 | verbose=False | No console output |

### Test 4.3: `consensus_net` — Edge Cases
**File:** `tests/test_consensus.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T4.3.1 | 1 graph only (K=1) | Algorithm should handle gracefully. Jaccard of 1 graph vs itself? Actually `combn(1:1, 2)` returns empty matrix → no pairwise comparison. Check R behavior first. |
| T4.3.2 | 2 identical graphs | Distance stays 0 → immediate convergence (Comp < tolerance), returns element-wise mean = original adjacency |
| T4.3.3 | 3 identical graphs | Same as T4.3.2 |
| T4.3.4 | All-zero adjacency matrices | All graphs empty. Distance? 0/0 for all pairs → NaN → 0 similarity → distance = 1. But initial Comp = 1 > tolerance, iterates. UnionGraph is empty (0 edges). Edgelist is 0×2 matrix. Inner loop: dim(Edgelist)[1] = 0 → for loop over z does nothing. Graphs unchanged. CompPost also 1. Comp > CompPost? 1 > 1 = FALSE → "Distance doesn't decrease" → break. Consensus: element-wise mean of all zeros = 0 everywhere → threshold → all 0. Consensus graph has N vertices, 0 edges. |
| T4.3.5 | Graphs with isolated vertices | Isolated vertices persist through algorithm (degree 0 → no edges connected → no weight updates for those vertices). Graph structure preserved for isolated nodes. |
| T4.3.6 | Very large weights (0.95+) | Weight clamping at 1.0 activated. Check no weight exceeds 1.0. |
| T4.3.7 | Weights exactly 0.0 → denominator wUso==0 check | Denominator `len(Inei)+len(Jnei)` (no -2 subtraction) |
| T4.3.8 | Weights exactly 1.0 | Normal operation, no overflow |
| T4.3.9 | Self-loop adjacency entries preserved through conversion | `mode="upper"` and `diag=FALSE` — self-loops and lower triangle ignored |

---

## Tier 5: Unit Tests — Post-Processing

### Test 5.1: `threshold_net`
**File:** `tests/test_postprocess.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T5.1.1 | similarGraphs from consensus_net output, threshold=0.3 | Different consensus from default 0.5 |
| T5.1.2 | threshold=0.0 | All non-zero mean weights preserved |
| T5.1.3 | threshold=1.0 | All edges zeroed (empty graph, N vertices) |
| T5.1.4 | Same similarGraphs, same threshold | Reproducible output |
| T5.1.5 | Match R output for adjL_3x5 run | Exact weight match |

### Test 5.2: `density_net`
**File:** `tests/test_postprocess.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T5.2.1 | similarGraphs from consensus_net | Quantiles computed, ranges valid |
| T5.2.2 | Quantile values ascending | `quantile[0] ≈ min`, `quantile[-1] ≈ max` |
| T5.2.3 | quantileNo0 | All values > 0 |
| T5.2.4 | Match R quantile output exactly | Tolerance 1e-12 |

### Test 5.3: `specific_net`
**File:** `tests/test_postprocess.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T5.3.1 | Original graphs + consensus | Each specific graph is subset of original edges |
| T5.3.2 | percentageOfSpecificity in [0,1] | Range check |
| T5.3.3 | Consensus = one of the original graphs | That layer's specific graph is empty |
| T5.3.4 | Consensus is empty | All original edges preserved as specific |

---

## Tier 6: Unit Tests — Measures

### Test 6.1: `measures_net`
**File:** `tests/test_measures.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T6.1.1 | Single graph, nodes_measures=True | All 9 graph measures + 4 node measures computed |
| T6.1.2 | Single graph, nodes_measures=False | Only graph measures, no node measures |
| T6.1.3 | 3 graphs | List of 3 result entries |
| T6.1.4 | Check measure value ranges: transitivity [0,1], density [0,1], assortativity [-1,1] | Valid ranges |
| T6.1.5 | Match R output for known graph | Within tolerance |

---

## Tier 7: Integration Tests — Full Workflows

### Test 7.1: End-to-End Pipeline
**File:** `tests/test_integration.py`

```python
def test_full_pipeline():
    """Simulate the vignette workflow end-to-end."""
    # 1. adj_rename (if needed)
    # 2. JWmean pre-check
    # 3. consensus_net
    # 4. threshold_net (re-threshold)
    # 5. density_net (inspect)
    # 6. specific_net
    # 7. All outputs match R
```

### Test 7.2: Vignette Workflow Equivalence
**File:** `tests/test_integration.py`

| Test ID | Steps | R Verification |
|---------|-------|----------------|
| T7.2.1 | adj_rename → consensusNet on tryL_data | Match output |
| T7.2.2 | constructionGraph → consensusNet on exampleL_data | Match output |
| T7.2.3 | consensusNet → thresholdNet → specificNet | Full chain match |

---

## Tier 8: Numerical Equivalence Tests

### Test 8.1: Floating-Point Precision
**File:** `tests/test_numerical.py`

| Test ID | Test | Tolerance |
|---------|------|-----------|
| T8.1.1 | Jaccard distance: Python vs R on 3×5 data | `1e-12` absolute |
| T8.1.2 | Consensus edge weights: Python vs R | `1e-12` absolute, `1e-10` relative |
| T8.1.3 | Comparison vector: exact distance values | `1e-12` |
| T8.1.4 | Number of iterations: Python vs R | Exact match (integer) |
| T8.1.5 | Edge existence (binary pattern): Python vs R | Exact match (no false positives/negatives) |
| T8.1.6 | Consensus vertex count: Python vs R | Exact match |
| T8.1.7 | Consensus edge count: Python vs R | Exact match |

### Test 8.2: Order Sensitivity
| Test ID | Test |
|---------|------|
| T8.2.1 | Reorder adjacency matrices in input list → same consensus |
| T8.2.2 | Reorder vertex names → same consensus structure |
| T8.2.3 | Different random seeds → consistent outputs (algorithm is deterministic) |

---

## Tier 9: Graph Equivalence Tests

### Test 9.1: Graph Structure Equivalence
**File:** `tests/test_graph_equivalence.py`

For two igraph objects (Python and R output), verify:
- Same vertex count
- Same edge count
- Same vertex names (if any)
- Same adjacency matrix (element-wise, tolerance 1e-12)
- Same edge list (as set of (u,v,weight) tuples)
- Same graph-level properties (density, modularity, etc.)

---

## Tier 10: Edge-Case Tests

### Test 10.1: Empty/Degenerate Inputs
**File:** `tests/test_edge_cases.py`

| Test ID | Input | Expected Behavior |
|---------|-------|-------------------|
| T10.1.1 | Empty list `[]` | Raise error (need at least 1 matrix) |
| T10.1.2 | 1 graph (K=1) | Consensus = input graph (element-wise mean of 1). Check no NaN in Jaccard. |
| T10.1.3 | 1×1 adjacency matrix (1 node) | Works, single-node graph |
| T10.1.4 | 2×2 adjacency matrix, all zeros | Works, empty graphs |
| T10.1.5 | Non-square matrix | Raise error early |
| T10.1.6 | Non-symmetric adjacency matrix | igraph's `mode="upper"` silently uses only upper triangle. Warn user. |
| T10.1.7 | Negative weights in adjacency | Clamping at 0? Actually, the algorithm doesn't clamp negatives. It only clamps >1. Negative weights would propagate. R's igraph accepts them. Document behavior. |
| T10.1.8 | NaN weights in adjacency | R's igraph may crash. Guard in Python. |
| T10.1.9 | Inf weights in adjacency | Guard in Python. |
| T10.1.10 | Weights beyond [0,1] range | Clamping at 1.0 for updates. Input weights >1 get accepted. |
| T10.1.11 | Matrices with different dimensions | Raise error (need adj_rename first) |
| T10.1.12 | ncores=0 | Fallback to single-process or error |
| T10.1.13 | ncores > available CPUs | Works, uses ncores processes |

### Test 10.2: Convergence Edge Cases
| Test ID | Scenario | Expected Behavior |
|---------|----------|-------------------|
| T10.2.1 | Distance oscillates (increases then decreases) | Algorithm breaks on first non-decrease. Accepts previous state. |
| T10.2.2 | Distance increases monotonically | Breaks after first iteration (Comp < CompPost fails) |
| T10.2.3 | nitermax=0 | Loop runs count=0, check nitermax: `count > 0`? FALSE. If Comp < tolerance: break. If not: iterate. After iteration, `count=1 > 0` → break. So effectively 1 iteration max. |
| T10.2.4 | tolerance=0.0 | Requires exactly 0 distance to converge. May run to nitermax. |
| T10.2.5 | theta very large (1000) | Most weights clamped to 1.0. Graph becomes nearly complete. |

---

## Tier 11: Reproducibility Tests

### Test 11.1: Deterministic Output
**File:** `tests/test_reproducibility.py`

| Test ID | Test |
|---------|------|
| T11.1.1 | Run consensus_net twice with same input → identical outputs |
| T11.1.2 | Run with ncores=1 and ncores=2 → identical outputs |
| T11.1.3 | Run on different machines (same OS) → identical outputs |
| T11.1.4 | Run on Windows vs Linux → identical outputs (Python should be OS-independent) |
| T11.1.5 | Save/load intermediate graphs → continuation produces same result |

---

## Tier 12: Performance Regression Tests

### Test 12.1: Benchmarking
**File:** `tests/test_performance.py`

| Test ID | Test |
|---------|------|
| T12.1.1 | 100-node, 3-graph consensus: < 60 seconds |
| T12.1.2 | 500-node, 3-graph consensus: < 5 minutes |
| T12.1.3 | Memory usage: < 2 GB for 500-node case |
| T12.1.4 | Parallel speedup: ncores=4 > ncores=1 by at least 1.5x |
| T12.1.5 | No memory leak: repeated runs don't increase memory |

---

## Test File Structure

```
tests/
├── conftest.py                    # Shared fixtures, data loading
├── test_internal.py               # T1.x tests
├── test_distance.py               # T2.x tests
├── test_preprocess.py             # T3.x tests
├── test_consensus.py              # T4.x tests
├── test_postprocess.py            # T5.x tests
├── test_measures.py               # T6.x tests
├── test_integration.py            # T7.x tests
├── test_numerical.py              # T8.x tests
├── test_graph_equivalence.py      # T9.x tests
├── test_edge_cases.py             # T10.x tests
├── test_reproducibility.py        # T11.x tests
├── test_performance.py            # T12.x tests
└── test_data/                     # Exported R reference data
    ├── adjL_data.rds
    ├── adjL_data_1.csv
    ├── adjL_data_2.csv
    ├── graphL_data.rds
    ├── tryL_data.rds
    ├── adjL_3x5.rds
    ├── result_3x5.rds
    ├── consensus_3x5.csv
    ├── comparison_3x5.csv
    ├── result_identical.rds
    ├── result_disjoint.rds
    ├── result_single.rds
    ├── result_3ident.rds
    └── jaccard_outputs.rds
```

---

## Shared Fixtures (conftest.py)

```python
import pytest
import numpy as np
import igraph

@pytest.fixture
def adj_matrices_2():
    """Two 5x5 adjacency matrices with controlled weights."""
    adj_A = np.array([
        [0.0, 0.3, 0.7, 0.0, 0.0],
        [0.3, 0.0, 0.5, 0.9, 0.0],
        [0.7, 0.5, 0.0, 0.2, 0.0],
        [0.0, 0.9, 0.2, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.4, 0.0]
    ])
    adj_B = np.array([
        [0.0, 0.4, 0.6, 0.0, 0.0],
        [0.4, 0.0, 0.8, 0.3, 0.0],
        [0.6, 0.8, 0.0, 0.1, 0.1],
        [0.0, 0.3, 0.1, 0.0, 0.5],
        [0.0, 0.0, 0.1, 0.5, 0.0]
    ])
    return [adj_A, adj_B]

@pytest.fixture
def adj_matrices_3(adj_matrices_2):
    """Three 5x5 adjacency matrices."""
    adj_C = np.array([
        [0.0, 0.5, 0.3, 0.7, 0.0],
        [0.5, 0.0, 0.4, 0.2, 0.0],
        [0.3, 0.4, 0.0, 0.6, 0.0],
        [0.7, 0.2, 0.6, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.3, 0.0]
    ])
    return adj_matrices_2 + [adj_C]

@pytest.fixture
def graphs_3(adj_matrices_3):
    """3 igraph graphs from 3 adjacency matrices."""
    graphs = []
    for adj in adj_matrices_3:
        g = igraph.Graph.Weighted_Adjacency(adj.tolist(), mode='upper', attr='weight')
        g.vs["name"] = ["A", "B", "C", "D", "E"]
        graphs.append(g)
    return graphs

@pytest.fixture
def r_reference_3x5():
    """Load R reference data for 3x5 case."""
    import pickle  # or rpy2 for .rds loading
    # This would load the saved R outputs
    # For now, placeholder
    return None

@pytest.fixture
def random_seed():
    np.random.seed(42)
    return 42
```

---

## Test Execution Commands

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/test_internal.py tests/test_distance.py tests/test_preprocess.py -v

# Run only numerical equivalence tests (requires R data)
pytest tests/test_numerical.py -v

# Run with coverage
pytest tests/ --cov=inet_python --cov-report=html

# Run performance tests (marked as slow)
pytest tests/test_performance.py -v -m slow
```

---

## Pass/Fail Criteria for Migration Acceptance

1. **Tier 1-5 (Unit):** 100% pass. Every unit test must pass.
2. **Tier 7 (Integration):** 100% pass. Full workflow produces expected outputs.
3. **Tier 8 (Numerical):** All tests pass at stated tolerances. Any deviation beyond tolerance is a regression bug.
4. **Tier 9 (Graph Equivalence):** Exact match for graph structure. No extra/missing edges or vertices.
5. **Tier 10 (Edge Cases):** All specified edge cases handled without crash. Correct error messages for invalid inputs.
6. **Tier 11 (Reproducibility):** 100% deterministic output for same input.
7. **Tier 12 (Performance):** Within 2× of R runtime for single-core. Parallel speedup ≥ 1.5× for ncores=4.

**No Python code may be written until all criteria above are verified.**
