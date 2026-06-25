# Migration Risk Assessment — INet-Tool (R → Python)

## Risk Categories

- **CRITICAL**: Functionally wrong results, silent data corruption, or impossible to implement
- **HIGH**: Significant effort, subtle behavioral differences likely, or major performance impact
- **MEDIUM**: Moderate effort, testable equivalence achievable
- **LOW**: Straightforward porting, minimal risk

---

## 1. CRITICAL Risks

### 1.1 r2r Hashmap Semantics
- **Source:** `ConsensusINet.R:249-270`
- **Issue:** The `r2r` package provides a specific hashmap implementation. The algorithm uses:
  - `r2r::hashmap()` — creation
  - `r2r::insert(m, key, value)` — in-place insertion with side effects
  - `m[key][[1]]` — retrieval with list-of-list semantics
- **Risk:** r2r hashmaps are mutable and support "append-to-key" semantics (if the same key is inserted twice). The `apply(Edgelist, 1, funWeights)` call iterates over all edges and relies on unique keys. If the `code()` pairing produces duplicate keys (e.g., from multigraphs), behavior is undefined.
- **Migrating to Python dict:** Need to verify that `dict`'s semantics match. Python dicts overwrite on duplicate key, which is fine if keys are guaranteed unique. The `[[1]]` double-bracket access pattern must be replicated (Pythons `dict.get()`).
- **Mitigation:** Use Python's `dict` with explicit uniqueness verification. Add an assertion that all edge codes are unique.

### 1.2 igraph Graph Operations Semantics
- **Source:** Multiple files; critical in `ConsensusINet.R:168-173`
- **Issue:** `igraph::union(graph1, graph2)` for weighted graphs has specific semantics for how edge attributes are combined when the same edge exists in both graphs. In igraph's C implementation, for `union()` with weighted graphs, when the same edge exists in both graphs, the resulting edge weight is **the sum of the two weights** (or the edge attribute of the first graph depending on version). This is NOT documented in the function's help page.
- **Risk:** If Python's igraph (python-igraph) implements `union()` differently, the union graph's structure will differ, cascading into different edge lists and different weight updates.
- **Mitigation:** Write a unit test that constructs two simple graphs, calls `igraph::union`, and records the resulting edge weights. Replicate exact behavior in Python.
- **Python-igraph note:** `python-igraph` has `Graph.union()` method. Need to verify edge attribute merge semantics match.

### 1.3 Cantor Pairing Function Numerics
- **Source:** `ConsensusINet.R:218`: `code(a,b) = (a+b)*(a+b+1)/2 + b`
- **Issue:** This works for integer inputs. If vertex IDs are ever non-integer, the Cantor pairing would produce non-unique keys or floats. The edgelist is extracted with `names=FALSE` ensuring integer IDs, but the function doesn't explicitly cast to integer.
- **Risk:** In Python, integer overflow behavior differs from R. R uses double-precision numerics which can represent integers exactly up to 2^53. Python ints are arbitrary precision. This is NOT a correctness risk, but a performance consideration — very large edge counts could make Python dict keys very large integers.
- **Mitigation:** Use the same formula. Validate that all vertex IDs are positive integers.

### 1.4 Parallel Execution Semantics
- **Source:** `ConsensusINet.R:195-460`
- **Issue:** R's `parallel::clusterApply` uses process-based parallelism (fork or PSOCK). The workers receive copies of data via `clusterExport`. The r2r hashmaps are exported by value.
- **Risk:** In R, each worker builds its own hashmaps (the code redundantly builds hashmaps for ALL graphs in EVERY worker). This is data duplication but works because hashmaps are immutable copies after export. Python multiprocessing would need the same approach or use shared memory.
- **Mitigation:** Python's `multiprocessing.Pool.map` or `concurrent.futures.ProcessPoolExecutor` can replicate this pattern. Since data must be copied to workers anyway, the redundancy in hashmap building can be preserved or optimized (build once, share).
- **Alternative:** Use threading instead of multiprocessing for lower overhead if the Python igraph operations release the GIL.

---

## 2. HIGH Risks

### 2.1 Jaccard Distance Computation Precision
- **Source:** `JaccardAll` closure in `ConsensusINet.R:114-121`, `JWmatrix.R:47-51`, `JWmean.R:42-46`
- **Issue:** Weighted Jaccard: `sum(min(a_i, b_i)) / sum(max(a_i, b_i))`. When all weights are zero for a pair, this produces `0/0 = NaN`, which is then converted: `sim.jac[is.na(sim.jac)] = 0`.
- **Risk:** Floating-point comparisons. In R, `min(0,0)/max(0,0) = NaN`. Python's behavior is identical (`0/0 = NaN` in IEEE 754), but the test `is.na()` differs — in Python you'd use `np.isnan()`. The important thing is to set diagonal entries correctly:
  - `JWmatrix`: `diag(sim.jac) = 1` (self-distance = 0)
  - `JWmean`/`JaccardAll`: `diag(sim.jac) = NA` (self-pair excluded from mean)
- **Mitigation:** Explicitly handle `den == 0` case before division.

### 2.2 Union Graph Edge Attribute Behavior
- **Source:** `ConsensusINet.R:168-173`
- **Issue:** The algorithm builds a union graph, then extracts its edgelist. The union graph's edge attributes (weights) are not used — only the edge list (which edges exist). However, the *way* union combines edge attributes could affect which edges appear as multigraph entries vs. single edges.
- **Risk:** If `igraph::union` treats weighted edges differently (e.g., duplicates edges when weights differ), the edgelist count would be wrong.
- **Mitigation:** Test union behavior carefully. The algorithm only needs which edges exist in the union, not their weights. Simplify: build the edge set directly by taking the union of all edgelists, bypassing `igraph::union` entirely.

### 2.3 Vertex Name Handling
- **Source:** Multiple places checking `length(rownames(adjL[[1]]))>0`
- **Issue:** R igraph stores vertex names as the `name` vertex attribute. When converting from adjacency matrix with `add.colnames="NA"`, this stores matrix colnames as vertex names. When names exist, functions like `get.edge.ids` behave differently (accepting names vs. requiring integer IDs).
- **Risk:** The `code()` function relies on integer vertex IDs from `as_edgelist(names=FALSE)`, which returns the internal numeric vertex indices regardless of name attributes. But name-based operations (`setdiff(V(consensus)$name, V(graph)$name)`) in `plotINet` assume name attributes exist.
- **Mitigation:** Python-igraph handles vertex names via the `"name"` vertex attribute identically. The `names=FALSE` equivalent in python-igraph returns integer IDs. Verify both code paths.

### 2.4 Missing Edge Handling
- **Source:** `ConsensusINet.R:292-294`, `ConsensusINet.R:308-310`, and throughout
- **Issue:** The pattern `w <- hashmap[key][[1]]; if(is.null(w)) w <- 0` is used extensively. The `[[1]]` double-bracket access returns the first element of the retrieved list, or NULL if the key doesn't exist.
- **Risk:** Python dict raises `KeyError` for missing keys. Must use `dict.get(key, default)` or `try/except`. The `[[1]]` pattern means the hashmap values are wrapped in single-element lists — this is an r2r-ism. Python dicts don't do this.
- **Mitigation:** Use `dict.get(key, 0)` pattern. The single-element-list wrapper is not needed.

---

## 3. MEDIUM Risks

### 3.1 Code Duplication
- **Issues:**
  - `get_lower_tri_noDiag` defined twice (`InternalFunction.R:9`, `ConsensusINet.R:78`)
  - Matrix element-wise mean computed three times (`densityNet.R:26-51`, `ThresholdConsensus.R:26-51`, `ConsensusINet.R:536-558`)
  - Weighted Jaccard computation in three places (`JWmatrix.R`, `JWmean.R`, `ConsensusINet.R:87-130`)
- **Risk:** During migration, different copies might be translated slightly differently. Must consolidate into shared functions.
- **Mitigation:** Implement each algorithm exactly once in Python and call from all sites.

### 3.2 Modularity Computation Without Weights
- **Source:** `constructionGraph.R:47` and `NetworkMeasures.R:28`
- **Issue:** `igraph::modularity(graph, cluster_louvain(graph, weights=NULL)$membership)` explicitly ignores edge weights in community detection for modularity calculation. This is intentional (Louvain with `weights=NULL` uses unweighted graph).
- **Risk:** Python-igraph's `cluster_louvain` also accepts `weights=None` to ignore edge weights. Behavioral equivalence is likely but must be verified.
- **Mitigation:** Test and document.

### 3.3 Random Seed / Determinism
- **Issue:** `cluster_louvain` in `constructionGraph` and `measuresNet` involves randomness. No `set.seed()` call is made in the package code.
- **Risk:** Non-deterministic modularity values on each run. This is existing R behavior — not a bug, but makes testing harder.
- **Mitigation:** In Python translation, set a fixed random seed for tests. The Louvain algorithm's stochasticity is inherent to the method.

### 3.4 Verbose Output
- **Source:** `ConsensusINet.R:155,470,499,512` — `cat()` calls
- **Issue:** Verbose output goes to console via `cat()`. In Python, `print()` is equivalent.
- **Mitigation:** Use Python's `logging` module or a `verbose` flag controlling `print()` calls.

### 3.5 Closure / Scoping Differences
- **Source:** `ConsensusINet.R:212-230` — closures that mutate outer variables via `<<-`
- **Issue:** R's `<<-` assignment operator modifies variables in the parent environment. `node_id <<- node_id + 1` and `edge_id <<- edge_id + 1` inside closures `funNeig`, `funWeights`, `funegoWeights` increment counters in the enclosing scope.
- **Risk:** Python closures can use `nonlocal` for this, but `lapply` in R passes functions as arguments — the closures are called by `lapply` and `apply`. The Python equivalent would be loops or explicit counter objects.
- **Mitigation:** Use simple for-loops instead of `lapply`/`apply` patterns, or use Python's `nonlocal` keyword for closures within the inner function.

### 3.6 `stats::quantile` Behavior
- **Source:** `constructionGraph.R:41`, `densityNet.R:63`
- **Issue:** R's `quantile()` has 9 different type parameters (default `type=7`). The code uses the default, which computes quantiles via a linear interpolation formula specific to R.
- **Risk:** Python's `numpy.quantile` or `pandas.DataFrame.quantile` may use different interpolation methods. The difference is negligible for large datasets but could affect exact threshold values on small datasets.
- **Mitigation:** Use `numpy.quantile(x, q, method='linear')` which matches R's type=7 default (numpy 1.22+). For older numpy, document the minor difference.

---

## 4. LOW Risks

### 4.1 Plotting Functions
- `plotINet`, `plotL`, `plotC` use base R graphics (`plot()`, `igraph.plot()`) and `multinet` for multi-layer plots.
- **Risk:** Low — these are visualization functions and can be ported to `matplotlib` or `python-igraph`'s plotting. Slight visual differences are acceptable.
- **Mitigation:** Use `python-igraph`'s built-in plotting (`igraph.plot()`) for the closest match.

### 4.2 Data Loading
- `.rda` files are R-specific binary format. Cannot be directly loaded in Python.
- **Mitigation:** Either convert `.rda` files to a portable format (JSON, HDF5, pickle), or re-implement the example data generation from scratch. The example data has known structure (3 patient similarity layers with 200 nodes each).

### 4.3 `utils::combn`
- R's `combn(n, 2)` generates all 2-element combinations from 1..n as a 2×C(n,2) matrix.
- **Python equivalent:** `itertools.combinations(range(n), 2)` returns an iterator of tuples. Transposing behavior differs (R returns column-major matrix).

### 4.4 `igraph::cluster_louvain` naming
- In python-igraph, the equivalent is `Graph.community_multilevel()` for Louvain or `Graph.community_leiden()` for the Leiden algorithm (improved Louvain).
- The `cluster_louvain` name in R-igraph matches python-igraph's deprecated `community_multilevel()` — need to use the correct Python API name.

---

## 5. Migration Effort Summary

| Component | Complexity | Lines | Risk |
|-----------|------------|-------|------|
| `consensusNet` core algorithm | VERY HIGH | ~350 | CRITICAL |
| r2r → Python dict conversion | HIGH | ~50 | CRITICAL |
| igraph graph operations | HIGH | ~100 | HIGH |
| Jaccard distance functions | MEDIUM | ~80 | HIGH |
| `constructionGraph` | MEDIUM | ~40 | MEDIUM |
| `densityNet` / `thresholdNet` | LOW | ~40/each | MEDIUM |
| `adj_rename` | LOW | ~40 | LOW |
| `measuresNet` | MEDIUM | ~60 | MEDIUM |
| `specificNet` | LOW | ~30 | LOW |
| Plotting functions | MEDIUM | ~100 | LOW |
| Data conversion | LOW | — | LOW |

**Estimated total: ~1000 lines of Python** (excluding tests and documentation).

---

## 6. Recommended Conversion Strategy

### Phase 1: Core Data Structures
1. Implement graph representation (python-igraph or NetworkX or custom)
2. Implement Cantor pairing function
3. Implement `get_lower_tri_noDiag`
4. Implement weighted Jaccard distance
5. **Verify:** Produce identical Jaccard distances on test data

### Phase 2: Algorithm Core
1. Implement hashmap-based neighbor/weight storage (Python dicts)
2. Implement union graph construction
3. Implement the edge weight update formula
4. Implement the ego-weight computation
5. Implement the iterative convergence loop
6. **Verify:** Produce identical consensus graph on `adjL_data`

### Phase 3: Parallelization
1. Wrap inner loop in multiprocessing
2. Implement the convergence tracking (Comparison vector)
3. **Verify:** Identical results with single vs. multi-core

### Phase 4: Auxiliary Functions
1. `constructionGraph`, `densityNet`, `thresholdNet`
2. `measuresNet`, `specificNet`
3. `adj_rename`

### Phase 5: Visualization
1. `plotC`, `plotINet`, `plotL`

### Phase 6: Integration & Testing
1. End-to-end tests with known data
2. Edge case testing (empty graphs, single graph, identical graphs, disjoint graphs)
3. Performance benchmarking vs. R original
