# Migration Specification — INet-Tool R → Python

## Part 1: Package Equivalence

### 1.1 R Package → Python Replacement

| R Package | Python Replacement | Justification | Compatibility Concern |
|-----------|-------------------|---------------|----------------------|
| **igraph** | `python-igraph` (`import igraph`) | Same library (C core). Nearly identical API. | Yes — careful. Some function names differ. `cluster_louvain` → `community_multilevel` (deprecated) or `community_leiden`. `transitivity` → `transitivity_undirected`. `centr_degree` → `Graph.knn()` usage differs. See §1.2. |
| **r2r** | `dict` (built-in) | Hashmap equivalent. Mutable, O(1) average lookup. | Yes — r2r's `insert(m,key,val)` is a side-effecting function; Python's `dict[key]=val` is an in-place operator. r2r supports duplicate-key append via `to_list()`; Python dict does not. Keys are guaranteed unique due to Cantor pairing — append-not-overwrite is unused. |
| **parallel** | `multiprocessing` or `concurrent.futures` | Process-based parallelism. | Yes — `clusterExport` ≈ `initargs`+ shared serialization. R's PSOCK clusters make deep copies; Python's `Pool.map` also pickles and unpickles. |
| **stats (cor)** | `numpy.corrcoef` or `scipy.stats.pearsonr` | Pearson correlation. | Yes — identical math. `numpy.corrcoef` returns full matrix; R's `cor` with a matrix input also returns full matrix. Column-wise correlation: R's `cor(X)` treats columns as variables; `numpy.corrcoef(X.T)` matches. |
| **stats (quantile)** | `numpy.quantile` | Quantile computation. | Minor — R's `quantile(x, probs, type=7)` uses a specific linear interpolation. `numpy.quantile(x, q, method='linear')` matches R type=7 behavior (available numpy ≥1.22). On older numpy, `np.percentile` uses different interpolation — set `interpolation='linear'`. |
| **ggplot2** | `matplotlib` + `seaborn` | Statistical plotting. | Low urgency — plotting functions are visual aids. Histograms: `matplotlib.pyplot.hist`. Density: `seaborn.kdeplot` or `pandas.DataFrame.plot.density`. |
| **ggpubr** | `matplotlib.pyplot.subplots` | Multi-panel plot arrangement. | Low urgency — `ggpubr::ggarrange` combines ggplot objects in a grid. `plt.subplots()` or `matplotlib.gridspec` achieve the same. |
| **multinet** | Drop / `matplotlib` manual layout | Multi-layer network visualization. | Low urgency — `multinet` is an R-specific package for multilayer network plotting. No direct Python equivalent for `ml_empty`/`add_igraph_layer_ml`. Replace with manual multi-panel layout via matplotlib, each panel showing one layer. |
| **robin** | `python-igraph` built-in + `cdlib` | Community detection wrapper. | Low urgency — `robin::membershipCommunities` wraps multiple igraph community detection methods. In python-igraph, use `Graph.community_walktrap().as_clustering().membership` etc. directly. `cdlib` provides additional algorithms. |
| **utils (combn)** | `itertools.combinations` | Pairwise index combinations. | Yes — `itertools.combinations(range(n), 2)` returns iterator of (i,j) tuples. R's `combn(n,2)` returns a 2×C(n,2) matrix. Python equivalent: `list(itertools.combinations(range(n), 2))` produces a list of 2-tuples. Column-major vs. row-major irrelevant since both are symmetric iteration. |
| **base R matrix** | `numpy.ndarray` | Matrix operations. | Yes — `rbind(A,vetI)` ≈ `np.vstack`. `which(is.na(x))` ≈ `np.where(np.isnan(x))`. `upper.tri(x)` ≈ `np.triu_indices`. `diag(x) <- NA` ≈ `np.fill_diagonal(x, np.nan)`. Element-wise `mean` over list of matrices ≈ `np.mean(stacked, axis=0)`. |

### 1.2 igraph Function Name Mapping

| R-igraph | python-igraph | Notes |
|----------|---------------|-------|
| `igraph::graph_from_adjacency_matrix(A, mode="upper", diag=FALSE, weighted=TRUE)` | `igraph.Graph.Weighted_Adjacency(A.tolist(), mode='upper', attr='weight')` | In python-igraph, `Weighted_Adjacency` takes a list-of-lists, not a numpy array. Use `.tolist()`. `mode='upper'` ignores lower triangle. |
| `igraph::graph_from_adjacency_matrix(A, mode="upper", diag=FALSE, weighted=TRUE, add.colnames="NA")` | Same as above, then set `g.vs["name"] = list_of_names` | R's `add.colnames="NA"` stores names. Python: manually assign after construction. |
| `igraph::as_adjacency_matrix(g, attr="weight")` | `g.get_adjacency(attribute='weight')` | Returns `igraph.Matrix`; convert via `np.array(list(...))` |
| `igraph::as_edgelist(g, names=FALSE)` | `g.get_edgelist()` | Returns list of (src, tgt) tuples using integer vertex indices. |
| `igraph::as_edgelist(g, names=TRUE)` | `[(g.vs[e.source]["name"], g.vs[e.target]["name"]) for e in g.es]` | Manual name extraction. |
| `igraph::E(g)$weight[edge_id]` | `g.es[edge_id]["weight"]` | 0-indexed in Python (`g.es[edge_id - 1]` if converting from R). |
| `igraph::V(g)$name` | `g.vs["name"]` | — |
| `igraph::get.edge.ids(g, c(u,v))` | `g.get_eid(u, v)` | Python raises ValueError if edge missing, R returns 0. |
| `igraph::add_edges(g, c(u,v))` | `g.add_edge(u, v)` | Adds single edge. |
| `igraph::union(g1, g2)` | `g1.union(g2)` | **CRITICAL**: Need to verify edge attribute merge semantics. |
| `igraph::as_adj_list(g)` | `g.get_adjlist()` | Returns list of lists of integer neighbors. |
| `igraph::ecount(g)` | `g.ecount()` | — |
| `igraph::vcount(g)` | `g.vcount()` | — |
| `igraph::degree(g)` | `g.degree()` | R returns named vector, Python returns list. |
| `igraph::transitivity(g, type="global")` | `g.transitivity_undirected()` | — |
| `igraph::transitivity(g, type="local")` | `g.transitivity_local_undirected()` | — |
| `igraph::diameter(g)` | `g.diameter()` | — |
| `igraph::modularity(g, membership)` | `g.modularity(membership)` | — |
| `igraph::edge_density(g)` | `g.density()` | — |
| `igraph::assortativity_degree(g)` | `g.assortativity_degree()` | — |
| `igraph::centr_degree(g)$centralization` | `g.maxdegree() - g.degree()` then apply formula. Better: compute manually or use `g.transitivity_avglocal_undirected()` approach. Actually: `centr_degree` returns `(centralization, theoretical_max, centrality_vector)`. Python: no direct equivalent — implement manually using formula. | HIGH EFFORT. |
| `igraph::centr_betw(g)$centralization` | Same issue — manually implement Freeman's centralization formula. | HIGH EFFORT. |
| `igraph::betweenness(g)` | `g.betweenness()` | R returns named vector. |
| `igraph::hub_score(g, weights=NULL)$vector` | `g.hub_score(weights=None)` | Returns `igraph.HubScore` named tuple with `.vector` attribute. |
| `igraph::cluster_louvain(g, weights=NULL)` | `g.community_multilevel(weights=None)` | Deprecated; use `community_leiden`. Need exact algorithm match. |
| `igraph::difference(g1, g2)` | `g1.difference(g2)` | — |
| `igraph::intersection(g1, g2)` | `g1.intersection(g2)` | — |
| `igraph::add_vertices(g, n)` | `g.add_vertices(n)` | — |
| `igraph::delete.vertices(g, vids)` | `g.delete_vertices(vids)` | — |
| `igraph::setdiff(a, b)` | `set(a) - set(b)` | Not igraph-specific; base R `setdiff`. |
| `igraph::intersect(a, b)` | `set(a) & set(b)` | Base R `intersect`. |

### 1.3 r2r → Python dict Specifics

**r2r pattern in code:**
```r
m <- r2r::hashmap()
r2r::insert(m, key, value)
result <- m[key][[1]]        # returns value or NULL if key missing
```

**Python equivalent:**
```python
m = {}
m[key] = value               # equivalent to insert
result = m.get(key)           # returns None if key missing (vs R's NULL)
```

**Critical difference:** R uses `[[1]]` because r2r wraps values in a list. Python dicts don't do this. The R code does:
```r
wUso <- Weights_list[[i]][code(nodes[1], nodes[2])][[1]]
if(is.null(wUso)) { wUso <- 0 }
```
Python equivalent:
```python
w_uso = weights_list[i].get(code(nodes[0], nodes[1]), 0)
```

All weight lookups MUST use `.get(key, 0)` since missing edges have weight 0.

---

## Part 2: Function-by-Function Migration Specification

### F1. `get_lower_tri_noDiag`

**Source:** `R/InternalFunction.R:9-12`, duplicated at `R/ConsensusINet.R:78-82`

**R signature:** `get_lower_tri_noDiag(cormat)`

**Input:** A square matrix (base R matrix). Dimensions N×N.

**Output:** The same matrix with upper triangle set to NA and diagonal set to NA. The lower triangle (below diagonal, exclusive) is preserved.

**Mathematical behavior:**
- `cormat[upper.tri(cormat)] <- NA` — sets all elements where `row < col` to NA
- `diag(cormat) <- NA` — sets all elements where `row == col` to NA
- Elements where `row > col` (strict lower triangle) are unchanged

**Side effects:** Modifies the input matrix in place (R semantics: `<-` on indexing modifies object), then returns it.

**R dependencies:** `upper.tri()` and `diag()` from base R.

**Python equivalent:**
```python
import numpy as np

def get_lower_tri_noDiag(cormat: np.ndarray) -> np.ndarray:
    """
    cormat: shape (N, N), float64
    Returns: shape (N, N), float64 — modified IN PLACE and returned
    """
    cormat[np.triu_indices_from(cormat, k=0)] = np.nan  # upper tri + diagonal
    return cormat
```
- `np.triu_indices_from(cormat, k=0)` returns indices for upper triangle including diagonal (k=0 means include diagonal). This matches `upper.tri(x) + diag(x)`.
- Note: R's `upper.tri` excludes the diagonal, then `diag <- NA` is separate. Combined: all elements row ≤ col become NA. The `triu_indices` with k=0 gives row ≤ col exactly.

**Import:** `numpy`

---

### F2. `adj_rename`

**Source:** `R/AdjSameName.R:15-59`

**R signature:** `adj_rename(adjL)`

**Input:** `adjL` — a list of square matrices (base R matrices). Each matrix may have different row/column names.

**Output:** A list of matrices, each of size N×N where N is the number of unique node names across all inputs. Each output matrix has identical row/col names (the union of all input names). Zero-filled for node pairs missing in the original.

**Mathematical behavior:**

Step 1: Collect all unique names (lines 19-25):
```
geneName = concatenation_of_all_rownames_from_all_matrices
genes = unique(geneName)
```

Step 2: For each matrix z in adjL (lines 31-56):
- Create N×N zero matrix with row/col names = `genes`
- For each (i,j) in original matrix's dimensions:
  - Copy weight from position (rownames[i], colnames[j]) of original matrix
  - To position (match_of_rownames[i], match_of_colnames[j]) in new matrix
  - Lookup uses `which(rownames(Mat[[z]])==target_name)` — exact string match, first occurrence wins

**Side effects:** None.

**R dependencies:** Base R only (`rownames`, `colnames`, `dim`, `unique`, `which`, `matrix`, `vector`).

**Python equivalent:**
```python
import numpy as np

def adj_rename(adjL: list[np.ndarray]) -> list[np.ndarray]:
    """
    adjL: list of np.ndarray, each shape (n_k, n_k), with named rows/cols
          (stored separately as lists of strings)
    Returns: list of np.ndarray, each shape (N, N)
    Also returns list of gene names for tracking.
    """
    # Collect all unique names
    all_names = []
    for mat in adjL:
        # Assume row/col names are stored alongside or as dict keys
        all_names.extend(row_names_of(mat))
    genes = sorted(set(all_names))  # unique, stable order
    N = len(genes)
    
    result = []
    for mat in adjL:
        new_mat = np.zeros((N, N), dtype=np.float64)
        # Build mapping: name -> index
        mat_names = row_names_of(mat)  # list of str, length = mat.shape[0]
        name_to_idx = {name: idx for idx, name in enumerate(mat_names)}
        for i_name in mat_names:
            for j_name in mat_names:
                old_i = name_to_idx[i_name]
                old_j = name_to_idx[j_name]
                new_i = genes.index(i_name)  # or pre-built mapping
                new_j = genes.index(j_name)
                new_mat[new_i, new_j] = mat[old_i, old_j]
        result.append(new_mat)
    return result, genes
```

**Design note:** R matrices store both numeric data AND row/col names. Python numpy arrays don't. The migration must either:
- Store names separately (list of strings alongside each ndarray)
- Use `pandas.DataFrame` which has `.index` and `.columns`
- Recommendation: use `pandas.DataFrame` for adjacency matrices throughout — it preserves R semantics exactly.

**Import:** `numpy`, `pandas` (recommended for name handling)

---

### F3. `JaccardAll` (closure inside `consensusNet`)

**Source:** `R/ConsensusINet.R:87-130`

**R signature:** `JaccardAll(grafi)`

**Input:** `grafi` — a list of igraph objects, all with identical vertex names.

**Output:** A single float: the mean weighted Jaccard distance across all pairs of graphs.

**Mathematical behavior:**

1. **Validation** (lines 91-97): For every pair of graphs (a,b), checks that all vertex names match. If any pair has non-identical vertex names, `stop()` with error.

2. **Flatten weights** (lines 99-107):
   - For each graph: `AdjW = as_adjacency_matrix(g, attr="weight")` — N×N sparse matrix
   - `triA = get_lower_tri_noDiag(AdjW)` — NA upper triangle and diag
   - `vettriA = as.vector(triA)` — column-major flatten (R's default)
   - `vetI = vettriA[!is.na(vettriA)]` — remove NA entries, keep only lower triangle values
   - `A = rbind(A, vetI)` — stack as rows of matrix A
   - **Result:** A is a K×M matrix where K = number of graphs, M = N(N-1)/2 (edges in a complete undirected graph without self-loops)

3. **Pairwise weighted Jaccard** (lines 111-121):
   For each pair (a,b) of graph rows:
   ```
   num = sum over all columns x of min(A[a, x], A[b, x])
   den = sum over all columns x of max(A[a, x], A[b, x])
   sim.jac[a,b] = num / den
   sim.jac[b,a] = num / den
   ```

4. **Cleanup** (lines 122-123):
   - `sim.jac[which(is.na(sim.jac))] = 0` — replace NaN (from 0/0) with 0 similarity
   - `diag(sim.jac) = NA` — exclude self-pairs

5. **Distance and mean** (lines 126-128):
   - `dist.jac = 1 - sim.jac` — distance = 1 - similarity
   - `mean(dist.jac, na.rm=TRUE)` — average over all non-NA entries (all non-diagonal)

**Side effects:** None (pure computation).

**R dependencies:** `igraph` (graph access), `utils::combn`, `get_lower_tri_noDiag` (internal).

**Python equivalent:**
```python
import numpy as np
from itertools import combinations

def jaccard_all(graphs: list) -> float:
    """
    graphs: list of igraph.Graph objects with identical vertex names
    Returns: mean weighted Jaccard distance (float)
    """
    # Validate vertex names
    for (k, (g1, g2)) in enumerate(combinations(graphs, 2)):
        if g1.vs["name"] != g2.vs["name"]:
            raise ValueError("Not same nodes in all graphs")

    K = len(graphs)
    N = graphs[0].vcount()
    M = N * (N - 1) // 2  # number of lower-triangle elements (no diag)

    A = np.zeros((K, M), dtype=np.float64)
    for l, g in enumerate(graphs):
        adj = np.array(list(g.get_adjacency(attribute='weight')))
        # Get lower triangle values (excluding diagonal)
        lower_idx = np.tril_indices(N, k=-1)
        A[l, :] = adj[lower_idx]

    sim_jac = np.zeros((K, K), dtype=np.float64)
    for a, b in combinations(range(K), 2):
        row_a = A[a, :]
        row_b = A[b, :]
        mins = np.minimum(row_a, row_b)
        maxs = np.maximum(row_a, row_b)
        den = maxs.sum()
        if den == 0:
            sim_jac[a, b] = 0.0
            sim_jac[b, a] = 0.0
        else:
            sim_jac[a, b] = mins.sum() / den
            sim_jac[b, a] = sim_jac[a, b]

    np.fill_diagonal(sim_jac, np.nan)
    sim_jac = np.nan_to_num(sim_jac, nan=0.0)
    np.fill_diagonal(sim_jac, np.nan)
    dist_jac = 1.0 - sim_jac
    return np.nanmean(dist_jac)
```

**Key numerical detail:** `np.nan_to_num(sim_jac, nan=0.0)` handles the 0/0 → NaN → 0 conversion. Then diag is reset to NaN for exclusion from mean. This exactly mirrors R's order of operations at lines 122-123.

**Import:** `numpy`, `itertools.combinations`

---

### F4. `JWmatrix`

**Source:** `R/JaccardWeightedMatrix.R:15-67`

**R signature:** `JWmatrix(graphL)`

**Input:** `graphL` — list of igraph objects with identical vertex names.

**Output:** A K×K matrix of weighted Jaccard distances, where `dist[i,j]` = 1 - weighted Jaccard similarity between graph i and graph j.

**Mathematical behavior:**

Same core computation as `JaccardAll` (F3) but differs in:
- **Diag handling:** `diag(sim.jac) = 1` (line 54) — self-similarity = 1, so self-distance = 0
- **Return value:** Returns the full K×K distance matrix (line 57), not the mean
- **Name propagation:** If `graphL` has names (list element names), they are propagated to row/col names of the distance matrix (lines 59-63)

**Python equivalent:**
```python
def jw_matrix(graphL: list) -> np.ndarray:
    """
    Returns: np.ndarray shape (K, K), weighted Jaccard distance matrix
    """
    # Same validation + flattening as jaccard_all
    K = len(graphL)
    N = graphL[0].vcount()
    M = N * (N - 1) // 2
    
    A = np.zeros((K, M), dtype=np.float64)
    for l, g in enumerate(graphL):
        adj = np.array(list(g.get_adjacency(attribute='weight')))
        lower_idx = np.tril_indices(N, k=-1)
        A[l, :] = adj[lower_idx]
    
    sim_jac = np.zeros((K, K), dtype=np.float64)
    for a, b in combinations(range(K), 2):
        mins = np.minimum(A[a, :], A[b, :])
        maxs = np.maximum(A[a, :], A[b, :])
        den = maxs.sum()
        if den == 0:
            sim_jac[a, b] = 0.0
            sim_jac[b, a] = 0.0
        else:
            sim_jac[a, b] = mins.sum() / den
            sim_jac[b, a] = sim_jac[a, b]
    
    sim_jac = np.nan_to_num(sim_jac, nan=0.0)
    np.fill_diagonal(sim_jac, 1.0)  # KEY DIFFERENCE from JaccardAll
    
    dist_jac = 1.0 - sim_jac
    return dist_jac
```

**Difference from `JaccardAll`:**
- `JaccardAll`: `diag(NA)` → excluded from mean → returns scalar
- `JWmatrix`: `diag(1)` → self-distance = 0 → returns matrix

---

### F5. `JWmean`

**Source:** `R/JaccardWeightedMean.R:14-56`

Identical to `JaccardAll` in behavior: validates names, computes pairwise weighted Jaccard, sets diag to NA, returns mean distance.

**Python equivalent:** Identical to `jaccard_all()` above.

---

### F6. `consensusNet` (THE CORE ALGORITHM)

**Source:** `R/ConsensusINet.R:34-589`

**R signature:**
```r
consensusNet(adjL, threshold=0.5, tolerance=0.1, theta=0.04,
             nitermax=50, ncores=2, verbose=TRUE)
```

#### F6.1 Input (Step 0)

**`adjL`**: List of K square matrices (preferably with row/col names). Weights in [0,1]. Must all have identical dimensions and identical row/col names. If not, user must first call `adj_rename()`.

#### F6.2 Phase 1: Graph Construction (lines 42-60)

For each matrix t in adjL:
- If rownames exist: `igraph::graph_from_adjacency_matrix(adjL[[t]], mode="upper", diag=FALSE, add.colnames="NA", weighted=TRUE)`
- If no rownames: same but without `add.colnames`

**Python:**
```python
graphs = []
for mat in adjL:
    if has_rownames(mat):
        g = igraph.Graph.Weighted_Adjacency(mat.tolist(), mode='upper', attr='weight')
        g.vs["name"] = list(rownames_of(mat))
    else:
        g = igraph.Graph.Weighted_Adjacency(mat.tolist(), mode='upper', attr='weight')
    graphs.append(g)
```

**Python-igraph note:** `Weighted_Adjacency` is a static method that takes a Python list-of-lists, NOT a numpy array. Use `.tolist()`.

#### F6.3 Phase 2: Internal Closures

**`weightMat(Weight, grafo, nodes)`** (lines 63-75):
```
In: Weight (float), grafo (igraph.Graph), nodes (vector of 2 integers = vertex indices)
Action:
  1. edgID = get.edge.ids(grafo, nodes)  # returns 0 if no such edge
  2. if edgID == 0:
       grafo = add_edges(grafo, nodes)  # adds edge with default weight 0
       edgID = get.edge.ids(grafo, nodes)
  3. E(grafo)$weight[edgID] = Weight
Out: grafo (modified in place)
```

**Python:**
```python
def weight_mat(weight: float, grafo: igraph.Graph, nodes: tuple[int, int]) -> igraph.Graph:
    try:
        eid = grafo.get_eid(nodes[0], nodes[1])
    except ValueError:
        grafo.add_edge(nodes[0], nodes[1], weight=0.0)
        eid = grafo.get_eid(nodes[0], nodes[1])
    grafo.es[eid]["weight"] = weight
    return grafo
```

**Critical difference:** R's `get.edge.ids` returns 0 for missing edges. Python's `Graph.get_eid` raises `ValueError` or returns -1 (depends on version). Must handle via try/except.

**`code(a,b)`** (line 218):
```
In: a, b — integers (igraph internal vertex indices, 0-based in Python)
Action: x = min(a,b); y = max(a,b); return (x+y)*(x+y+1)//2 + y
Out: integer — unique for unordered pair (a,b)
```

**Python:** Identical formula using integer arithmetic.
```python
def code(a: int, b: int) -> int:
    x, y = (a, b) if a < b else (b, a)
    return (x + y) * (x + y + 1) // 2 + y
```

**Note on 0-based vs 1-based indices:** R igraph vertex indices are 1-based. Python igraph indices are 0-based. The Cantor function operates on whatever indices are passed. The edgelist from `get_edgelist()` in Python returns 0-based indices naturally. The `code()` function must operate on these 0-based indices, which produces different integer keys than the R version (which uses 1-based). **This is fine** because the keys only need to be self-consistent within a single run — they never persist or cross-reference between runs. The key ordering is irrelevant to the algorithm's correctness.

**Note on integer overflow:** Python integers are arbitrary precision. No overflow risk. The formula `(x+y)*(x+y+1)//2` for vertices up to millions produces integers well within Python's fast-int range.

**`funNeig(x)`, `funWeights(x)`, `funegoWeights(x)`** (lines 212-230):

These are adapters called by R's `lapply`/`apply` to build hashmaps. Each captures outer variables (`m`, `s`, `t`, `node_id`, `edge_id`, `E`) via closure and mutates `node_id`/`edge_id` via `<<-`.

**Python:** Replace R's functional `lapply`/`apply` iteration with simple for-loops. No closures needed for hashmaps since Python's `dict` supports direct iteration.

Hashmap build for neighbors (R lines 249-253 → Python):
```python
neig_list = []  # Neig_list
for h in range(K):
    m = {}  # r2r::hashmap()
    adj_list = graphs[h].get_adjlist()
    for node_id, neighbors in enumerate(adj_list):
        m[node_id] = list(neighbors)
    neig_list.append(m)
```

Wait — the r2r hashmap is keyed by `node_id` (1,2,3,...) and stores neighbor *vectors*. This mimics `as_adj_list()` already. In fact, `igraph::as_adj_list(g)` returns a list where element `i` is the vector of neighbors of vertex `i`. The r2r hashmap then maps `1 → neighbors_of_v1`, `2 → neighbors_of_v2`, etc. This is equivalent to a Python list-of-lists or list-of-sets.

**CRITICAL OBSERVATION:** The r2r hashmap for neighbors simply mirrors the adjacency list. R's `as_adj_list(g)[[i]]` gives neighbors of vertex i. The storage as r2r hashmap with key=i is functionally identical to `neig_list[i]`. This means the neighbor hashmaps could be replaced with simple lists-of-sets directly.

Hashmap build for weights (R lines 257-261 → Python):
```python
weights_list = []  # Weights_list
for h in range(K):
    s = {}  # r2r::hashmap()
    edgelist_h = graphs[h].get_edgelist()
    for edge_id, (u, v) in enumerate(edgelist_h):
        weight = graphs[h].es[edge_id]["weight"]
        s[code(u, v)] = weight
    weights_list.append(s)
```

Hashmap for ego-weights (R lines 265-270 → Python):
```python
ego_weights_list = []  # EgoWeights_list
for h in range(K):
    t = {}  # r2r::hashmap()
    for edge_id, (u, v) in enumerate(union_edgelist):
        t[code(u, v)] = 0.0
    ego_weights_list.append(t)
```

**NOTE:** The ego-weights hashmap is built but NEVER USED as a hashmap (the insertions at lines 366 and 429 are commented out). The `EgoWeights_list` is constructed but all subsequent ego-weight lookups use `Weights_list`, not `EgoWeights_list`. The `EgoWeights_list` construction is dead code that wastes time. In migration, it can be removed.

#### F6.4 Phase 3: Core Iterative Loop

**Step A — Jaccard Distance** (line 147):
```python
comp = jaccard_all(graphs)
```

**Step B — Immediate Convergence** (lines 150-159):
```python
if count == 0 and comp < tolerance:
    if verbose: print("Multilayer network distance: less than tolerance.")
    break
```

**Step C — Save State** (line 164):
```python
graph_backup = [g.copy() for g in graphs]
```

**Step D — Union Graph** (lines 168-173):
```python
union_g = graphs[0].copy()
for g in graphs[1:]:
    union_g = union_g.union(g)
```
**CRITICAL — Union semantics for weighted graphs:** In igraph, `union()` of weighted graphs merges vertices by ID/name and merges edges. When the same edge exists in both graphs, the resulting edge gets the attribute from the first graph (in R's C implementation). However, **the union graph's edge weights are never used in the algorithm** — only `as_edgelist(union_g, names=FALSE)` is called to get which edges exist. So the edge attribute merge semantics don't matter. What matters is that the union correctly identifies all edges that exist in ANY graph.

**Simplification:** The union graph is used solely to enumerate all edges that appear in at least one layer. This can be simplified to:
```python
union_edges = set()
for g in graphs:
    for edge in g.get_edgelist():
        union_edges.add(tuple(sorted(edge)))
union_edgelist = [[u,v] for u,v in union_edges]
```

This avoids the `igraph::union` semantic ambiguity entirely and is more efficient.

**Step E — Extract Union Edgelist** (line 175):
```python
# If using igraph union:
# edgelist = union_g.get_edgelist()  # list of (u,v) tuples, integer indices
# If using set approach:
edgelist = sorted(union_edges)  # list of (u,v) tuples
```

**Step F — Parallel Weight Update** (lines 195-460):

For each graph `i` (0 to K-1), compute new weights for all edges in the union set.

**Building structures** (per worker, for all graphs):

```python
K = len(graphs)

# Neighbor sets for each graph
neig_list = []
for h in range(K):
    adj = graphs[h].get_adjlist()
    neig_list.append([set(neighbors) for neighbors in adj])

# Weight hashmaps for each graph
weights_list = []
for h in range(K):
    wmap = {}
    for e in graphs[h].es:
        u, v = e.source, e.target
        wmap[code(u, v)] = e["weight"]
    weights_list.append(wmap)

# Intersect_list — computed per edge in the loop below
intersect_list = [None] * K
```

**Per-edge computation** (for each (u,v) in union edgelist):

```python
for u, v in edgelist:
    # --- Neighbor intersection for graph i ---
    inei = neig_list[i][u]
    jnei = neig_list[i][v]
    intersect_list[i] = inei & jnei  # set intersection

    # --- Edge weight in graph i ---
    w_uso = weights_list[i].get(code(u, v), 0.0)

    # --- Edge weights and intersections in other graphs ---
    w_others = []
    for j in range(K):
        if j == i:
            continue
        intersect_list[j] = neig_list[j][u] & neig_list[j][v]
        w_altri = weights_list[j].get(code(u, v), 0.0)
        w_others.append(w_altri)

    # Base weight
    peso = (w_uso + np.mean(w_others)) / 2.0

    # --- Ego weight for graph i (pesiEgoNUS) ---
    if len(intersect_list[i]) == 0:
        pesi_ego_nus = 0.0
    else:
        # Build flattened list of all intersections for numberCom
        all_intersections_flat = []
        for j in range(K):
            all_intersections_flat.extend(intersect_list[j])
        from collections import Counter
        com_counter = Counter(all_intersections_flat)

        pint_sum = 0.0
        for k in intersect_list[i]:
            p1 = weights_list[i].get(code(u, k), 0.0)
            p2 = weights_list[i].get(code(v, k), 0.0)
            number_com = com_counter[k]
            pint_sum += (number_com / K) * (p1 + p2)

        if w_uso == 0:
            denomin = len(inei) + len(jnei)
        else:
            denomin = len(inei) + len(jnei) - 2

        if denomin == 0:
            pesi_ego_nus = 0.0
        else:
            pesi_ego_nus = (pint_sum / denomin) ** (1.0 / len(intersect_list[i]))

    # --- Ego weights for other graphs (pesiEgoOthers) ---
    pesi_ego_others = []
    for j in range(K):
        if j == i:
            continue

        if len(intersect_list[j]) == 0:
            pesi_ego_altri = 0.0
        else:
            pint_sum = 0.0
            for k in intersect_list[j]:
                p1 = weights_list[j].get(code(u, k), 0.0)
                p2 = weights_list[j].get(code(v, k), 0.0)
                number_com = com_counter[k]
                pint_sum += (number_com / K) * (p1 + p2)

            graphs_other = [x for x in range(K) if x != i]
            pos = graphs_other.index(j)
            if w_others[pos] == 0:
                denomin = len(neig_list[j][u]) + len(neig_list[j][v])
            else:
                denomin = len(neig_list[j][u]) + len(neig_list[j][v]) - 2

            if denomin == 0:
                pesi_ego_altri = 0.0
            else:
                pesi_ego_altri = (pint_sum / denomin) ** (1.0 / len(intersect_list[j]))

        pesi_ego_others.append(pesi_ego_altri)

    # Final ego weight
    pesi_ego = (pesi_ego_nus + np.mean(pesi_ego_others)) / 2.0 if pesi_ego_others else pesi_ego_nus

    # Final edge weight
    weight_i1 = peso + theta * pesi_ego
    if weight_i1 > 1.0:
        weight_i1 = 1.0

    # Update graph i
    graphs[i] = weight_mat(weight_i1, graphs[i], (u, v))
```

**Divergence note on `mean(pesiEgoOthers)`:** When K=2, there's only one "other" graph (j ≠ i). `mean(wOthers)` with 1 element equals that element. `mean(pesiEgoOthers)` similarly. Python's `np.mean` on a single-element list returns that element, matching R.

**Divergence note on `length(graph)` in numberCom:** The formula uses `numberCom / length(graph)` where `length(graph)` = K (total number of layers). This is correct: the multiplier accounts for how many of the K layers share this common neighbor.

**NOTE ON PARALLELIZATION:**
R's `parallel::clusterApply(cl, vet, function(i) { ... graph[[i]] })` runs each graph index `i` on a separate worker. Each worker:
1. Receives a COPY of `graph`, `Edgelist`, `theta`, `weightMat` via `clusterExport`
2. Rebuilds all hashmaps for ALL graphs locally (redundant but ensures data locality)
3. Modifies `graph[[i]]` only (its assigned graph)
4. Returns `graph[[i]]`
5. The main process collects all K modified graphs into `Graphs` list

Python equivalent:
```python
from copy import deepcopy
from multiprocessing import Pool

def process_graph(i, graphs_data, edgelist, theta):
    # Rebuild local copies from serialized data
    graphs = deserialize(graphs_data)  # deep copy
    # ... compute new weights for edges, modify graphs[i] ...
    return serialize(graphs[i])

with Pool(processes=ncores) as pool:
    args = [(i, serialize(graphs), edgelist, theta) for i in range(K)]
    results = pool.starmap(process_graph, args)
    new_graphs = [deserialize(r) for r in results]
```

For single-process execution (ncores=1 or debugging), use a simple for-loop.

#### F6.5 Phase 4: Convergence Check (lines 462-530)

```python
graph_change = new_graphs  # from parallel results
graph_original = graph_backup  # restore to compare

# Compute post-iteration distance
comp_post = jaccard_all(graph_change)
if verbose:
    print(f"Multilayer network distance: {comp_post}")

if count == 0:
    comparison = [comp]  # Comparison starts with initial distance
comparison.append(comp_post)

# Check 1: Did distance decrease?
if comp > comp_post:
    graphs = graph_change  # accept new weights
else:
    if verbose:
        print("Distance doesn't decrease.")
    break

count += 1

# Check 2: Max iterations?
if count > nitermax:
    if verbose:
        print("Maximum iteration.")
    break

# Check 3: Within tolerance?
if comp_post < tolerance:
    break
```

#### F6.6 Phase 5: Consensus Construction (lines 534-569)

```python
# Convert graphs to adjacency matrices
mats = []
for g in graphs:
    adj = np.array(list(g.get_adjacency(attribute='weight')))
    mats.append(adj)

# Element-wise mean
matrix_mean = np.mean(np.stack(mats, axis=0), axis=0)

# Threshold
matrix_mean[matrix_mean < threshold] = 0.0

# Reassign names
if has_rownames(adjL[0]):
    # Set row/col order from original names
    names = rownames_of(adjL[0])
    # matrix_mean already has the right order since graphs preserve it
    pass

# Create consensus graph
consensus_g = igraph.Graph.Weighted_Adjacency(
    matrix_mean.tolist(), mode='upper', attr='weight'
)
if has_rownames(adjL[0]):
    consensus_g.vs["name"] = dict(enumerate(names)).values()  # preserve order

# Reassign names to similar graphs
if has_rownames(adjL[0]):
    names = rownames_of(adjL[0])
    for g in graphs:
        g.vs["name"] = list(names)
```

#### F6.7 Return Value (lines 584-588)

```python
return {
    "graphConsensus": consensus_g,
    "Comparison": np.array(comparison),  # shape (T,), distances per iteration
    "similarGraphs": graphs
}
```

**Structure of Comparison:**
- `Comparison[0]` = initial weighted Jaccard distance before any iteration
- `Comparison[1]` = distance after iteration 1
- `Comparison[t]` = distance after acceptance of iteration t
- Length = 1 + number_of_accepted_iterations

---

### F7. `constructionGraph`

**Source:** `R/ConstructionGraph.R:20-62`

**R signature:** `constructionGraph(data, perc=0.95)`

**Input:** `data` — list of matrices where columns = nodes/features and rows = samples/observations.

**Mathematical behavior:**

For each data layer i:
1. `corr[j,k] = Pearson_correlation(data[[i]][,j], data[[i]][,k])` — columns are variables
2. Plot histogram of lower-triangle correlation values
3. `threshold = quantile(corr_matrix, perc)` — computes perc-th quantile of the FULL correlation matrix (not lower triangle)
4. `corr_matrix[corr_matrix < threshold] = 0`
5. Build igraph from thresholded matrix (upper triangular, undirected, weighted)
6. Compute Louvain modularity: `modularity(cluster_louvain(g))`
7. Collect stats: threshold value, edge count, vertex count, modularity

**Output:** List with:
- `$Threshold`: list of named vectors (threshold quantile value, edge count, node count, modularity)
- `$Graphs`: list of igraph objects
- `$Adj`: list of thresholded correlation matrices

**Side effects:** Prints a histogram grid via `ggpubr::ggarrange`.

**R dependencies:** `stats::cor`, `stats::quantile`, `igraph`, `ggplot2`, `ggpubr`.

**Python equivalent:**
```python
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def construction_graph(data: list[np.ndarray], perc: float = 0.95, plot: bool = True) -> dict:
    """
    data: list of np.ndarray, each shape (samples, nodes)
          columns = nodes, rows = observations
    perc: percentile threshold (default 0.95 = keep top 5%)
    """
    K = len(data)
    graphs_list = []
    corr_list = []
    threshold_info = []

    for i in range(K):
        # Pearson correlation matrix (columns as variables)
        mat = data[i]  # shape (samples, nodes)
        corr_mat = np.corrcoef(mat, rowvar=False)  # rowvar=False = columns are variables
        
        # Plot histogram of lower triangle values
        if plot:
            lower_vals = corr_mat[np.tril_indices_from(corr_mat, k=-1)]
            plt.hist(lower_vals, bins=30, color='#69b3a2', edgecolor='#e9ecef')
            plt.xlabel(f'Weights {i}')
            plt.show()  # or collect into subplots
        
        # Quantile threshold
        thresh = np.quantile(corr_mat, perc, method='linear')
        corr_mat[corr_mat < thresh] = 0.0
        
        # Build graph
        g = igraph.Graph.Weighted_Adjacency(corr_mat.tolist(), mode='upper', attr='weight')
        
        # Louvain modularity
        communities = g.community_multilevel(weights=None)  # or community_leiden
        mod = g.modularity(communities.membership)
        
        e = g.ecount()
        v = g.vcount()
        
        graphs_list.append(g)
        corr_list.append(corr_mat)
        threshold_info.append({
            str(perc): thresh,
            "edge": e,
            "node": v,
            "modularity Louvain": mod
        })
    
    return {
        "Threshold": threshold_info,
        "Graphs": graphs_list,
        "Adj": corr_list
    }
```

**Key details:**
- `numpy.corrcoef(mat, rowvar=False)` treats columns as variables, matching R's `cor(mat)` (R treats columns as variables by default).
- `np.quantile(corr_mat, perc, method='linear')` — R's `quantile` default type=7 uses linear interpolation. numpy's method='linear' matches this.
- `community_multilevel` in python-igraph corresponds to R's `cluster_louvain`. Some python-igraph versions may have renamed it. Use `community_leiden` if multilevel is deprecated, though the algorithm differs slightly.
- The `format(ecount, scientific=FALSE)` in R converts large numbers to non-scientific notation string. In Python, use `str(e)` which does the same by default.

---

### F8. `densityNet`

**Source:** `R/DensityWeight.R:21-78`

**R signature:** `densityNet(graphL)`

**Input:** List of igraph graphs.

**Mathematical behavior:**
1. Convert each graph to adjacency matrix
2. Compute element-wise mean across all matrices → `matrixMean` (N×N)
3. Extract lower triangle (no diag) of `matrixMean` → `vect`
4. Compute quantiles at `seq(0, 1, 0.05)` (21 quantile points) for all values → `quant`
5. Filter to values > 0 → `vect0`
6. Compute quantiles of non-zero values → `quant0`
7. Plot density of non-zero values

**Output:** List with `$quantile` and `$quantileNo0`.

**Side effects:** Prints density plot.

**Python equivalent:**
```python
def density_net(graphL: list) -> dict:
    mats = []
    for g in graphL:
        adj = np.array(list(g.get_adjacency(attribute='weight')))
        mats.append(adj)
    
    matrix_mean = np.mean(np.stack(mats, axis=0), axis=0)
    
    # Lower triangle values (no diag)
    lower_idx = np.tril_indices_from(matrix_mean, k=-1)
    vect = matrix_mean[lower_idx]
    
    # Quantiles at 0, 0.05, ..., 1.0
    probs = np.arange(0, 1.01, 0.05)
    quant = np.quantile(vect, probs, method='linear')
    
    # Non-zero values only
    vect0 = vect[vect > 0]
    quant0 = np.quantile(vect0, probs, method='linear') if len(vect0) > 0 else np.full_like(probs, np.nan)
    
    # Plot density
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(vect0, bins=30, density=True, color='#69b3a2', edgecolor='#e9ecef')
    plt.xlabel("mean weights without 0")
    plt.show()
    
    return {
        "quantile": quant,
        "quantileNo0": quant0
    }
```

---

### F9. `thresholdNet`

**Source:** `R/ThresholdConsensus.R:20-65`

**R signature:** `thresholdNet(sim.graphL, threshold=0.5)`

**Input:** `sim.graphL` — list of similar graphs (output of `consensusNet`'s `$similarGraphs`).

**Mathematical behavior:**
1. Convert each graph to adjacency matrix (with `names=TRUE` to preserve row/col names)
2. Element-wise mean across matrices
3. Threshold: all values < threshold become 0
4. Reassign row/col names
5. Build new igraph

**Output:** Single igraph object (the new consensus).

**Python equivalent:**
```python
def threshold_net(sim_graphL: list, threshold: float = 0.5) -> igraph.Graph:
    mats = []
    for g in sim_graphL:
        adj = np.array(list(g.get_adjacency(attribute='weight')))
        mats.append(adj)
    
    matrix_mean = np.mean(np.stack(mats, axis=0), axis=0)
    matrix_mean[matrix_mean < threshold] = 0.0
    
    g = igraph.Graph.Weighted_Adjacency(matrix_mean.tolist(), mode='upper', attr='weight')
    if "name" in sim_graphL[0].vs.attributes():
        g.vs["name"] = sim_graphL[0].vs["name"]
    
    return g
```

---

### F10. `specificNet`

**Source:** `R/SpecificFunction.R:21-66`

**R signature:** `specificNet(graphL, graph.consensus)`

**Input:** 
- `graphL` — list of original layer graphs
- `graph.consensus` — consensus graph from `consensusNet`

**Mathematical behavior:**
For each graph t:
- `GraphsDifference[[t]] = igraph::difference(graphL[[t]], graph.consensus)` — edges in graph t but NOT in consensus
- `percentageOfSpecificity[t] = ecount(GraphsDifference[[t]]) / ecount(graphL[[t]])` — fraction of edges unique to this layer

**Output:** List with `$GraphsDifference` and `$percentageOfSpecificity`.

**Python equivalent:**
```python
def specific_net(graphL: list, graph_consensus: igraph.Graph) -> dict:
    diffs = []
    percentages = []
    for g in graphL:
        diff = g.difference(graph_consensus)
        diffs.append(diff)
        pct = diff.ecount() / g.ecount() if g.ecount() > 0 else 0.0
        percentages.append(pct)
    
    return {
        "GraphsDifference": diffs,
        "percentageOfSpecificity": percentages
    }
```

---

### F11. `measuresNet`

**Source:** `R/NetworkMeasures.R:17-78`

**R signature:** `measuresNet(graphL, nodes.measures=TRUE)`

**Graph-level measures computed (lines 24-35):**

| Measure | R call | Python equivalent |
|---------|--------|-------------------|
| Vertex count | `vcount(g)` | `g.vcount()` |
| Edge count | `ecount(g)` | `g.ecount()` |
| Transitivity (global) | `transitivity(g, type="global")` | `g.transitivity_undirected()` |
| Diameter | `diameter(g)` | `g.diameter()` |
| Modularity (Louvain) | `modularity(g, cluster_louvain(g, weights=NULL)$membership)` | `g.community_multilevel(weights=None)`. Then `g.modularity(com.membership)` |
| Edge density | `edge_density(g)` | `g.density()` |
| Assortativity | `assortativity_degree(g)` | `g.assortativity_degree()` |
| Degree centralization | `centr_degree(g)$centralization` | **No direct equivalent.** Implement Freeman's formula manually. |
| Betweenness centralization | `centr_betw(g)$centralization` | **No direct equivalent.** Implement Freeman's formula manually. |

**Node-level measures** (if `nodes.measures=TRUE`, lines 42-46):

| Measure | R call | Python equivalent |
|---------|--------|-------------------|
| Degree | `degree(g)` | `g.degree()` |
| Local transitivity | `transitivity(g, type="local")` | `g.transitivity_local_undirected()` |
| Betweenness | `betweenness(g)` | `g.betweenness()` |
| Hub score | `hub_score(g, weights=NULL)$vector` | `g.hub_score(weights=None).vector` if weights=None (depends on version) |

**Centralization formulas** (Freeman's):

Degree centralization:
```
deg = degree(g)
max_deg = max(deg)
n = vcount(g)
centralization = sum(max_deg - deg) / ((n-1)*(n-2))
```
Reference: Freeman, L.C. (1979). Centrality in networks: I. Conceptual clarification. Social Networks.

Betweenness centralization:
```
bet = betweenness(g)
max_bet = max(bet)
n = vcount(g)
# For undirected graphs:
centralization = 2 * sum(max_bet - bet) / ((n-1)*(n-2)*(n-1))
```
Note: The exact formula depends on igraph's implementation. Recommend running R and Python side-by-side to verify or simply implementing the formula from the igraph C source.

**Python equivalent:**
```python
def measures_net(graphL: list, nodes_measures: bool = True) -> list:
    results = []
    for g in graphL:
        v = g.vcount()
        e = g.ecount()
        tran = g.transitivity_undirected()
        diam = g.diameter()
        com = g.community_multilevel(weights=None)
        modl = g.modularity(com.membership)
        den = g.density()
        ass = g.assortativity_degree()
        
        # Degree centralization (Freeman)
        deg = g.degree()
        max_deg = max(deg)
        ceD = sum(max_deg - d for d in deg) / ((v - 1) * (v - 2))
        
        # Betweenness centralization
        bet = g.betweenness()
        max_bet = max(bet)
        ceB = 2 * sum(max_bet - b for b in bet) / ((v - 1) * (v - 2) * (v - 1))
        
        graph_measures = {
            "vertices": v, "edges": e,
            "transitivity": tran, "diameter": diam,
            "modularityLouvain": modl, "edgeDensity": den,
            "assortativity": ass,
            "centrDegree": ceD, "centrBetween": ceB
        }
        
        if nodes_measures:
            nd = g.degree()
            ntran = g.transitivity_local_undirected()
            nbet = g.betweenness()
            # Hub score — some python-igraph versions have it
            try:
                nhub = g.hub_score(weights=None)
            except:
                nhub = [np.nan] * v
            
            node_measures = {
                "degree": nd, "transitivityLocal": ntran,
                "betweenness": nbet, "hub": nhub
            }
            results.append({"graphsMeasures": graph_measures, "nodeMeasures": node_measures})
        else:
            results.append({"graphsMeasures": graph_measures})
    
    return results
```

---

### F12. `plotINet`

**Source:** `R/Plots.R:34-118`

**R signature:** `plotINet(adj, graph.consensus, edge.width=3, vertex.label.cex=0.5, vertex.size=10, edge.curved=0.2, method="NA", ...)`

**Behavior:**
1. Convert `adj` (adjacency matrix) to igraph graph → `graph`
2. Ensure both graphs have vertex names. If not: set to string representation of vertex index.
3. Add any vertices present in consensus but not in the original → `Graph`
4. Compute `UnionGraph = union(Graph, graph.consensus)`
5. Compute `Diff = difference(graph.consensus, Graph)` — edges only in consensus
6. Compute `Inter = intersection(graph.consensus, Graph)` — edges in both
7. Color edges: gray (base) → light blue `#619CFF` (diff) → red `#F8766D` (intersection)
8. Community detection on `Graph` using `robin::membershipCommunities(method=method, ...)`; if method="NA", use single green color
9. Remove isolated vertices
10. Plot with base R `plot.igraph()`

**Python equivalent:**
```python
def plot_inet(adj, graph_consensus, edge_width=3, vertex_size=10, 
              edge_curved=0.2, method="NA", **kwargs):
    # Convert adjacency to graph
    g = igraph.Graph.Weighted_Adjacency(adj.tolist(), mode='upper', attr='weight')
    
    # Ensure vertex names
    if "name" not in graph_consensus.vs.attributes():
        graph_consensus.vs["name"] = [str(i) for i in range(graph_consensus.vcount())]
    if "name" not in g.vs.attributes():
        g.vs["name"] = [str(i) for i in range(g.vcount())]
    
    # Add missing vertices
    cons_names = set(graph_consensus.vs["name"])
    g_names = set(g.vs["name"])
    to_add = cons_names - g_names
    if to_add:
        g.add_vertices(len(to_add))
        g.vs[-len(to_add):]["name"] = list(to_add)
    
    # Set operations
    union_g = g.union(graph_consensus)
    diff_g = graph_consensus.difference(g)
    inter_g = graph_consensus.intersection(g)
    
    # Edge colors
    ecol = ["gray80"] * union_g.ecount()
    
    if diff_g.ecount() > 0:
        for e in diff_g.es:
            src_name = diff_g.vs[e.source]["name"]
            tgt_name = diff_g.vs[e.target]["name"]
            eid = union_g.get_eid(src_name, tgt_name, error=False)
            if eid != -1:
                ecol[eid] = "#619CFF"
    
    if inter_g.ecount() > 0:
        for e in inter_g.es:
            src_name = inter_g.vs[e.source]["name"]
            tgt_name = inter_g.vs[e.target]["name"]
            eid = union_g.get_eid(src_name, tgt_name, error=False)
            if eid != -1:
                ecol[eid] = "#F8766D"
    
    # Community detection
    if method == "NA":
        members = "#00BA38"
    else:
        # Use python-igraph community detection directly
        if method == "louvain":
            com = g.community_multilevel()
        elif method == "leiden":
            com = g.community_leiden()
        # ... other methods
        members = com.membership
    
    # Remove isolated
    deg = union_g.degree()
    isolated = [v for v, d in enumerate(deg) if d == 0]
    union_g_clean = union_g.copy()
    union_g_clean.delete_vertices(isolated)
    
    # Python-igraph plotting
    visual_style = {
        "vertex_size": vertex_size,
        "edge_width": edge_width,
        "edge_color": ecol,
        "vertex_color": members,
        "edge_curved": edge_curved,
        "bbox": (600, 600),
        "margin": 50
    }
    igraph.plot(union_g_clean, **visual_style)
```

**Note:** The `robin::membershipCommunities` function wraps many igraph community detection methods. In Python, use python-igraph's built-in methods directly. Method name mapping:
- "walktrap" → `g.community_walktrap().as_clustering()`
- "edgeBetweenness" → `g.community_edge_betweenness().as_clustering()`
- "fastGreedy" → `g.community_fastgreedy().as_clustering()`
- "louvain" → `g.community_multilevel()` (R's `cluster_louvain`)
- "spinglass" → `g.community_spinglass()`
- "leadingEigen" → `g.community_leading_eigenvector()`
- "labelProp" → `g.community_label_propagation()`
- "infomap" → `g.community_infomap()`
- "optimal" → `g.community_optimal_modularity()`
- "leiden" → `g.community_leiden()`

---

### F13. `plotL`

**Source:** `R/Plots.R:138-161`

R implementation uses the `multinet` R package (`ml_empty`, `add_igraph_layer_ml`) for multi-layer visualization. No direct Python equivalent exists.

**Python fallback:** Plot each layer in separate subplots using matplotlib or python-igraph's plot:
```python
def plot_l(graphL: list) -> None:
    fig, axes = plt.subplots(1, len(graphL), figsize=(5*len(graphL), 5))
    if len(graphL) == 1:
        axes = [axes]
    for ax, g in zip(axes, graphL):
        # Ensure names
        if "name" not in g.vs.attributes():
            g.vs["name"] = [str(i) for i in range(g.vcount())]
        igraph.plot(g, target=ax, **kwargs)
    plt.show()
```

---

### F14. `plotC`

**Source:** `R/Plots.R:181-187`

Simple: remove isolated nodes, plot.

**Python:**
```python
def plot_c(graph: igraph.Graph, **kwargs) -> None:
    deg = graph.degree()
    isolated = [v for v, d in enumerate(deg) if d == 0]
    g_clean = graph.copy()
    g_clean.delete_vertices(isolated)
    igraph.plot(g_clean, **kwargs)
```

---

## Part 3: Consolidated Python API

```python
# Module: inet_python

# Internal
from inet_python._internal import get_lower_tri_noDiag, code, weight_mat

# Pre-processing
from inet_python.preprocess import adj_rename, construction_graph

# Distance metrics
from inet_python.distance import jw_matrix, jw_mean, jaccard_all

# Core algorithm
from inet_python.consensus import consensus_net

# Post-processing
from inet_python.postprocess import threshold_net, density_net, specific_net

# Analysis
from inet_python.measures import measures_net

# Visualization
from inet_python.plots import plot_inet, plot_l, plot_c
```

---

## Part 4: Complete Dependencies (Python)

```
# Core
numpy >= 1.22        # quantile(..., method='linear')
scipy >= 1.8         # pearsonr (optional, could use numpy.corrcoef)
python-igraph >= 0.10

# Parallel
multiprocessing (stdlib)
concurrent.futures (stdlib)

# Optional — plotting
matplotlib >= 3.5
seaborn (optional, for density plots)

# Optional — community detection
cdlib (optional, more algorithms)

# Development
pytest
pytest-benchmark
```

---

## Part 5: Execution Order for Implementation

1. **`_internal.py`**: `get_lower_tri_noDiag`, `code`, `weight_mat`
2. **`distance.py`**: `jaccard_all`, `jw_matrix`, `jw_mean` (depends on #1)
3. **`consensus.py`**: `consensus_net` (depends on #1, #2)
4. **`preprocess.py`**: `adj_rename`, `construction_graph` (depends on #1)
5. **`postprocess.py`**: `density_net`, `threshold_net`, `specific_net` (depends on igraph, numpy)
6. **`measures.py`**: `measures_net` (depends on igraph, numpy)
7. **`plots.py`**: `plot_c`, `plot_inet`, `plot_l` (depends on igraph, matplotlib)
