# Dependency Analysis — INet-Tool

## R Package Dependencies (from DESCRIPTION)

| Package | Version Constraint | Type | Purpose in Codebase |
|---------|-------------------|------|---------------------|
| R | >= 3.5.0 | Depends | Base R runtime |
| **igraph** | — | Imports | Core graph data structure, all graph operations (construction, manipulation, measures, community detection, set operations) |
| **r2r** | — | Imports | Hashmap data structure for O(1) edge/neighbor lookups in the inner algorithm loop |
| **ggplot2** | — | Imports | Histograms (constructionGraph), density plots (densityNet), general plotting |
| **parallel** | — | Imports | `makeCluster`, `clusterExport`, `clusterApply`, `stopCluster` — parallel processing across graphs |
| **ggpubr** | — | Imports | `ggarrange` — arranging multiple ggplot objects in a grid |
| **multinet** | — | Imports | Multi-layer network visualization (`ml_empty`, `add_igraph_layer_ml`, plot) |
| **robin** | — | Imports | `membershipCommunities` — community detection wrapper used by plotINet |
| **stats** | base R | ImportFrom | `cor` (Pearson correlation), `quantile` |
| **utils** | base R | (implicit) | `combn` — generating all pairwise combinations |
| knitr | — | Suggests | Vignette building |
| rmarkdown | — | Suggests | Vignette building |

---

## Function-Level Dependency Graph

```
Legend:
  → means "calls"
  [E] = exported function
  [I] = internal function
  [C] = closure (defined inside another function)

consensusNet [E]
    ├── igraph::graph_from_adjacency_matrix
    ├── JaccardAll [C]
    │   ├── utils::combn
    │   ├── igraph::V, igraph::as_adjacency_matrix
    │   ├── get_lower_tri_noDiag [C] (duplicate)
    │   └── (base) sapply, sum, min, max, mean
    ├── igraph::union
    ├── igraph::as_edgelist
    ├── parallel::makeCluster
    ├── parallel::clusterExport
    ├── parallel::clusterApply
    │   └── (per-worker closure)
    │       ├── r2r::hashmap
    │       ├── r2r::insert
    │       ├── igraph::as_adj_list, igraph::as_edgelist, igraph::E
    │       ├── funNeig [C]
    │       ├── code [C]  (Cantor pairing)
    │       ├── funWeights [C]
    │       ├── funegoWeights [C]
    │       ├── weightMat [C]
    │       │   ├── igraph::get.edge.ids
    │       │   ├── igraph::add_edges
    │       │   └── igraph::E<-
    │       └── (base) intersect, table, unlist, mean, sum, length, min, max, sapply
    ├── parallel::stopCluster
    ├── igraph::as_adjacency_matrix (for consensus construction)
    └── (base) mean, matrix operations

adj_rename [E]
    └── (base only) unique, rownames, colnames, dim, which, matrix

constructionGraph [E]
    ├── stats::cor
    ├── get_lower_tri_noDiag [I]
    ├── ggplot2::ggplot, ggplot2::geom_histogram, ggplot2::aes, ggplot2::xlab
    ├── stats::quantile
    ├── igraph::graph_from_adjacency_matrix
    ├── igraph::modularity, igraph::cluster_louvain
    ├── igraph::ecount, igraph::vcount
    └── ggpubr::ggarrange

densityNet [E]
    ├── igraph::as_adjacency_matrix
    ├── get_lower_tri_noDiag [I]
    ├── stats::quantile
    └── ggplot2::ggplot, ggplot2::geom_density, ggplot2::aes, ggplot2::xlab

JWmatrix [E]
    ├── utils::combn
    ├── igraph::V, igraph::as_adjacency_matrix
    └── get_lower_tri_noDiag [I]

JWmean [E]
    ├── utils::combn
    ├── igraph::V, igraph::as_adjacency_matrix
    └── get_lower_tri_noDiag [I]

measuresNet [E]
    ├── igraph::vcount, igraph::ecount
    ├── igraph::transitivity
    ├── igraph::diameter
    ├── igraph::modularity, igraph::cluster_louvain
    ├── igraph::edge_density
    ├── igraph::assortativity_degree
    ├── igraph::centr_degree, igraph::centr_betw
    ├── igraph::degree (node)
    ├── igraph::betweenness (node)
    └── igraph::hub_score (node)

plotINet [E]
    ├── igraph::graph_from_adjacency_matrix
    ├── igraph::V, igraph::V<-
    ├── igraph::add_vertices
    ├── igraph::union, igraph::difference, igraph::intersection
    ├── igraph::ecount, igraph::as_edgelist, igraph::get.edge.ids
    ├── robin::membershipCommunities
    ├── igraph::degree, igraph::delete.vertices
    └── (base) plot

plotL [E]
    ├── igraph::V, igraph::V<-, igraph::vcount
    ├── multinet::ml_empty
    └── multinet::add_igraph_layer_ml

plotC [E]
    ├── igraph::degree
    └── igraph::delete.vertices

specificNet [E]
    ├── igraph::difference
    └── igraph::ecount

thresholdNet [E]
    ├── igraph::as_adjacency_matrix
    └── igraph::graph_from_adjacency_matrix

get_lower_tri_noDiag [I]
    └── (base) upper.tri, diag
```

---

## Dependency Criticality Assessment

### CRITICAL (algorithm cannot function without)
| Package | Justification |
|---------|---------------|
| **igraph** | Entire data model is built on igraph objects. All graph operations, measures, and set operations depend on it. |
| **r2r** | The hashmap data structure is essential for O(1) edge weight and neighbor lookups in the inner loop. Without it, performance would degrade from O(E × k) to O(E × V). |

### HIGH (feature removal would significantly break functionality)
| Package | Justification |
|---------|---------------|
| **parallel** | The inner loop is parallelized per-graph. Without it, the algorithm runs but may be significantly slower. Parallelization is done once per algorithm run (not per iteration), so overhead is manageable. |

### MEDIUM (affects auxiliary functionality)
| Package | Justification |
|---------|---------------|
| **ggplot2** | Used in `constructionGraph` (weight histograms) and `densityNet` (density plot). These are visualization aids, not core algorithm components. |
| **ggpubr** | `ggarrange` is only used to display histogram grids in `constructionGraph`. Purely cosmetic. |
| **stats** | `cor` used only in `constructionGraph` (optional pre-processing). `quantile` used in `constructionGraph` and `densityNet` — both are helpers, not the core algorithm. |

### LOW (cosmetic / convenience only)
| Package | Justification |
|---------|---------------|
| **multinet** | Only used in `plotL()` for multi-layer visualization. No algorithmic dependency. |
| **robin** | Only used in `plotINet()` for community detection coloring. Default `method="NA"` bypasses it entirely. |
| **knitr, rmarkdown** | Vignette building only. Not needed at runtime. |

---

## Data Structure Dependencies

The codebase relies on these specific igraph object properties:

1. **igraph vertex IDs**: The `code()` Cantor pairing function (line 218 of ConsensusINet.R) operates on igraph's internal integer vertex IDs. Edgelist is extracted with `names=FALSE` to get these IDs. Any replacement must preserve 1-based integer vertex identification.

2. **igraph weighted edges**: Edge weights are stored as edge attributes (`weight`), accessed via `igraph::E(g)$weight[edge_id]`. The `weightMat` closure modifies edge weights by ID.

3. **igraph upper-triangular adjacency**: All `graph_from_adjacency_matrix` calls use `mode="upper"` with `diag=FALSE`. This means the graphs are undirected and the lower triangle of the adjacency matrix is ignored during construction.

4. **igraph set operations**: `union`, `difference`, `intersection` are used in `consensusNet`, `plotINet`, and `specificNet`. These are igraph-specific and their semantics (how edge attributes are handled, how vertex sets are merged) need precise replication.

5. **r2r hashmap specifics**:
   - `r2r::hashmap()` creates an empty hashmap
   - `r2r::insert(m, key, value)` inserts a key-value pair (side-effect on m)
   - `m[key]` retrieves a value
   - Keys are integers (from `code()` pairing function)
   - Values are vectors (neighbors list) or scalars (weights)

6. **S3 generics**: The code uses S3 dispatch for `plot()`, `as_edgelist()`, `E()`, `V()`, etc. All are igraph S3 methods.

---

## Hidden Assumptions

1. **Integer vertex IDs**: The `code()` function assumes vertex IDs are small enough that `(x+y)*(x+y+1)/2 + y` fits in R's numeric representation and is unique for all pairs. This holds for vertex IDs up to ~10^7 before potential precision issues.

2. **Undirected graphs only**: All operations use `mode="upper"`, enforcing symmetric adjacency. Directed graphs are not supported.

3. **Weights in [0,1]**: The algorithm assumes edge weights in [0,1] and clamps values exceeding 1. Negative weights are not handled.

4. **Symmetric adjacency matrices**: The algorithm extracts only the lower triangle for Jaccard computation, assuming symmetry. Non-symmetric inputs would silently lose half the information.

5. **No self-loops**: `diag=FALSE` in all graph constructions. Self-loop information is discarded.

6. **r2r parallel serialization**: r2r hashmaps are exported to cluster workers via `parallel::clusterExport`. This requires that r2r objects survive serialization/deserialization across the parallel transport.

7. **Named rows/columns optional**: The code checks `length(rownames(adjL[[1]]))>0` before using names. Functions work with unnamed matrices, using positional indices.
