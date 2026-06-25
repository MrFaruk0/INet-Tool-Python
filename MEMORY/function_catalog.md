# Function Catalog — INet-Tool

Every function is documented with: source file, line range, signature, type (public/internal/closure), dependencies, and behavioral notes. Statements are backed by actual source code.

---

## Exported (Public API) Functions — 12 total

### 1. `adj_rename`
- **File:** `R/AdjSameName.R`, lines 15–59
- **Signature:** `adj_rename(adjL)`
- **Type:** Public, exported
- **Purpose:** Normalizes a list of adjacency matrices to have identical row/column names (union of all node names across inputs). Missing nodes get zero-filled rows/columns.
- **Algorithm:**
  1. Collects all unique gene/node names across all matrices (line 19–25)
  2. For each original matrix, creates a new zero-matrix of size `N×N` where N = total unique names (line 34–35)
  3. For each (i,j) pair, copies the weight from the original matrix to the corresponding position in the enlarged matrix using name-based lookup (lines 38–50)
- **Complexity:** O(k × n²) where k = number of matrices, n = unique node count
- **Dependencies:** None (base R only)
- **Side effects:** None
- **Notes:** The name-matching uses `which(rownames(Mat[[z]])==rownames(adjL[[z]])[i])` which relies on exact string match. No deduplication of duplicate node names within a matrix.

---

### 2. `consensusNet`
- **File:** `R/ConsensusINet.R`, lines 34–589
- **Signature:** `consensusNet(adjL, threshold=0.5, tolerance=0.1, theta=0.04, nitermax=50, ncores=2, verbose=TRUE)`
- **Type:** Public, exported — **MAIN ENTRY POINT**
- **Purpose:** Runs the full INet iterative consensus algorithm. See `algorithm_flow.md` for complete step-by-step trace.
- **Input:** List of weighted adjacency matrices (weights in [0,1]), same row/col names
- **Output:** List with `$graphConsensus` (igraph), `$Comparison` (vector of Jaccard distances per iteration), `$similarGraphs` (list of igraph objects before thresholding)
- **Dependencies:** igraph, r2r, parallel, stats, utils
- **Internal closures defined:**
  - `weightMat(Weight, grafo, nodes)` — line 63–75
  - `get_lower_tri_noDiag(cormat)` — line 78–82 (duplicate of the one in InternalFunction.R)
  - `JaccardAll(grafi)` — line 87–130
  - `funNeig(x)` — line 212–214
  - `code(a,b)` — line 218
  - `funWeights(x)` — line 220–223
  - `funegoWeights(x)` — line 227–229
- **Side effects:** Spawns parallel cluster (`parallel::makeCluster`), prints to console if `verbose=TRUE`
- **Key numerical details:**
  - Edge encoding uses Cantor pairing function: `code(a,b) = (a+b)(a+b+1)/2 + b` where a = min, b = max (line 218)
  - Weights clamped to [0,1] at line 446–448
  - Ego-network weight uses `^(1/n)` root power (line 360, 421)
  - Denominator adjustment: subtracts 2 when edge exists, 0 otherwise (lines 352–357, 413–418)
  - Convergence: breaks on distance non-decrease, max iterations, or tolerance reached

---

### 3. `constructionGraph`
- **File:** `R/ConstructionGraph.R`, lines 20–62
- **Signature:** `constructionGraph(data, perc=0.95)`
- **Type:** Public, exported
- **Purpose:** Constructs networks from raw data matrices using Pearson correlation + proportional thresholding.
- **Algorithm:**
  1. For each data layer, computes Pearson correlation matrix (line 31: `stats::cor(data[[i]])`)
  2. Plots histogram of lower-triangle weights (line 37–39, uses ggplot2)
  3. Computes threshold as the `perc`-th quantile of correlation values (line 41: `stats::quantile(CorrM[[i]], perc)`)
  4. Zeros out all correlations below threshold (line 42)
  5. Creates igraph from thresholded matrix with `mode="upper"`, `diag=FALSE`, `weighted=TRUE` (lines 43–46)
  6. Computes Louvain modularity, edge count, vertex count (lines 47–49)
- **Output:** List with `$Threshold` (data frame: threshold value, edge count, node count, modularity), `$Graphs` (list of igraph objects), `$Adj` (list of thresholded correlation matrices)
- **Dependencies:** stats (cor, quantile), igraph, ggplot2, ggpubr
- **Side effects:** Prints a grid of histograms (line 55: `ggpubr::ggarrange`)
- **Notes:** `perc=0.95` means top 5% of weights retained. The function assumes columns represent nodes and rows represent observations (transposes are not done — `cor()` computes column-wise correlation).

---

### 4. `densityNet`
- **File:** `R/DensityWeight.R`, lines 21–78
- **Signature:** `densityNet(graphL)`
- **Type:** Public, exported
- **Purpose:** Computes density distribution of mean edge weights across all graphs in the list. Used to help choose the consensus threshold.
- **Algorithm:**
  1. Converts each graph to adjacency matrix (line 30)
  2. Computes element-wise mean across all matrices via triple nested loop (lines 35–51)
  3. Extracts lower triangle (no diagonal) of the mean matrix (line 54)
  4. Computes quantiles of all values and quantiles of non-zero values (lines 63, 68)
  5. Plots density of non-zero values (lines 69–72)
- **Output:** List with `$quantile` and `$quantileNo0`
- **Dependencies:** igraph, ggplot2, stats (quantile)
- **Side effects:** Prints a ggplot density plot
- **Notes:** Identical matrix-mean logic is duplicated in `thresholdNet()` and the post-algorithm section of `consensusNet()`. This is a code duplication issue.

---

### 5. `JWmatrix`
- **File:** `R/JaccardWeightedMatrix.R`, lines 15–67
- **Signature:** `JWmatrix(graphL)`
- **Type:** Public, exported
- **Purpose:** Computes the pairwise weighted Jaccard distance matrix between all graphs in the list.
- **Algorithm:**
  1. Validates that all graphs have identical node names (lines 19–24, stops if mismatch)
  2. Extracts lower triangle (no diag) of each graph's adjacency matrix as a flattened vector (lines 30–38)
  3. Stacks vectors into matrix A (rows = graphs, columns = edge positions)
  4. Computes weighted Jaccard similarity for each pair: `num = sum(min(a,b) over columns)`, `den = sum(max(a,b) over columns)` (lines 47–51)
  5. Sets `sim.jac[is.na(sim.jac)] = 0` (line 53)
  6. Sets `diag(sim.jac) = 1` (line 54)
  7. Returns `dist.jac = 1 - sim.jac` (line 57)
- **Dependencies:** igraph, utils (combn)
- **Notes:** Diag is set to 1 (self-similarity = 1). This differs from `JWmean` and `JaccardAll` where diag is NA.

---

### 6. `JWmean`
- **File:** `R/JaccardWeightedMean.R`, lines 14–56
- **Signature:** `JWmean(graphL)`
- **Type:** Public, exported
- **Purpose:** Computes the mean weighted Jaccard distance across all graph pairs (scalar summary).
- **Algorithm:** Same weighted Jaccard computation as `JWmatrix` but:
  - `diag(sim.jac) = NA` (line 49) — self-pairs excluded from mean
  - Returns `mean(dist.jac, na.rm=TRUE)` (line 54)
- **Dependencies:** igraph, utils (combn)
- **Notes:** This function duplicates most of `JWmatrix` logic. Only difference: diag handling and return value.

---

### 7. `measuresNet`
- **File:** `R/NetworkMeasures.R`, lines 17–78
- **Signature:** `measuresNet(graphL, nodes.measures=TRUE)`
- **Type:** Public, exported
- **Purpose:** Computes graph-level and optionally node-level network measures for each layer.
- **Graph measures computed** (lines 24–35): vertex count, edge count, global transitivity, diameter, Louvain modularity, edge density, degree assortativity, degree centralization, betweenness centralization
- **Node measures computed** (lines 42–46, if `nodes.measures=TRUE`): degree, local transitivity, betweenness, hub score
- **Dependencies:** igraph
- **Notes:** Modularity uses `cluster_louvain(graphL[[i]], weights=NULL)$membership` — explicitly ignoring edge weights in the Louvain clustering used for modularity calculation (line 28). Hub score also uses `weights=NULL` (line 45).

---

### 8. `plotINet`
- **File:** `R/Plots.R`, lines 34–118
- **Signature:** `plotINet(adj, graph.consensus, edge.width=3, vertex.label.cex=0.5, vertex.size=10, edge.curved=0.2, method="NA", ...)`
- **Type:** Public, exported
- **Purpose:** Plots the union of one original layer and the consensus network with color-coded edges.
- **Edge colors:** Red (`#F8766D`) = edges in both original and consensus (intersection); Light blue (`#619CFF`) = edges only in consensus (difference); Gray = edges only in original
- **Community detection:** Uses the `robin::membershipCommunities` function (line 105). Default `method="NA"` uses a single green color. Supports: walktrap, edgeBetweenness, fastGreedy, louvain, spinglass, leadingEigen, labelProp, infomap, optimal, leiden.
- **Dependencies:** igraph, robin
- **Side effects:** Produces a base R plot (line 114)

---

### 9. `plotL`
- **File:** `R/Plots.R`, lines 138–161
- **Signature:** `plotL(graphL, ...)`
- **Type:** Public, exported
- **Purpose:** Plots all layers in a multi-layer visualization using the `multinet` package.
- **Algorithm:** Assigns numeric vertex names if missing, converts each igraph to a multinet layer, plots all layers together.
- **Dependencies:** multinet, igraph
- **Side effects:** Produces a multi-layer network plot

---

### 10. `plotC`
- **File:** `R/Plots.R`, lines 181–187
- **Signature:** `plotC(graph, ...)`
- **Type:** Public, exported
- **Purpose:** Plots a graph after removing isolated nodes.
- **Dependencies:** igraph
- **Side effects:** Produces a base R plot

---

### 11. `specificNet`
- **File:** `R/SpecificFunction.R`, lines 21–66
- **Signature:** `specificNet(graphL, graph.consensus)`
- **Type:** Public, exported
- **Purpose:** Computes case-specific networks by graph set difference.
- **Algorithm:** For each layer t, computes `igraph::difference(graphL[[t]], graph.consensus)` (line 53). Also computes `percentageOfSpecificity = ecount(diff) / ecount(original)` (line 56).
- **Dependencies:** igraph
- **Output:** List with `$GraphsDifference` and `$percentageOfSpecificity`
- **Notes:** Contains commented-out code (lines 26–44) that is a duplicate of the graph-construction preamble from `consensusNet()`.

---

### 12. `thresholdNet`
- **File:** `R/ThresholdConsensus.R`, lines 20–65
- **Signature:** `thresholdNet(sim.graphL, threshold=0.5)`
- **Type:** Public, exported
- **Purpose:** Reconstructs the consensus network from similar graphs using a different threshold. Avoids re-running the full algorithm.
- **Algorithm:** Converts similar graphs to adjacency matrices, computes element-wise mean, thresholds at `threshold`, creates igraph.
- **Dependencies:** igraph
- **Notes:** Contains the same matrix-mean logic as `densityNet()` and the post-algorithm consensus construction in `consensusNet()` — code duplication. Uses `names=TRUE` in `as_adjacency_matrix` (line 29) unlike `densityNet()`.

---

## Internal Functions — 1 standalone + 7 closures

### `get_lower_tri_noDiag` (standalone)
- **File:** `R/InternalFunction.R`, lines 9–12
- **Signature:** `get_lower_tri_noDiag(cormat)`
- **Type:** Internal (not exported, `@keywords internal`)
- **Purpose:** Sets the upper triangle and diagonal of a matrix to NA, preserving the lower triangle.
- **Used by:** `JWmatrix()`, `JWmean()`, `constructionGraph()`, `densityNet()`, `JaccardAll()` (closure inside `consensusNet()`)
- **Note:** This function is duplicated as a closure inside `consensusNet()` at line 78–82.

### Closures inside `consensusNet()`:

| Closure | Lines | Purpose |
|---------|-------|---------|
| `weightMat(Weight, grafo, nodes)` | 63–75 | Sets/creates an edge with given weight in an igraph |
| `get_lower_tri_noDiag(cormat)` | 78–82 | Duplicate of standalone internal function |
| `JaccardAll(grafi)` | 87–130 | Computes mean Jaccard distance across all graphs in the list |
| `funNeig(x)` | 212–214 | Inserts neighbor vectors into r2r hashmap, increments `node_id` via `<<-` |
| `code(a,b)` | 218 | Cantor pairing function: `(x+y)(x+y+1)/2 + y` for unordered edge encoding |
| `funWeights(x)` | 220–223 | Inserts edge weights into r2r hashmap using `code()` as key |
| `funegoWeights(x)` | 227–229 | Initializes ego-weight hashmap entries to 0 |

### `code(a,b)` — Critical detail
The Cantor pairing function at line 218:
```
code <- function(a,b) {x <- min(a,b); y <- max(a,b); (x+y)*(x+y+1)/2 + y}
```
This produces a unique integer for every unordered pair (a,b). Used as hashmap key for edges. Note: this assumes node IDs are small integers (1-based indices from edgelists). If node IDs are large or non-integer strings, this WILL produce collisions or incorrect results. The edgelist is extracted with `names=FALSE` (line 175), so it uses igraph's internal numeric vertex IDs.

---

## Data Objects (lazy-loaded)

| Object | File | Description |
|--------|------|-------------|
| `exampleL_data` | `data.R:13` | List of 3 datasets: Gene_Expression, Methy_Expression, Mirna_Expression (glioblastoma patients) |
| `graphL_data` | `data.R:24` | List of 2 igraph objects |
| `adjL_data` | `data.R:36` | List of 2 adjacency matrices |
| `tryL_data` | `data.R:49` | List of 2 adjacency matrices with different node names (for testing `adj_rename()`) |

---

## Summary Statistics

- **Total R functions:** 12 exported + 1 internal + 7 closures = 20
- **Lines of R code:** ~700 (excluding roxygen comments)
- **Largest file:** `ConsensusINet.R` (592 lines, ~85% of codebase complexity)
- **Code duplication instances:** 3
  - `get_lower_tri_noDiag` in `InternalFunction.R:9` and `ConsensusINet.R:78`
  - Matrix-mean computation in `densityNet.R:26-51`, `ThresholdConsensus.R:26-51`, `ConsensusINet.R:536-558`
  - Weighted Jaccard computation in `JWmatrix.R:42-52` and `JWmean.R:42-52` and `ConsensusINet.R:114-121`
