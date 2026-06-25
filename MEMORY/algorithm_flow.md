# Algorithm Flow — INet (`consensusNet`)

This document traces the exact step-by-step execution of the INet consensus algorithm as implemented in `R/ConsensusINet.R`. Every line reference is to that file.

---

## Phase 0: Input Processing (lines 42–60)

**Input:** `adjL` — a list of weighted adjacency matrices with weights in [0,1].

For each matrix t = 1..k:
1. If rownames exist: convert to igraph with `graph_from_adjacency_matrix(adjL[[t]], mode="upper", diag=FALSE, add.colnames="NA", weighted=TRUE)` — line 48–52
2. If no rownames: same but without `add.colnames` — line 55–58

Result: `graph` — a list of k igraph objects (undirected, weighted, upper-triangular).

---

## Phase 1: Core Iterative Loop (lines 144–530)

The algorithm uses `repeat` (infinite loop with explicit breaks).

### Step 1.1: Compute initial Jaccard distance (line 147)

```
Comp <- JaccardAll(grafi=graph)
```

`JaccardAll` (lines 87–130):
1. **Validate node names**: For every pair of graphs, checks that vertex names match exactly. Stops with error if names differ (line 94–96).
2. **Extract weights**: For each graph, converts to weighted adjacency matrix, takes lower triangle (no diag), flattens to vector of non-NA values. Stacks as rows of matrix A (lines 99–107).
3. **Compute pairwise weighted Jaccard**: For each pair of rows (graphs) a,b:
   - `num = sum(min(A[a, col], A[b, col]) for each column)` — line 115–116
   - `den = sum(max(A[a, col], A[b, col]) for each column)` — line 117–118
   - `sim.jac[a,b] = num/den` — line 119
   - `sim.jac[b,a] = num/den` — line 120
4. **Post-process**: `sim.jac[is.na(sim.jac)] = 0` (line 122), `diag(sim.jac) = NA` (line 123)
5. **Distance**: `dist.jac = 1 - sim.jac` (line 126)
6. **Return**: `mean(dist.jac, na.rm=TRUE)` (line 128) — the mean distance across all pairs

This is a scalar representing how dissimilar the graphs currently are.

### Step 1.2: Check immediate convergence (lines 150–159)

If `count == 0` and `Comp < tolerance` (default tolerance=0.1):
- Print message and `break` — the graphs are already similar enough, no iteration needed.

### Step 1.3: Save current state (line 164)

```
graphBeginning <- graph
```

An independent copy of current graphs before modification. This is needed because the algorithm only accepts modifications that decrease distance.

### Step 1.4: Build union graph (lines 167–173)

```
unionGraph <- graph[[1]]
for(j in seq(1,length(graph))[-1])
    unionGraph <- igraph::union(unionGraph, graph[[j]])
```

This creates a graph containing all edges from all layers. `igraph::union` with weighted graphs merges edges — the resulting edge set is the set of all edges present in any layer.

### Step 1.5: Extract union edge list (line 175)

```
Edgelist <- igraph::as_edgelist(unionGraph, names=FALSE)
```

Returns an N×2 matrix of vertex ID pairs. `names=FALSE` is critical: it returns igraph's internal integer vertex IDs, not string names. The `code()` pairing function depends on these being small integers.

### Step 1.6: Parallel weight computation (lines 195–460)

A parallel cluster is created with `ncores` workers (default 2). Each worker processes one graph.

For each graph index `i` (1..k):

#### 1.6a: Build hashmaps for all graphs (lines 245–271)

For each graph h in 1..k (note: all graphs' hashmaps are built in each worker — this is redundant but done for data locality):

- **Neighbor hashmap `Neig_list[[h]]`** (`m`): Maps vertex ID → vector of neighbor IDs. Built using `igraph::as_adj_list()` → `lapply(l, funNeig)` → stores in r2r hashmap with incrementing keys 1,2,3,... (lines 249–253).
  - `funNeig(x)` inserts `as.vector(x)` into `m` at key `node_id`, then increments `node_id` via `<<-` (lines 212–214).

- **Weight hashmap `Weights_list[[h]]`** (`s`): Maps edge_code → weight. Edge_code = `code(node1, node2)`. Built from graph's edgelist using `apply(E, 1, funWeights)` (lines 257–261).
  - `funWeights(x)` inserts the edge weight at key `code(E[edge_id,1], E[edge_id,2])` (lines 220–223).

- **Ego-weight hashmap `EgoWeights_list[[h]]`** (`t`): Same keys as weight hashmap, initialized to 0. Built from the UNION edgelist with `apply(E, 1, funegoWeights)` (lines 265–270).
  - This hashmap stores only edges from the union graph (not just the current graph's own edges).

#### 1.6b: For each edge in the union graph (z = 1..N, lines 281–453):

**Get neighbor intersection for graph i:**
```
nodes <- c(Edgelist[z,1], Edgelist[z,2])
Inei <- Neig_list[[i]][nodes[1]][[1]]    # neighbors of node 1 in graph i
Jnei <- Neig_list[[i]][nodes[2]][[1]]    # neighbors of node 2 in graph i
Intersect_list[[i]] <- intersect(Inei, Jnei)   # common neighbors
```
(line 284–287)

**Get edge weight in graph i (wUso):**
```
wUso <- Weights_list[[i]][code(nodes[1], nodes[2])][[1]]
if(is.null(wUso)) { wUso <- 0 }
```
(lines 291–293) — If the edge does not exist in graph i, weight is 0.

**Get edge weights in all other graphs and compute neighbor intersections for them:**
```
wOthers <- NULL
for(j in seq(1,length(graph))[-i]) {
    Intersect_list[[j]] <- intersect(Neig_list[[j]][nodes[1]][[1]],
                                     Neig_list[[j]][nodes[2]][[1]])
    wAltri <- Weights_list[[j]][code(nodes[1], nodes[2])][[1]]
    if(is.null(wAltri)) { wAltri <- 0 }
    wOthers <- c(wOthers, wAltri)
}
```
(lines 299–313)

**Compute base weight (peso):**
```
peso <- (wUso + mean(wOthers)) / 2
```
(line 317)

This averages the edge's own weight and the mean of its weight in all other networks, then divides by 2.

**Compute ego-network weight for graph i (pesiEgoNUS):**

If `length(Intersect_list[[i]]) == 0` (no common neighbors): `pesiEgoNUS = 0` (line 324).

Otherwise, for each common neighbor k:
```
# Get edge weights from graph i: (node1, k) and (node2, k)
pesiint1 <- Weights_list[[i]][code(nodes[1], Intersect_list[[i]][k])][[1]]
pesiint2 <- Weights_list[[i]][code(nodes[2], Intersect_list[[i]][k])][[1]]
# Default to 0 if null

# Count how many layers have this common neighbor in their intersection
numberCom <- table(unlist(Intersect_list))[names(table(unlist(Intersect_list)))==Intersect_list[[i]][k]]

# Contribution from this neighbor
pint1_2[k] <- (numberCom / length(graph)) * (pesiint1 + pesiint2)
```
(lines 329–347)

The `numberCom` factor up-weights common neighbors that appear in many layers' intersections.

Then:
```
Spesiint1_2 <- sum(pesiint1_2)   # total contribution

# Denominator:
if(wUso == 0) {
    denomin <- length(Inei) + length(Jnei)        # edge doesn't exist
} else {
    denomin <- length(Inei) + length(Jnei) - 2    # edge exists, subtract the endpoints themselves
}

pesiEgoNUS <- (Spesiint1_2 / denomin) ^ (1 / length(Intersect_list[[i]]))
```
(lines 349–360)

This is effectively: `(weighted_intensity / total_neighbor_count) ^ (1 / num_common_neighbors)`. The exponent 1/n acts like a geometric mean normalization.

**Compute ego-network weights for other layers (pesiEgoOthers):**

For each other layer j (lines 372–432):

Same logic but uses that layer's neighbor lists and weight hashmaps. The denominator adjustment checks `wOthers[positionVector] == 0` to decide whether to subtract 2 (lines 410–418). 

Then:
```
pesiEgoAltri[j] = (Spesiint1_2 / denomin) ^ (1 / length(Intersect_list[[j]]))
```

**Combine into final ego weight:**
```
pesiEgo <- (pesiEgoNUS + mean(pesiEgoOthers)) / 2
```
(line 435)

**Final edge weight for graph i:**
```
WeightI1 <- peso + (theta * pesiEgo)
if(WeightI1 > 1) { WeightI1 <- 1 }
```
(lines 441–448)

Where `theta` (default 0.04) controls the influence of the neighborhood component.

**Update graph i:**
```
graph[[i]] <- weightMat(Weight=WeightI1, grafo=graph[[i]], nodes=nodes)
```
(line 450–451)

`weightMat` (lines 63–75): If the edge already exists, sets its weight. If not, adds it with `add_edges` then sets weight.

### Step 1.7: Collect parallel results (lines 462–466)

```
graphChange <- Graphs      # new modified graphs
graph <- graphBeginning    # restore original graphs (for comparison)
```

### Step 1.8: Compute post-iteration Jaccard distance (lines 469)

```
CompPost <- JaccardAll(grafi=graphChange)
```

### Step 1.9: Track comparison history (lines 473–480)

First iteration: `Comparison <- Comp` (initial distance). Subsequently: `Comparison <- rbind(Comparison, CompPost)` — builds a vector of distances across iterations.

### Step 1.10: Convergence checks (lines 493–523)

**Check 1: Distance decreased?**
```
if(Comp > CompPost) {
    graph <- graphChange   # accept new weights
} else {
    break   # "Distance doesn't decrease"
}
```
(line 493–502)
If the modification made the graphs LESS similar (distance increased or stayed same), STOP.

**Check 2: Max iterations?**
```
count <- count + 1
if(count > nitermax) { break }   # "Maximum iteration"
```
(line 504–515)

**Check 3: At tolerance?**
```
if(CompPost < tolerance) { break }
```
(line 519–522)

The loop repeats until one of these three conditions triggers.

---

## Phase 2: Consensus Construction (lines 534–569)

After the iterative loop converges:

1. Convert each graph to adjacency matrix (lines 536–539)
2. Compute element-wise mean across all matrices via triple-nested loop (lines 541–558):
   ```
   for i in 1:N, for j in 1:N:
       matrixMean[i,j] = mean(Mat[[1]][i,j], Mat[[2]][i,j], ..., Mat[[k]][i,j])
   ```
3. Apply threshold: `matrix[matrix < threshold] <- 0` (line 560, default threshold=0.5)
4. Reassign node names if available (lines 563–567)
5. Create final consensus graph (lines 568–569): `igraph::graph_from_adjacency_matrix(matrix, mode="upper", diag=FALSE, weighted=TRUE)`

---

## Phase 3: Output Assembly (lines 575–588)

1. Reassign names to similar graphs (lines 575–581)
2. Return list:
   - `$graphConsensus`: the consensus network (igraph)
   - `$Comparison`: vector of Jaccard distances (initial + one per completed iteration)
   - `$similarGraphs`: the graphs from the last accepted iteration (before thresholding)

---

## Algorithm Pseudo-Code Summary

```
function consensusNet(adjL, threshold, tolerance, theta, nitermax, ncores):
    graphs = convert_to_igraph(adjL)
    
    loop:
        d_current = mean_weighted_jaccard_distance(graphs)
        if first_iteration and d_current < tolerance: break
        
        graphs_backup = graphs
        union_graph = union_of_all(graphs)
        edgelist = extract_edges(union_graph)
        
        for each graph g in parallel:
            for each graph h:
                build neighbor_hashmap(h)
                build weight_hashmap(h)
                build ego_hashmap(h, union_edgelist)
            
            for each edge (u,v) in edgelist:
                w_own = weight(g, u, v) or 0
                w_others = [weight(h, u, v) for h != g]
                
                base_weight = (w_own + mean(w_others)) / 2
                
                ego_own = compute_ego_weight(g, u, v, all_graphs)
                ego_others = [compute_ego_weight(h, u, v, all_graphs) for h != g]
                ego_weight = (ego_own + mean(ego_others)) / 2
                
                new_weight = min(base_weight + theta * ego_weight, 1.0)
                update_edge(g, u, v, new_weight)
        
        g_new = parallel_results
        d_new = mean_weighted_jaccard_distance(g_new)
        
        if d_new < d_current:
            graphs = g_new
        else:
            break
        
        if iterations > nitermax: break
        if d_new < tolerance: break
    
    consensus = elementwise_mean(graphs_as_matrices)
    consensus[consensus < threshold] = 0
    return consensus_graph
```

---

## Parameters & Their Effects

| Parameter | Default | Meaning | Effect |
|-----------|---------|---------|--------|
| `threshold` | 0.5 | Final binarization cutoff | Higher → sparser consensus |
| `tolerance` | 0.1 | Jaccard distance to stop | Lower → more iterations needed |
| `theta` | 0.04 | Neighborhood influence weight | Higher → topology matters more |
| `nitermax` | 50 | Max iterations | Safety cap |
| `ncores` | 2 | Parallel workers | More workers = faster per iteration |
| `verbose` | TRUE | Console output | Prints distance per iteration |
