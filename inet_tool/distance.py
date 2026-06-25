"""
Weighted Jaccard distance functions for multi-layer networks.
Translated from R package INetTool.

Source files:
  - R/JaccardWeightedMatrix.R:15-67  (JWmatrix)
  - R/JaccardWeightedMean.R:14-56    (JWmean)
  - R/ConsensusINet.R:87-130         (JaccardAll closure)
"""

import numpy as np
from itertools import combinations

import networkx as nx


# ---------------------------------------------------------------------------
# Shared helper: flatten graphs to matrix A
# ---------------------------------------------------------------------------

def _graphs_to_matrix_A(graphs):
    """
    Convert a list of NetworkX graphs to a stacked matrix A where each row
    is the lower-triangle (no diagonal) edge weights of one graph.

    This corresponds to lines 99-107 of consensusNet and lines 30-38 of
    JWmatrix / JWmean:
      AdjW <- igraph::as_adjacency_matrix(graph, attr="weight")
      triA <- get_lower_tri_noDiag(AdjW)
      vettriA <- as.vector(triA)
      vetI <- vettriA[!is.na(vettriA)]
      A <- rbind(A, vetI)

    Parameters
    ----------
    graphs : list of networkx.Graph
        All graphs must have the same number of vertices and the same
        vertex set (identical vertex ordering is assumed).

    Returns
    -------
    A : np.ndarray, shape (K, M)
        K = number of graphs, M = N*(N-1)/2 lower-triangle positions.
    names_valid : bool
        True if all graphs have consistent vertex names.
    """
    from ._internal import get_lower_tri_noDiag

    K = len(graphs)
    N = graphs[0].number_of_nodes()
    M = N * (N - 1) // 2

    A = np.zeros((K, M), dtype=np.float64)

    for idx, g in enumerate(graphs):
        adj = nx.to_numpy_array(g, weight="weight", dtype=np.float64)
        adj_copy = adj.copy()
        adj_copy = get_lower_tri_noDiag(adj_copy)
        lower_idx = np.tril_indices(N, k=-1)
        A[idx, :] = adj_copy[lower_idx]

    return A


# ---------------------------------------------------------------------------
# Node name validation (R equivalent logic)
# ---------------------------------------------------------------------------

def _validate_same_nodes(graphs, err_msg="Not same nodes in all the graphs"):
    """
    Validate that all graphs have the same vertex names.
    Stops with an error if any pair differs.

    Original R (JWmatrix.R:19-24):
      comp <- utils::combn(1:length(grafi), 2)
      for (j in 1:(dim(comp)[2])) {
        if(names(table(V(grafi[[comp[1,j]]])==V(grafi[[comp[2,j]]])))=="TRUE")
        {}else{stop("Check:Not same nodes in all the graphs")}
      }
    """
    ref_nodes = sorted(graphs[0].nodes())
    for i, g in enumerate(graphs[1:], start=1):
        if sorted(g.nodes()) != ref_nodes:
            raise ValueError(
                f"Check: {err_msg} (graph 0 vs graph {i} have different nodes). "
                "Add missing nodes as isolated vertices."
            )


# ---------------------------------------------------------------------------
# Weighted Jaccard similarity / distance core
# ---------------------------------------------------------------------------

def _weighted_jaccard_sim(A: np.ndarray) -> np.ndarray:
    """
    Compute pairwise weighted Jaccard similarity matrix from stacked
    lower-triangle weight vectors.

    Original R (JWmatrix.R:47-52):
      num <- sum(sapply(1:ncol(A), function(x)(min(A[pairs[i,1],x], A[pairs[i,2],x]))))
      den <- sum(sapply(1:ncol(A), function(x)(max(A[pairs[i,1],x], A[pairs[i,2],x]))))
      sim.jac[pairs[i,1],pairs[i,2]] <- num/den
      sim.jac[pairs[i,2],pairs[i,1]] <- num/den

    Parameters
    ----------
    A : np.ndarray, shape (K, M)
        Stacked weight vectors (one row per graph).

    Returns
    -------
    sim_jac : np.ndarray, shape (K, K)
        Pairwise weighted Jaccard similarity. Entries may be NaN where
        both graphs have all-zero weight vectors (0/0).
    """
    K = A.shape[0]
    sim_jac = np.zeros((K, K), dtype=np.float64)

    for a, b in combinations(range(K), 2):
        row_a = A[a, :]
        row_b = A[b, :]
        mins = np.minimum(row_a, row_b)
        maxs = np.maximum(row_a, row_b)
        den = maxs.sum()
        if den == 0.0:
            sim_jac[a, b] = np.nan
            sim_jac[b, a] = np.nan
        else:
            val = mins.sum() / den
            sim_jac[a, b] = val
            sim_jac[b, a] = val

    return sim_jac


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def jaccard_all(graphs):
    """
    Compute the mean weighted Jaccard distance across all graph pairs.

    Direct translation of the JaccardAll closure:
      R/ConsensusINet.R:87-130

    Original R steps:
      1. Validate vertex names (stop on mismatch)
      2. Extract lower-triangle weights into matrix A
      3. Compute pairwise weighted Jaccard similarity
         sim.jac[is.na(sim.jac)] <- 0
         diag(sim.jac) <- NA
      4. dist.jac <- 1 - sim.jac
      5. return(mean(dist.jac, na.rm=TRUE))

    Parameters
    ----------
    graphs : list of networkx.Graph
        All graphs must share the same vertex set.

    Returns
    -------
    float
        Mean weighted Jaccard distance across all non-diagonal pairs.
        Returns 1.0 if all graph pairs are maximally dissimilar (0/0 edges).
    """
    _validate_same_nodes(graphs, "Not same nodes in all the graphs. Add them as isolated nodes")

    K = len(graphs)
    if K <= 1:
        return 0.0

    A = _graphs_to_matrix_A(graphs)

    sim_jac = _weighted_jaccard_sim(A)

    sim_jac = np.nan_to_num(sim_jac, nan=0.0)      # line 122: sim.jac[is.na(sim.jac)] <- 0
    np.fill_diagonal(sim_jac, np.nan)               # line 123: diag(sim.jac) <- NA

    dist_jac = 1.0 - sim_jac                        # line 126
    return float(np.nanmean(dist_jac))               # line 128


def jw_matrix(graphs):
    """
    Compute the pairwise weighted Jaccard distance matrix.

    Direct translation of:
      R/JaccardWeightedMatrix.R:15-67

    Differs from jaccard_all in two ways:
      - diag(sim.jac) is set to 1 (self-distance = 0) instead of NA
      - Returns the full KxK distance matrix, not the mean

    Parameters
    ----------
    graphs : list of networkx.Graph

    Returns
    -------
    np.ndarray, shape (K, K)
        Pairwise weighted Jaccard distance matrix.
        dist[i,i] = 0, dist[i,j] = dist[j,i] in [0, 1].
    """
    _validate_same_nodes(graphs, "Not same nodes in all the graphs")

    K = len(graphs)
    A = _graphs_to_matrix_A(graphs)

    sim_jac = _weighted_jaccard_sim(A)

    sim_jac = np.nan_to_num(sim_jac, nan=0.0)      # line 53
    np.fill_diagonal(sim_jac, 1.0)                  # line 54: diag(sim.jac) <- 1

    dist_jac = 1.0 - sim_jac                        # line 57

    return dist_jac


def jw_mean(graphs):
    """
    Compute the mean weighted Jaccard distance for a multi-layer network.

    Direct translation of:
      R/JaccardWeightedMean.R:14-56

    Identical logic to jaccard_all but lives in its own source file.

    Parameters
    ----------
    graphs : list of networkx.Graph

    Returns
    -------
    float
        Mean weighted Jaccard distance across all non-diagonal pairs.
    """
    _validate_same_nodes(graphs, "Not same nodes in all the graphs")

    K = len(graphs)
    if K <= 1:
        return 0.0

    A = _graphs_to_matrix_A(graphs)

    sim_jac = _weighted_jaccard_sim(A)

    sim_jac = np.nan_to_num(sim_jac, nan=0.0)      # line 48
    np.fill_diagonal(sim_jac, np.nan)               # line 49: diag(sim.jac) <- NA

    dist_jac = 1.0 - sim_jac                        # line 52
    return float(np.nanmean(dist_jac))               # line 54
