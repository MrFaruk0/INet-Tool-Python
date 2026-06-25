"""
INet-Tool internal helper functions.
Translated from R package INetTool.

Source files:
  - R/InternalFunction.R:9-12  (get_lower_tri_noDiag)
  - R/ConsensusINet.R:218       (code Cantor pairing)
  - R/ConsensusINet.R:63-75     (weightMat closure)
"""

import numpy as np


def get_lower_tri_noDiag(cormat: np.ndarray) -> np.ndarray:
    """
    Set upper triangle and diagonal of a square matrix to NaN.
    Preserves the strict lower triangle.

    Direct translation of:
      R/InternalFunction.R:9-12
      R/ConsensusINet.R:78-82 (duplicate closure)

    Original R:
      get_lower_tri_noDiag <- function(cormat){
        cormat[upper.tri(cormat)] <- NA
        diag(cormat) <- NA
        return(cormat)
      }

    Parameters
    ----------
    cormat : np.ndarray, shape (N, N)
        Input square matrix.

    Returns
    -------
    np.ndarray, shape (N, N)
        Matrix with upper triangle (row < col) and diagonal set to NaN.
        Lower triangle values preserved.
        Modified IN PLACE and returned.
    """
    cormat[np.triu_indices_from(cormat, k=0)] = np.nan
    return cormat


def code(a: int, b: int) -> int:
    """
    Cantor pairing function for unordered pairs.
    Produces a unique integer for every unordered pair (a, b).

    Direct translation of:
      R/ConsensusINet.R:218
      code <- function(a,b) {x <- min(a,b); y <- max(a,b); (x+y)*(x+y+1)/2 + y}

    Used as a hashmap key for edge lookups (combines source/target vertex IDs).

    Parameters
    ----------
    a, b : int
        Vertex indices (0-based in Python, 1-based in R).
        Order does not matter.

    Returns
    -------
    int
        Unique integer key for the unordered pair (a, b).
    """
    x, y = (a, b) if a < b else (b, a)
    return (x + y) * (x + y + 1) // 2 + y


def weight_mat(
    weight: float,
    graph: "nx.Graph",
    nodes: tuple[int, int],
) -> "nx.Graph":
    """
    Set or create an edge in a NetworkX graph with the given weight.

    Direct translation of weightMat closure:
      R/ConsensusINet.R:63-75

    Original R:
      weightMat <- function(Weight, grafo, nodes) {
        edgID <- igraph::get.edge.ids(grafo, nodes)
        if (edgID == 0) {
          grafo <- igraph::add_edges(grafo, nodes)
          edgID <- igraph::get.edge.ids(grafo, nodes)
        }
        igraph::E(grafo)$weight[edgID] <- Weight
        return(grafo)
      }

    Parameters
    ----------
    weight : float
        Edge weight to assign.
    graph : networkx.Graph
        The graph to modify IN PLACE.
    nodes : tuple of (int, int)
        Source and target vertex indices of the edge.

    Returns
    -------
    networkx.Graph
        The same graph object (modified in place).
    """
    u, v = nodes
    graph.add_edge(u, v, weight=weight)
    return graph
