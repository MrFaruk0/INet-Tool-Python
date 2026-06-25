"""
Post-processing functions for INet-Tool.
Translated from R package INetTool.

Source files:
  - R/DensityWeight.R:21-78       (densityNet)
  - R/ThresholdConsensus.R:20-65  (thresholdNet)
  - R/SpecificFunction.R:21-66    (specificNet)
"""

import numpy as np
import networkx as nx

from ._internal import get_lower_tri_noDiag


def density_net(graphL):
    """
    Compute density distribution of mean weights across graphs.
    Useful for choosing the consensus threshold.

    Direct translation of:
      R/DensityWeight.R:21-78

    Parameters
    ----------
    graphL : list of networkx.Graph

    Returns
    -------
    dict
        "quantile" : np.ndarray
            Quantiles at steps of 0.05 for all mean weight values.
        "quantileNo0" : np.ndarray
            Quantiles excluding zero entries.
    """
    mats = []
    for g in graphL:
        adj = nx.to_numpy_array(g, weight="weight", dtype=np.float64)
        mats.append(adj)

    matrix_mean = np.mean(np.stack(mats, axis=0), axis=0)

    tri = get_lower_tri_noDiag(matrix_mean.copy())
    lower_idx = np.tril_indices_from(matrix_mean, k=-1)
    vect = tri[lower_idx]
    vect = vect[~np.isnan(vect)]

    probs = np.arange(0.0, 1.01, 0.05)
    quant = np.quantile(vect, probs, method="linear")

    vect0 = vect[vect > 0]
    if len(vect0) > 0:
        quant0 = np.quantile(vect0, probs, method="linear")
    else:
        quant0 = np.full_like(probs, np.nan, dtype=np.float64)

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(vect0, bins=30, density=True, color="#69b3a2", edgecolor="#e9ecef")
        plt.xlabel("mean weights without 0")
        plt.show()
    except ImportError:
        pass

    return {
        "quantile": quant,
        "quantileNo0": quant0,
    }


def threshold_net(sim_graphL, threshold=0.5):
    """
    Reconstruct the consensus network from similar graphs using a different
    threshold, without re-running the full algorithm.

    Direct translation of:
      R/ThresholdConsensus.R:20-65

    Parameters
    ----------
    sim_graphL : list of networkx.Graph
        Output of consensus_net's $similarGraphs.
    threshold : float
        New consensus threshold (default 0.5).

    Returns
    -------
    networkx.Graph
    """
    mats = []
    for g in sim_graphL:
        adj = nx.to_numpy_array(g, weight="weight", dtype=np.float64)
        mats.append(adj)

    matrix_mean = np.mean(np.stack(mats, axis=0), axis=0)
    matrix_mean[matrix_mean < threshold] = 0.0

    N = matrix_mean.shape[0]
    g_out = nx.Graph()
    g_out.add_nodes_from(range(N))

    if "name" in sim_graphL[0].nodes[0] if sim_graphL[0].number_of_nodes() > 0 else False:
        try:
            names = [sim_graphL[0].nodes[v].get("name", str(v)) for v in range(N)]
            nx.set_node_attributes(g_out, dict(enumerate(names)), "name")
        except Exception:
            pass

    for i in range(N):
        for j in range(i + 1, N):
            w = float(matrix_mean[i, j])
            if w != 0.0:
                g_out.add_edge(i, j, weight=w)

    return g_out


def specific_net(graphL, graph_consensus):
    """
    Create Case-Specific Networks — edges present in each layer
    but NOT in the consensus.

    Direct translation of:
      R/SpecificFunction.R:21-66

    Parameters
    ----------
    graphL : list of networkx.Graph
        Original layer graphs.
    graph_consensus : networkx.Graph
        Consensus network from consensus_net.

    Returns
    -------
    dict
        "GraphsDifference" : list of networkx.Graph
        "percentageOfSpecificity" : list of float
    """
    GraphsDifference = []
    percentageOfSpecificity = []

    consensus_edges = set()
    for u, v in graph_consensus.edges():
        consensus_edges.add((min(u, v), max(u, v)))

    for g in graphL:
        diff_g = nx.Graph()
        diff_g.add_nodes_from(g.nodes(data=True))
        for u, v in g.edges():
            edge = (min(u, v), max(u, v))
            if edge not in consensus_edges:
                w = g[u][v].get("weight", 0.0)
                diff_g.add_edge(u, v, weight=w)
        GraphsDifference.append(diff_g)

        if g.number_of_edges() > 0:
            pct = diff_g.number_of_edges() / g.number_of_edges()
        else:
            pct = 0.0
        percentageOfSpecificity.append(pct)

    return {
        "GraphsDifference": GraphsDifference,
        "percentageOfSpecificity": percentageOfSpecificity,
    }
