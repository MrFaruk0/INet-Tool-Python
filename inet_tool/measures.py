"""
Network measures for multi-layer graphs.
Translated from R package INetTool.

Source file:
  R/NetworkMeasures.R:17-78  (measuresNet)
"""

import numpy as np
import networkx as nx


def measures_net(graphL, nodes_measures=True):
    """
    Compute graph-level and optionally node-level network measures
    for each layer.

    Direct translation of:
      R/NetworkMeasures.R:17-78

    Graph measures (lines 24-35):
      - vertex count (vcount)
      - edge count (ecount)
      - global transitivity (transitivity type="global")
      - diameter
      - Louvain modularity
      - edge density
      - degree assortativity
      - degree centralization (Freeman)
      - betweenness centralization (Freeman)

    Node measures (lines 42-46, if nodes.measures=TRUE):
      - degree
      - local transitivity (type="local")
      - betweenness (shortest-path)
      - hub score (HITS)

    Parameters
    ----------
    graphL : list of networkx.Graph
    nodes_measures : bool
        If True, compute per-node measures (default True).

    Returns
    -------
    list of dict
        Each entry: {"graphsMeasures": np.ndarray, "nodeMeasures": np.ndarray}
    """
    MeasuresGraphs = []

    for g in graphL:
        v = g.number_of_nodes()
        e = g.number_of_edges()

        tran = nx.transitivity(g)

        try:
            diam = nx.diameter(g)
        except nx.NetworkXError:
            diam = np.nan

        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(g, weight=None, seed=42)
        except ImportError:
            communities = []
        if communities:
            membership = [0] * v
            for cid, comm in enumerate(communities):
                for node in comm:
                    membership[node] = cid
            try:
                modl = nx.community.modularity(g, communities, weight=None)
            except ZeroDivisionError:
                modl = 0.0
        else:
            modl = np.nan

        den_val = nx.density(g)

        try:
            assort = nx.degree_assortativity_coefficient(g)
        except Exception:
            assort = np.nan

        deg_vals = dict(g.degree())
        deg_list = list(deg_vals.values())
        max_deg = max(deg_list) if deg_list else 0
        if v > 2:
            ceD = sum(max_deg - d for d in deg_list) / ((v - 1) * (v - 2))
        else:
            ceD = 0.0

        try:
            bet = nx.betweenness_centrality(g, weight="weight", normalized=True)
            bet_vals = list(bet.values())
            max_bet = max(bet_vals) if bet_vals else 0.0
            if v > 2:
                denom_bet = (v - 1) * (v - 2) * (v - 1)
                if denom_bet > 0:
                    ceB = 2.0 * sum(max_bet - b for b in bet_vals) / denom_bet
                else:
                    ceB = 0.0
            else:
                ceB = 0.0
        except Exception:
            ceB = np.nan

        graphsMeasures = np.array(
            [[v], [e], [tran], [diam], [modl], [den_val], [assort], [ceD], [ceB]],
            dtype=np.float64,
        )

        if nodes_measures:
            d = np.array(list(deg_vals.values()), dtype=np.float64)

            tranloc_list = []
            for node in range(v):
                neighbors = list(g.neighbors(node))
                if len(neighbors) < 2:
                    tranloc_list.append(0.0)
                else:
                    subg = g.subgraph(neighbors)
                    n_edges = subg.number_of_edges()
                    n_possible = len(neighbors) * (len(neighbors) - 1) // 2
                    tranloc_list.append(n_edges / n_possible if n_possible > 0 else 0.0)
            tranloc = np.array(tranloc_list, dtype=np.float64)

            bet_arr = np.array(list(bet.values()), dtype=np.float64)

            try:
                h, a = nx.hits(g, max_iter=100, tol=1e-8)
                hub = np.array([h.get(i, 0.0) for i in range(v)], dtype=np.float64)
            except Exception:
                hub = np.full(v, np.nan, dtype=np.float64)

            NodeMeasures = np.column_stack([d, tranloc, bet_arr, hub])

            MeasuresGraphs.append({
                "graphsMeasures": graphsMeasures,
                "nodeMeasures": NodeMeasures,
            })
        else:
            MeasuresGraphs.append({
                "graphsMeasures": graphsMeasures,
            })

    return MeasuresGraphs
