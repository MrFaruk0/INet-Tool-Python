"""
Visualization functions for INet-Tool.
Translated from R package INetTool.

Source files:
  - R/Plots.R:34-118   (plotINet)
  - R/Plots.R:138-161  (plotL)
  - R/Plots.R:181-187  (plotC)
"""

import networkx as nx


def plot_inet(
    adj,
    graph_consensus,
    edge_width=3,
    vertex_size=10,
    edge_curved=0.2,
    method="NA",
    vertex_label_size=10,
    **kwargs,
):
    """
    Plot the union of one original layer and the consensus network with
    color-coded edges.

    Direct translation of:
      R/Plots.R:34-118

    Edge colors:
      - Red (#F8766D) : edges in both original and consensus (intersection)
      - Light blue (#619CFF) : edges only in consensus (difference)
      - Gray (#808080) : edges only in original

    Parameters
    ----------
    adj : np.ndarray or pandas.DataFrame
        One of the beginning adjacency matrices.
    graph_consensus : networkx.Graph
        Consensus network from consensus_net.
    edge_width : float
        Edge width (default 3).
    vertex_size : float
        Vertex size (default 10).
    edge_curved : float
        Edge curvature (default 0.2). Unused in matplotlib fallback.
    method : str
        Community detection method. "NA" means no community coloring.
        Supported: "louvain", "leiden", "label_propagation", "greedy_modularity".
    vertex_label_size : float
        Font size for vertex labels.

    Returns
    -------
    None (displays a plot).
    """
    import numpy as np
    import pandas as pd

    if isinstance(adj, pd.DataFrame):
        mat = adj.values.copy()
    else:
        mat = np.array(adj, dtype=np.float64)

    N = mat.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            w = float(mat[i, j])
            if w != 0.0:
                graph.add_edge(i, j, weight=w)

    cons_has_names = False
    try:
        if graph_consensus.number_of_nodes() > 0:
            cons_has_names = "name" in graph_consensus.nodes[0]
    except Exception:
        pass

    if cons_has_names:
        pass
    else:
        for v in graph_consensus.nodes():
            graph_consensus.nodes[v]["name"] = str(v)

    graph_has_names = "name" in graph.nodes[0] if graph.number_of_nodes() > 0 else False
    if not graph_has_names:
        for v in graph.nodes():
            graph.nodes[v]["name"] = str(v)

    cons_names = set(graph_consensus.nodes())
    graph_names = set(graph.nodes())
    to_add = cons_names - graph_names
    if to_add:
        graph.add_nodes_from(to_add)
        for v in to_add:
            if "name" not in graph.nodes[v]:
                graph.nodes[v]["name"] = cons_name_map.get(v, str(v))

    union_g = nx.compose(graph, graph_consensus)

    graph_edges = set()
    for u, v in graph.edges():
        graph_edges.add((min(u, v), max(u, v)))
    consensus_edges = set()
    for u, v in graph_consensus.edges():
        consensus_edges.add((min(u, v), max(u, v)))

    diff_edges = consensus_edges - graph_edges
    inter_edges = consensus_edges & graph_edges

    ecol = []
    for u, v in union_g.edges():
        edge = (min(u, v), max(u, v))
        if edge in diff_edges:
            ecol.append("#619CFF")
        elif edge in inter_edges:
            ecol.append("#F8766D")
        else:
            ecol.append("gray80")

    members = "#00BA38"
    if method != "NA":
        try:
            from networkx.algorithms.community import (
                louvain_communities,
                label_propagation_communities,
                greedy_modularity_communities,
            )
            if method == "louvain":
                communities = louvain_communities(graph, weight="weight", seed=42)
            elif method == "leiden":
                communities = louvain_communities(graph, weight="weight", seed=42)
            elif method == "label_propagation":
                communities = label_propagation_communities(graph)
            elif method == "greedy_modularity":
                communities = greedy_modularity_communities(graph, weight="weight")
            else:
                communities = []

            if communities:
                membership = {}
                for cid, comm in enumerate(communities):
                    for node in comm:
                        membership[node] = cid
                members = [membership.get(n, 0) for n in union_g.nodes()]
        except ImportError:
            pass

    deg = dict(union_g.degree())
    isolated = [v for v, d in deg.items() if d == 0]
    union_g_clean = union_g.copy()
    union_g_clean.remove_nodes_from(isolated)

    try:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(union_g_clean, seed=42)

        plt.figure(figsize=(10, 8))
        if isinstance(members, str):
            nx.draw_networkx_nodes(
                union_g_clean, pos,
                node_color=members,
                node_size=vertex_size,
            )
        else:
            nx.draw_networkx_nodes(
                union_g_clean, pos,
                node_color=members,
                node_size=vertex_size,
                cmap=plt.cm.tab10,
            )
        nx.draw_networkx_edges(
            union_g_clean, pos,
            edge_color=ecol[:union_g_clean.number_of_edges()],
            width=edge_width,
            connectionstyle=f"arc3,rad={edge_curved}",
        )

        labels = {}
        for v in union_g_clean.nodes():
            name = union_g_clean.nodes[v].get("name", str(v))
            labels[v] = name
        nx.draw_networkx_labels(union_g_clean, pos, labels, font_size=vertex_label_size)

        plt.axis("off")
        plt.show()
    except ImportError:
        import igraph as ig
        g_ig = _nx_to_igraph(union_g_clean)
        ig.plot(g_ig)


def plot_l(graphL, **kwargs):
    """
    Plot all layers in separate subplots.

    Direct translation of:
      R/Plots.R:138-161

    Original used multinet R package. Python equivalent displays
    each layer in its own subplot.

    Parameters
    ----------
    graphL : list of networkx.Graph
    """
    import matplotlib.pyplot as plt

    K = len(graphL)
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 5))
    if K == 1:
        axes = [axes]

    for ax, g in zip(axes, graphL):
        g_clean = g.copy()
        deg = dict(g_clean.degree())
        isolated = [v for v, d in deg.items() if d == 0]
        g_clean.remove_nodes_from(isolated)

        pos = nx.spring_layout(g_clean, seed=42)
        nx.draw_networkx_nodes(g_clean, pos, node_size=30, ax=ax)
        nx.draw_networkx_edges(g_clean, pos, alpha=0.5, ax=ax)
        ax.set_title(f"Layer {graphL.index(g) + 1}")
        ax.axis("off")

    plt.show()


def plot_c(graph, **kwargs):
    """
    Plot a graph without isolated nodes.

    Direct translation of:
      R/Plots.R:181-187

    Parameters
    ----------
    graph : networkx.Graph
    """
    deg = dict(graph.degree())
    isolated = [v for v, d in deg.items() if d == 0]
    g_clean = graph.copy()
    g_clean.remove_nodes_from(isolated)

    try:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(g_clean, seed=42)
        nx.draw_networkx_nodes(g_clean, pos, node_size=30)
        nx.draw_networkx_edges(g_clean, pos, alpha=0.5)
        labels = {v: g_clean.nodes[v].get("name", str(v)) for v in g_clean.nodes()}
        if labels:
            nx.draw_networkx_labels(g_clean, pos, labels, font_size=8)
        plt.axis("off")
        plt.show()
    except ImportError:
        pass


def _nx_to_igraph(g):
    """Convert networkx Graph to igraph Graph (internal helper)."""
    import igraph as ig
    mapping = {n: i for i, n in enumerate(g.nodes())}
    ig_g = ig.Graph()
    ig_g.add_vertices(len(mapping))
    for u, v, data in g.edges(data=True):
        ig_g.add_edge(mapping[u], mapping[v], weight=data.get("weight", 1.0))
    for n in g.nodes():
        ig_g.vs[mapping[n]]["name"] = g.nodes[n].get("name", str(n))
    return ig_g
