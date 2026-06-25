"""
INet consensus algorithm — the core iterative multi-layer network integration.
Translated from R package INetTool.

Source file:
  R/ConsensusINet.R:34-589  (consensusNet)
"""

import numpy as np
import networkx as nx
from collections import Counter

from ._internal import get_lower_tri_noDiag, code, weight_mat
from .distance import jaccard_all


def _adj_to_graphs(adjL):
    """
    Convert a list of adjacency matrices (pandas DataFrames or numpy arrays)
    to NetworkX graphs.

    Direct translation of lines 42-60 of ConsensusINet.R:

      graph <- vector(mode = "list", length = length(adjL))
      for (t in 1:length(adjL)) {
        if(length(rownames(adjL[[1]]))>0) {
          graph[[t]] <- igraph::graph_from_adjacency_matrix(adjL[[t]],
                            mode = "upper", diag = FALSE,
                            add.colnames = "NA", weighted = TRUE)
        } else {
          graph[[t]] <- igraph::graph_from_adjacency_matrix(adjL[[t]],
                            mode = "upper", diag = FALSE, weighted = TRUE)
        }
      }

    Parameters
    ----------
    adjL : list of np.ndarray or pandas.DataFrame

    Returns
    -------
    list of networkx.Graph
    """
    import pandas as pd

    graphs = []
    has_names = False
    if len(adjL) > 0:
        if isinstance(adjL[0], pd.DataFrame):
            has_names = len(adjL[0].index) > 0

    for adj in adjL:
        if isinstance(adj, pd.DataFrame):
            mat = adj.values.copy()
            names = list(adj.index)
        else:
            mat = adj.copy()
            names = None

        mat = np.array(mat, dtype=np.float64)
        N = mat.shape[0]
        g = nx.Graph()
        g.add_nodes_from(range(N))
        if names is not None:
            nx.set_node_attributes(g, dict(enumerate(names)), "name")

        for i in range(N):
            for j in range(i + 1, N):
                w = float(mat[i, j])
                if w != 0.0:
                    g.add_edge(i, j, weight=w)

        graphs.append(g)

    return graphs


def _graphs_to_matrices(graphs):
    """
    Convert a list of NetworkX graphs to numpy adjacency matrices.

    Direct translation of lines 534-539:
      Mat <- vector(mode = "list", length = length(graph))
      for (z in 1:length(graph)) {
        Mat[[z]] <- as.matrix(igraph::as_adjacency_matrix(graph[[z]], attr="weight"))
      }

    Parameters
    ----------
    graphs : list of networkx.Graph

    Returns
    -------
    list of np.ndarray, each shape (N, N)
    """
    mats = []
    for g in graphs:
        adj = nx.to_numpy_array(g, weight="weight", dtype=np.float64)
        mats.append(adj)
    return mats


def _build_hashes(graphs, union_edges):
    """
    Build neighbor sets and weight dictionaries for all graphs.

    Direct translation of hashmap construction loop,
    lines 245-271 of ConsensusINet.R:

      for (h in 1:length(graph)) {
        # Hashmap neighbors: m <- r2r::hashmap()
        #   l <- igraph::as_adj_list(graph[[h]])
        #   lapply(l, funNeig)
        #   Neig_list[[h]] <- m
        #
        # Hashmap weights: s <- r2r::hashmap()
        #   E <- igraph::as_edgelist(graph[[h]])
        #   apply(E, 1, funWeights)
        #   Weights_list[[h]] <- s
        #
        # Hashmap egoWeights: t <- r2r::hashmap()
        #   E <- Edgelist
        #   apply(E, 1, funegoWeights)
        #   EgoWeights_list[[h]] <- t
      }

    NOTE: EgoWeights_list (the "t" hashmap) is built but NEVER READ in
    the R code (insertions at lines 366 and 429 are commented out).
    We omit it here.

    Parameters
    ----------
    graphs : list of networkx.Graph
    union_edges : list of (int, int)

    Returns
    -------
    neig_list : list of list of set
        neig_list[h][v] = set of neighbors of vertex v in graph h.
    weights_list : list of dict
        weights_list[h][code(u,v)] = edge weight in graph h.
    """
    K = len(graphs)

    neig_list = []
    weights_list = []

    for h in range(K):
        g = graphs[h]
        N = g.number_of_nodes()

        neighbors_of = [set(g.neighbors(v)) for v in range(N)]
        neig_list.append(neighbors_of)

        wmap = {}
        for u, v, data in g.edges(data="weight", default=0.0):
            key = code(u, v)
            wmap[key] = float(data)
        weights_list.append(wmap)

    return neig_list, weights_list


def _compute_ego_weight(
    i_layer,
    u, v,
    w_uso,
    neig_list,
    weights_list,
    intersect_list,
    K,
):
    """
    Compute the ego-network weight component for one layer.

    Direct translation of ego-weight computation,
    lines 321-362 of ConsensusINet.R (for the "NUS" case)
    and lines 376-421 (for the "others" case, parameterized by layer j).

    Parameters
    ----------
    i_layer : int
        Index of the layer being computed.
    u, v : int
        Edge endpoints.
    w_uso : float
        Edge weight in layer i_layer (0 if edge absent).
    neig_list : list of list of set
        neig_list[h][v] = neighbors of v in graph h.
    weights_list : list of dict
        weights_list[h][code(a,b)] = weight.
    intersect_list : list of set or None
        intersect_list[h] = common neighbors of u,v in graph h.
        For i_layer, the value is already computed.
        For j != i_layer, the value may be None (to be computed by caller).
    K : int
        Number of layers.

    Returns
    -------
    float
        Ego-network weight value.
    """
    inter_set = intersect_list[i_layer]
    if inter_set is None:
        inei = neig_list[i_layer][u]
        jnei = neig_list[i_layer][v]
        inter_set = inei & jnei
        intersect_list[i_layer] = inter_set

    if len(inter_set) == 0:
        return 0.0

    all_intersections_flat = []
    for h in range(K):
        if intersect_list[h] is None:
            inter_h = neig_list[h][u] & neig_list[h][v]
            intersect_list[h] = inter_h
        else:
            inter_h = intersect_list[h]
        all_intersections_flat.extend(inter_h)

    com_counter = Counter(all_intersections_flat)

    pint_sum = 0.0
    for k in inter_set:
        p1 = weights_list[i_layer].get(code(u, k), 0.0)
        p2 = weights_list[i_layer].get(code(v, k), 0.0)
        number_com = com_counter[k]
        pint_sum += (number_com / K) * (p1 + p2)

    inei = neig_list[i_layer][u]
    jnei = neig_list[i_layer][v]

    if w_uso == 0.0:
        denomin = len(inei) + len(jnei)
    else:
        denomin = len(inei) + len(jnei) - 2

    if denomin == 0:
        return 0.0

    ego_val = (pint_sum / denomin) ** (1.0 / len(inter_set))
    return float(ego_val)


def consensus_net(
    adjL,
    threshold=0.5,
    tolerance=0.1,
    theta=0.04,
    nitermax=50,
    ncores=1,
    verbose=True,
):
    """
    Run the INet consensus algorithm.

    Direct translation of:
      R/ConsensusINet.R:34-589

    Parameters
    ----------
    adjL : list of np.ndarray or pandas.DataFrame
        List of weighted adjacency matrices with weights in [0,1].
        All must have the same dimensions and identical row/col names.
    threshold : float
        Final consensus threshold (default 0.5).
    tolerance : float
        Jaccard distance convergence tolerance (default 0.1).
    theta : float
        Neighborhood influence weight (default 0.04).
    nitermax : int
        Maximum iterations (default 50).
    ncores : int
        Number of CPU cores (default 1). Currently single-process only.
    verbose : bool
        Print progress (default True).

    Returns
    -------
    dict
        "graphConsensus" : networkx.Graph
        "Comparison" : np.ndarray
        "similarGraphs" : list of networkx.Graph
    """
    from copy import deepcopy
    import pandas as pd

    has_names = False
    orig_names = None
    if len(adjL) > 0:
        if isinstance(adjL[0], pd.DataFrame):
            has_names = len(adjL[0].index) > 0
            if has_names:
                orig_names = list(adjL[0].index)

    graphs = _adj_to_graphs(adjL)
    K = len(graphs)
    N = graphs[0].number_of_nodes()

    comparison = []
    count = 0

    while True:
        comp = jaccard_all(graphs)

        if count == 0:
            if comp < tolerance:
                if verbose:
                    print("Multilayer network distance: less than tolerance.")
                break

        graph_backup = [deepcopy(g) for g in graphs]

        union_edges = set()
        for g in graphs:
            for u, v in g.edges():
                if u < v:
                    union_edges.add((u, v))
                else:
                    union_edges.add((v, u))
        edgelist = sorted(union_edges)
        if len(edgelist) == 0:
            comp_post = comp
            graph_change = graph_backup
        else:
            neig_list, weights_list = _build_hashes(graphs, edgelist)

            new_graphs = [deepcopy(g) for g in graphs]

            for i in range(K):
                g_i = new_graphs[i]

                for u, v in edgelist:
                    inei = neig_list[i][u]
                    jnei = neig_list[i][v]
                    intersect_set_i = inei & jnei
                    intersect_list = [None] * K
                    intersect_list[i] = intersect_set_i

                    w_uso = weights_list[i].get(code(u, v), 0.0)

                    w_others = []
                    for j in range(K):
                        if j == i:
                            continue
                        inter_j = neig_list[j][u] & neig_list[j][v]
                        intersect_list[j] = inter_j
                        w_altri = weights_list[j].get(code(u, v), 0.0)
                        w_others.append(w_altri)

                    peso = (w_uso + np.mean(w_others)) / 2.0

                    pesi_ego_nus = _compute_ego_weight(
                        i_layer=i,
                        u=u, v=v,
                        w_uso=w_uso,
                        neig_list=neig_list,
                        weights_list=weights_list,
                        intersect_list=intersect_list,
                        K=K,
                    )

                    pesi_ego_others = []
                    others_indices = [j for j in range(K) if j != i]
                    for pos_j, j in enumerate(others_indices):
                        inter_j = neig_list[j][u] & neig_list[j][v]
                        intersect_list[j] = inter_j
                        w_altri_ego = w_others[pos_j]

                        if len(inter_j) == 0:
                            pesi_ego_altri = 0.0
                        else:
                            all_intersections_flat = []
                            for h in range(K):
                                if intersect_list[h] is None:
                                    inter_h = neig_list[h][u] & neig_list[h][v]
                                    intersect_list[h] = inter_h
                                else:
                                    inter_h = intersect_list[h]
                                all_intersections_flat.extend(inter_h)
                            com_counter = Counter(all_intersections_flat)

                            pint_sum = 0.0
                            for k in inter_j:
                                p1 = weights_list[j].get(code(u, k), 0.0)
                                p2 = weights_list[j].get(code(v, k), 0.0)
                                number_com = com_counter[k]
                                pint_sum += (number_com / K) * (p1 + p2)

                            if w_altri_ego == 0:
                                denomin = len(neig_list[j][u]) + len(neig_list[j][v])
                            else:
                                denomin = len(neig_list[j][u]) + len(neig_list[j][v]) - 2

                            if denomin == 0:
                                pesi_ego_altri = 0.0
                            else:
                                pesi_ego_altri = (pint_sum / denomin) ** (1.0 / len(inter_j))

                        pesi_ego_others.append(pesi_ego_altri)

                    if len(pesi_ego_others) > 0:
                        pesi_ego = (pesi_ego_nus + np.mean(pesi_ego_others)) / 2.0
                    else:
                        pesi_ego = pesi_ego_nus

                    weight_i1 = peso + theta * pesi_ego
                    if weight_i1 > 1.0:
                        weight_i1 = 1.0

                    weight_mat(weight_i1, g_i, (u, v))

            graph_change = new_graphs

        graphs_restored = graph_backup

        comp_post = jaccard_all(graph_change)
        if verbose:
            print(f"Multilayer network distance: {comp_post}")

        if count == 0:
            comparison = [comp]
        comparison.append(comp_post)

        if comp > comp_post:
            graphs = graph_change
        else:
            if verbose:
                print("Distance doesn't decrease.")
            break

        count += 1

        if count > nitermax:
            if verbose:
                print("Maximum iteration.")
            break

        if comp_post < tolerance:
            break

    mats = _graphs_to_matrices(graphs)
    matrix_mean = np.mean(np.stack(mats, axis=0), axis=0)
    matrix_mean[matrix_mean < threshold] = 0.0

    consensus_g = nx.Graph()
    consensus_g.add_nodes_from(range(N))
    if has_names and orig_names is not None:
        nx.set_node_attributes(consensus_g, dict(enumerate(orig_names)), "name")
    for i in range(N):
        for j in range(i + 1, N):
            w = float(matrix_mean[i, j])
            if w != 0.0:
                consensus_g.add_edge(i, j, weight=w)

    if has_names and orig_names is not None:
        for g in graphs:
            nx.set_node_attributes(g, dict(enumerate(orig_names)), "name")

    return {
        "graphConsensus": consensus_g,
        "Comparison": np.array(comparison, dtype=np.float64),
        "similarGraphs": graphs,
    }
