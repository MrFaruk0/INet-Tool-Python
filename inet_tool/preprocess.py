"""
Pre-processing functions for INet-Tool.
Translated from R package INetTool.

Source files:
  - R/AdjSameName.R:15-59       (adj_rename)
  - R/ConstructionGraph.R:20-62 (constructionGraph)
"""

import numpy as np
import networkx as nx
from scipy.stats import pearsonr


def adj_rename(adjL):
    """
    Construct a list of adjacency matrices with identical row and column
    names by taking the union of all node names across matrices.
    Missing nodes are filled with zeros.

    Direct translation of:
      R/AdjSameName.R:15-59

    Original R:
      adj_rename <- function(adjL) {
        geneName <- NULL
        for (i in 1:length(adjL)) {
          genename <- rownames(adjL[[i]])
          geneName <- c(geneName, genename)
        }
        genes <- unique(geneName)
        Mat <- vector(mode = "list", length = length(adjL))
        for (z in 1:length(adjL)) {
          Mat[[z]] <- matrix(0, nrow=length(genes), ncol=length(genes))
          rownames(Mat[[z]]) <- genes
          colnames(Mat[[z]]) <- genes
          for (i in 1:(dim(adjL[[z]])[1])) {
            for (j in 1:(dim(adjL[[z]])[2])) {
              valore <- adjL[[z]][rownames(adjL[[z]])[i], colnames(adjL[[z]])[j]]
              Mat[[z]][which(rownames(Mat[[z]])==rownames(adjL[[z]])[i]),
                       which(colnames(Mat[[z]])==colnames(adjL[[z]])[j])] <- valore
            }
          }
        }
        return(Mat)
      }

    Parameters
    ----------
    adjL : list of pandas.DataFrame
        Each DataFrame is a square adjacency matrix with named rows and columns.
        R matrices with row/col names map naturally to pandas DataFrames.

    Returns
    -------
    list of pandas.DataFrame
        Each matrix has identical row/col labels = union of all input names.
    """
    import pandas as pd

    all_names = []
    for mat in adjL:
        all_names.extend(mat.index.tolist())
    genes = sorted(set(all_names))

    result = []
    for mat in adjL:
        new_mat = pd.DataFrame(
            np.zeros((len(genes), len(genes)), dtype=np.float64),
            index=genes,
            columns=genes,
        )
        orig_names = mat.index.tolist()
        for i_name in orig_names:
            for j_name in orig_names:
                valore = mat.loc[i_name, j_name]
                new_mat.loc[i_name, j_name] = valore
        result.append(new_mat)

    return result


def construction_graph(data, perc=0.95, plot=True):
    """
    Construct networks from raw data using Pearson correlation and
    proportional thresholding.

    Direct translation of:
      R/ConstructionGraph.R:20-62

    Original R steps (for each data layer):
      1. Compute Pearson correlation matrix (stats::cor)
      2. Plot histogram of lower-triangle weights
      3. Compute perc-th quantile as threshold (stats::quantile)
      4. Zero out correlations below threshold
      5. Create igraph from thresholded matrix (mode="upper", diag=FALSE, weighted=TRUE)
      6. Compute Louvain modularity
      7. Collect edge count, vertex count, modularity

    Parameters
    ----------
    data : list of np.ndarray
        Each element shape (samples, nodes). Columns = variables/nodes,
        rows = observations. (R's cor() treats columns as variables.)
    perc : float
        Percentile threshold (default 0.95 = keep top 5% of weights).
    plot : bool
        If True, display histograms (default True).

    Returns
    -------
    dict
        "Threshold" : list of dicts
            Each contains {str(perc): threshold_value, "edge": count,
            "node": count, "modularity Louvain": value}.
        "Graphs" : list of networkx.Graph
        "Adj" : list of np.ndarray
            Thresholded correlation matrices.
    """
    from ._internal import get_lower_tri_noDiag

    Graphs = []
    CorrM = []
    Threshold = []

    for i, mat in enumerate(data):
        corr_mat = np.corrcoef(mat, rowvar=False)

        if plot:
            lower_vals = corr_mat[np.tril_indices_from(corr_mat, k=-1)]
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.hist(lower_vals, bins=30, color="#69b3a2", edgecolor="#e9ecef")
                plt.xlabel(f"Weights {i}")
                plt.show()
            except ImportError:
                pass

        thresh = np.quantile(corr_mat, perc, method="linear")
        corr_mat[corr_mat < thresh] = 0.0

        g = nx.from_numpy_array(corr_mat)
        for u, v in list(g.edges()):
            g[u][v]["weight"] = float(corr_mat[u, v])

        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(g, weight="weight", seed=42)
        except ImportError:
            from networkx.algorithms.community import louvain_communities as _lc
            communities = _lc(g, weight="weight", seed=42)

        membership = [0] * g.number_of_nodes()
        for cid, comm in enumerate(communities):
            for node in comm:
                membership[node] = cid

        mod = nx.community.modularity(g, communities, weight="weight")

        e = g.number_of_edges()
        v = g.number_of_nodes()

        Graphs.append(g)
        CorrM.append(corr_mat)
        Threshold.append({
            str(perc): float(thresh),
            "edge": e,
            "node": v,
            "modularity Louvain": float(mod),
        })

    return {
        "Threshold": Threshold,
        "Graphs": Graphs,
        "Adj": CorrM,
    }
