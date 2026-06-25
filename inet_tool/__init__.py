"""
INet-Tool Python package — network integration for multi-layer networks.

Translated from R package INetTool v0.1.1
Original authors: Policastro, Magnani, Angelini, Carissimo (2024)
Paper: doi:10.1007/s00180-024-01536-8
"""

from ._internal import get_lower_tri_noDiag, code
from .distance import jaccard_all, jw_matrix, jw_mean
from .preprocess import adj_rename, construction_graph
from .consensus import consensus_net
from .postprocess import density_net, threshold_net, specific_net
from .measures import measures_net
from .plots import plot_inet, plot_l, plot_c

__version__ = "0.1.1"
__all__ = [
    "get_lower_tri_noDiag",
    "code",
    "jaccard_all",
    "jw_matrix",
    "jw_mean",
    "adj_rename",
    "construction_graph",
    "consensus_net",
    "density_net",
    "threshold_net",
    "specific_net",
    "measures_net",
    "plot_inet",
    "plot_l",
    "plot_c",
]
