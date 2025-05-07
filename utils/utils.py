import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Set, DefaultDict

__all__ = [
    "correlation_matrix", "distance_matrix", "mst_from_distance",
    "metrics", "partition_similarity"
]

def correlation_matrix(ret_win: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for a given window of log returns.

    Parameters:
        ret_win (pd.DataFrame): DataFrame of log returns with shape (T, N), 
                                where T is number of time points, N is number of assets.

    Returns:
        pd.DataFrame: NxN Pearson correlation matrix.
    """
    return ret_win.corr()


def distance_matrix(C: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Mantegna distance matrix from a correlation matrix.

    Uses the transformation: D_ij = sqrt(2 * (1 - C_ij))

    Parameters:
        C (pd.DataFrame): NxN correlation matrix.

    Returns:
        pd.DataFrame: NxN distance matrix with zero diagonals.
    """
    C = C.copy().clip(-1, 1)
    D = np.sqrt(2 * (1 - C))
    np.fill_diagonal(D.values, 0.0)
    return D


def mst_from_distance(D: pd.DataFrame) -> nx.Graph:
    """
    Construct a Minimum Spanning Tree (MST) from a distance matrix.

    Parameters:
        D (pd.DataFrame): NxN symmetric distance matrix with assets as indices/columns.

    Returns:
        nx.Graph: Undirected MST with 'weight' attribute on edges.
    
    Raises:
        ValueError: If the cleaned distance matrix has fewer than 2 valid assets.
    """
    D_clean = D.dropna(axis=0, how='any').dropna(axis=1, how='any')
    if D_clean.shape[0] < 2:
        raise ValueError("Not enough valid assets to build an MST: {}".format(D_clean.shape))
    
    G_full = nx.from_pandas_adjacency(D_clean)
    mst = nx.minimum_spanning_tree(G_full, weight="weight")
    return mst


def metrics(series: list[float]) -> dict[str, float]:
    """
    Compute annualized return, volatility, and Sharpe ratio.

    Parameters:
        series (list of float): Daily returns of the strategy.

    Returns:
        dict[str, float]: Dictionary with 'annual_return', 'annual_vol', and 'sharpe'.
    """
    if len(series) == 0:
        return dict(annual_return=0, annual_vol=0, sharpe=0)
    
    s = np.array(series)
    mu = s.mean()
    vol = s.std()
    ann_ret = (1 + mu) ** 252 - 1
    ann_vol = vol * np.sqrt(252)
    sharpe = 0 if ann_vol == 0 else ann_ret / ann_vol
    return dict(annual_return=ann_ret,
                annual_vol=ann_vol,
                sharpe=sharpe)


def partition_similarity(p: Dict[str, int], q: Dict[str, int]) -> float:
    """
    Compute the Jaccard-based similarity between two partitions.

    Parameters:
        p (Dict[str, int]): Mapping of node to community label at time t.
        q (Dict[str, int]): Mapping of node to community label at time t+1.

    Returns:
        float: Average maximum Jaccard overlap across communities.
    """
    A = DefaultDict(set)
    B = DefaultDict(set)
    
    for n, c in p.items():
        A[c].add(n)
    for n, c in q.items():
        B[c].add(n)
    
    _j = lambda a, b: len(a & b) / len(a | b) if a | b else 1.0
    return float(np.mean([max(_j(a, b) for b in B.values()) for a in A.values()]))
