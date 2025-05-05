import numpy as np
import pandas as pd
import networkx as nx

__all__ = [
    "correlation_matrix", "distance_matrix", "mst_from_distance"
]

def correlation_matrix(ret_win: pd.DataFrame) -> pd.DataFrame:
    """Equal‐time Pearson correlations."""
    return ret_win.corr()

def distance_matrix(C: pd.DataFrame) -> pd.DataFrame:
    """Mantegna distance d = √(2(1‐ρ))."""
    C = C.copy().clip(-1, 1)
    D = np.sqrt(2 * (1 - C))
    np.fill_diagonal(D.values, 0.0)
    return D

def mst_from_distance(D: pd.DataFrame) -> nx.Graph:
    """
    Return MST (N-1 edges) with distance weights,
    dropping any assets that have NaNs in their distances.
    """
    # Drop any rows/cols that contain NaN
    D_clean = D.dropna(axis=0, how='any').dropna(axis=1, how='any')
    if D_clean.shape[0] < 2:
        raise ValueError("Not enough valid assets to build an MST: {}".format(D_clean.shape))
    
    G_full = nx.from_pandas_adjacency(D_clean)
    mst = nx.minimum_spanning_tree(G_full, weight="weight")
    return mst

# ───────── performance metrics  ────────────────────────────────────────
def metrics(series: list[float]) -> dict[str, float]:
    if len(series) == 0:
        return dict(ann_ret=0, ann_vol=0, sharpe=0)
    s  = np.array(series)
    mu = s.mean()
    vol= s.std()
    ann_ret = (1 + mu) ** 252 - 1
    ann_vol = vol * np.sqrt(252)
    sharpe  = 0 if ann_vol == 0 else ann_ret / ann_vol
    return dict(annual_return=ann_ret,
                annual_vol   =ann_vol,
                sharpe       =sharpe)
