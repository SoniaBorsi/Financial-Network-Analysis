# community_tools.py  ──────────────────────────────────────────────────
import json, pathlib, logging, random
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from scipy.optimize import linear_sum_assignment

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import cvxpy as cp
import community as community_louvain
from sklearn.covariance import LedoitWolf   

# --------------- helpers ------------------------------
def save_partition(part: Dict[str, int], path: pathlib.Path) -> None:
    path.write_text(json.dumps(part, indent=2))

def plot_partition(sim_graph: nx.Graph, part: Dict[str, int],
                   path: pathlib.Path, dpi: int = 300) -> None:
    pos = nx.spring_layout(sim_graph, seed=42, weight="weight")
    cmap = plt.cm.get_cmap("tab20", len(set(part.values())))
    plt.figure(figsize=(10, 8), dpi=dpi)
    for cid in sorted(set(part.values())):
        nx.draw_networkx_nodes(sim_graph, pos,
            nodelist=[n for n, c in part.items() if c == cid],
            node_size=40, node_color=[cmap(cid)])
    nx.draw_networkx_edges(sim_graph, pos, width=0.5, alpha=0.4)
    plt.axis("off"); plt.savefig(path, bbox_inches="tight"); plt.close()

# --------------- community utilities ----------------------------------
def louvain_partition(G: nx.Graph) -> Dict[str, int]:
    return community_louvain.best_partition(G, weight="weight")

def eigen_centrality(G: nx.Graph) -> Dict[str, float]:
    return nx.eigenvector_centrality_numpy(G, weight="weight")

def select_nodes(
    G: nx.Graph,
    part: Dict[str, int],
    k: int,
    mode: str = "peripheral",
    rng: random.Random | None = None,
) -> List[str]:
    """
    Pick *k* nodes per community using:
      • 'peripheral'  : lowest eigen‑centrality (default)
      • 'central'     : highest eigen‑centrality
      • 'random'      : uniform random
    """
    rng = rng or random.Random()
    ec = eigen_centrality(G)
    selected: list[str] = []
    for cid in set(part.values()):
        nodes = [n for n in G.nodes if part[n] == cid]
        if mode == "peripheral":
            nodes = sorted(nodes, key=lambda n: ec[n])
        elif mode == "central":
            nodes = sorted(nodes, key=lambda n: ec[n], reverse=True)
        elif mode == "random":
            rng.shuffle(nodes)
        else:
            raise ValueError(f"unknown mode {mode}")
        selected.extend(nodes[: min(k, len(nodes))])
    return selected

# ───────────────────────────────────────────────────────────
#  mean–variance Markowitz with shrinkage + weight cap
# ───────────────────────────────────────────────────────────
def markowitz_meanvar(
    ret_win   : pd.DataFrame,
    assets    : list[str],
    lam       : float = 5.0,
    max_w     : float = 0.10,
) -> pd.Series:
    """
    Long‑only mean–variance optimiser
    ( minimise w'Σ w − λ w' μ ),  Σ estimated by Ledoit‑Wolf.
    """
    if not assets:
        raise ValueError("empty asset list")

    X   = ret_win[assets].dropna(how="all", axis=1)
    mu  = X.mean().values
    Σ   = LedoitWolf().fit(X.values).covariance_

    n   = len(mu)
    w   = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, Σ) - lam * mu @ w),
        [cp.sum(w) == 1, w >= 0, w <= max_w]
    )
    prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None or np.isnan(w.value).any():
        logging.warning("mean‑var failed → equal weights")
        w_val = np.full(n, 1 / n)
    else:
        w_val = np.maximum(w.value, 0)
        w_val /= w_val.sum()

    return pd.Series(w_val, index=X.columns)

def markowitz_meanvar_full(ret_win: pd.DataFrame, **kw) -> pd.Series:
    return markowitz_meanvar(ret_win, list(ret_win.columns), **kw)

# --------------- partition comparison (unchanged) ---------------------
def _j(a: set, b: set) -> float: return len(a&b)/len(a|b) if a|b else 1
def partition_similarity(p, q):
    A = defaultdict(set); B = defaultdict(set)
    for n,c in p.items(): A[c].add(n)
    for n,c in q.items(): B[c].add(n)
    return float(np.mean([max(_j(a,b) for b in B.values()) for a in A.values()]))

def born_dead_comms(p, q):
    return list(set(q.values())-set(p.values())), list(set(p.values())-set(q.values()))




def node_layers(mst: nx.Graph, central: str) -> dict[str, int]:
    """Shortest‑path layer (#edges) of every node from the chosen central vertex."""
    return nx.single_source_shortest_path_length(mst, central)


def mean_layer_per_cluster(
    mst: nx.Graph,
    partition: dict[str, int],
    central: str,
) -> dict[int, float]:
    """Average layer depth for each Louvain community."""
    layers = node_layers(mst, central)
    comm_layers: dict[int, list[int]] = {}
    for node, cid in partition.items():
        comm_layers.setdefault(cid, []).append(layers[node])
    return {cid: float(np.mean(v)) for cid, v in comm_layers.items()}


