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


# ------------------ Helper Functions ------------------

def save_partition(part: Dict[str, int], path: pathlib.Path) -> None:
    """
    Save a community partition to a JSON file.

    Args:
        part (Dict[str, int]): Dictionary mapping node to community ID.
        path (Path): Output path for the JSON file.
    """
    path.write_text(json.dumps(part, indent=2))


def plot_partition(sim_graph: nx.Graph, part: Dict[str, int],
                   path: pathlib.Path, dpi: int = 300) -> None:
    """
    Visualize and save a graph colored by community partition.

    Args:
        sim_graph (nx.Graph): Graph with weights (typically MST).
        part (Dict[str, int]): Node-to-community mapping.
        path (Path): Output path for the plot image.
        dpi (int): Resolution of the saved image.
    """
    pos = nx.spring_layout(sim_graph, seed=42, weight="weight")
    cmap = plt.cm.get_cmap("tab20", len(set(part.values())))
    plt.figure(figsize=(10, 8), dpi=dpi)
    for cid in sorted(set(part.values())):
        nx.draw_networkx_nodes(sim_graph, pos,
            nodelist=[n for n, c in part.items() if c == cid],
            node_size=40, node_color=[cmap(cid)])
    nx.draw_networkx_edges(sim_graph, pos, width=0.5, alpha=0.4)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ------------------ Community Utilities ------------------

def louvain_partition(G: nx.Graph) -> Dict[str, int]:
    """
    Run Louvain community detection on a graph.

    Args:
        G (nx.Graph): Weighted graph (e.g., MST with distances as weights).

    Returns:
        Dict[str, int]: Node-to-community mapping.
    """
    return community_louvain.best_partition(G, weight="weight")


def eigen_centrality(G: nx.Graph) -> Dict[str, float]:
    """
    Compute eigenvector centrality of each node in the graph.

    Args:
        G (nx.Graph): Weighted undirected graph.

    Returns:
        Dict[str, float]: Node-to-centrality score mapping.
    """
    return nx.eigenvector_centrality_numpy(G, weight="weight")


def select_nodes(
    G: nx.Graph,
    part: Dict[str, int],
    k: int,
    mode: str = "peripheral",
    rng: random.Random | None = None,
) -> List[str]:
    """
    Select representative nodes from each community.

    Args:
        G (nx.Graph): Input graph.
        part (Dict[str, int]): Community partition of the graph.
        k (int): Number of nodes to select per community.
        mode (str): Selection strategy: 'peripheral', 'central', or 'random'.
        rng (Random, optional): Random number generator instance.

    Returns:
        List[str]: Selected nodes from all communities.
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
            raise ValueError(f"Unknown mode {mode}")
        selected.extend(nodes[: min(k, len(nodes))])
    return selected


# ------------------ Portfolio Optimization ------------------

def markowitz_meanvar(
    ret_win: pd.DataFrame,
    assets: List[str],
    lam: float = 5.0,
    max_w: float = 0.10,
) -> pd.Series:
    """
    Perform long-only Markowitz mean–variance optimization with Ledoit-Wolf shrinkage.

    Args:
        ret_win (pd.DataFrame): Window of returns (T x N).
        assets (List[str]): Asset symbols to include.
        lam (float): Risk aversion parameter.
        max_w (float): Maximum weight per stock.

    Returns:
        pd.Series: Optimal portfolio weights indexed by asset.
    """
    if not assets:
        raise ValueError("empty asset list")

    X = ret_win[assets].dropna(how="all", axis=1)
    mu = X.mean().values
    Σ = LedoitWolf().fit(X.values).covariance_

    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, Σ) - lam * mu @ w),
        [cp.sum(w) == 1, w >= 0, w <= max_w]
    )
    prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None or np.isnan(w.value).any():
        logging.warning("mean–var failed → equal weights")
        w_val = np.full(n, 1 / n)
    else:
        w_val = np.maximum(w.value, 0)
        w_val /= w_val.sum()

    return pd.Series(w_val, index=X.columns)


def markowitz_meanvar_full(ret_win: pd.DataFrame, **kw) -> pd.Series:
    """
    Shortcut for Markowitz optimization on all available assets.

    Args:
        ret_win (pd.DataFrame): Return window (T x N).

    Returns:
        pd.Series: Portfolio weights.
    """
    return markowitz_meanvar(ret_win, list(ret_win.columns), **kw)


# ------------------ Partition Similarity ------------------

def _j(a: set, b: set) -> float:
    """
    Jaccard similarity between two sets.

    Args:
        a (set): First set.
        b (set): Second set.

    Returns:
        float: Jaccard similarity.
    """
    return len(a & b) / len(a | b) if a | b else 1.0


def partition_similarity(p: Dict[str, int], q: Dict[str, int]) -> float:
    """
    Compute similarity between two partitions using average max Jaccard overlap.

    Args:
        p (Dict[str, int]): Partition at time t.
        q (Dict[str, int]): Partition at time t+1.

    Returns:
        float: Similarity score in [0, 1].
    """
    A = defaultdict(set)
    B = defaultdict(set)
    for n, c in p.items(): A[c].add(n)
    for n, c in q.items(): B[c].add(n)
    return float(np.mean([max(_j(a, b) for b in B.values()) for a in A.values()]))


def born_dead_comms(p: Dict[str, int], q: Dict[str, int]) -> Tuple[List[int], List[int]]:
    """
    Identify communities that are born (in q but not in p) or dead (in p but not in q).

    Args:
        p (Dict[str, int]): Partition at t.
        q (Dict[str, int]): Partition at t+1.

    Returns:
        Tuple[List[int], List[int]]: (born, dead) community IDs.
    """
    return list(set(q.values()) - set(p.values())), list(set(p.values()) - set(q.values()))


# ------------------ MST Layer Analysis ------------------

def node_layers(mst: nx.Graph, central: str) -> Dict[str, int]:
    """
    Compute shortest-path layer of each node from a central node.

    Args:
        mst (nx.Graph): Minimum Spanning Tree.
        central (str): Node to serve as root.

    Returns:
        Dict[str, int]: Node-to-layer distance.
    """
    return nx.single_source_shortest_path_length(mst, central)


def mean_layer_per_cluster(
    mst: nx.Graph,
    partition: Dict[str, int],
    central: str,
) -> Dict[int, float]:
    """
    Compute average path depth for each cluster in the MST.

    Args:
        mst (nx.Graph): Minimum Spanning Tree.
        partition (Dict[str, int]): Node-to-community assignment.
        central (str): Central reference node.

    Returns:
        Dict[int, float]: Community ID to average layer depth.
    """
    layers = node_layers(mst, central)
    comm_layers: Dict[int, List[int]] = {}
    for node, cid in partition.items():
        comm_layers.setdefault(cid, []).append(layers[node])
    return {cid: float(np.mean(v)) for cid, v in comm_layers.items()}
