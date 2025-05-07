# tracker.py

from collections import defaultdict
from typing import Dict, Set, Tuple, List
import numpy as np


def overlap(a: set, b: set) -> float:
    """
    Compute the normalized overlap between two sets.

    Args:
        a (set): First set.
        b (set): Second set.

    Returns:
        float: Overlap score ∈ [0, 1], defined as intersection / min(|a|, |b|).
    """
    return len(a & b) / max(1, min(len(a), len(b)))


class CommunityTracker:
    """
    Tracks the identity and evolution of communities over time.

    Attributes:
        sim_tol (float): Overlap threshold to consider a community the same across time steps.
        max_miss (int): Maximum number of windows a community can disappear before considered 'dead'.
        next_id (int): Counter for assigning new global community IDs.
        active (Dict[int, dict]): Dictionary of currently tracked communities with metadata.

    Methods:
        step(clusters, t): Updates the tracker with new communities at time t and returns:
                           - node to global community ID mapping
                           - list of new community IDs ("births")
                           - list of removed community IDs ("deaths")
        age(gid): Returns the age (in steps) of a given global community ID.
        half_life(): Returns the median age of all currently active communities.
    """

    def __init__(self, sim_tol=0.3, max_miss=5):
        """
        Initialize a CommunityTracker.

        Args:
            sim_tol (float): Jaccard overlap threshold for matching.
            max_miss (int): Max missed steps allowed before declaring a community as 'dead'.
        """
        self.sim_tol = sim_tol
        self.max_miss = max_miss
        self.next_id = 0
        self.active: Dict[int, dict] = {}  # global_id → {"members", "last_seen", "age", "miss"}

    def _new_id(self) -> int:
        """
        Generate a new unique global community ID.

        Returns:
            int: New unique ID.
        """
        nid = self.next_id
        self.next_id += 1
        return nid

    def age(self, gid: int) -> int:
        """
        Get the age of a community in number of time windows.

        Args:
            gid (int): Global community ID.

        Returns:
            int: Age of the community.
        """
        return self.active.get(gid, {}).get("age", 0)

    def half_life(self) -> float:
        """
        Return the median age of currently active communities.

        Returns:
            float: Median community lifetime.
        """
        ages = [info["age"] for info in self.active.values()]
        return float(np.median(ages)) if ages else 0.0

    def step(self, clusters: Dict[int, Set[str]], t: int
            ) -> Tuple[Dict[str, int], List[int], List[int]]:
        """
        Perform a single update step of the tracker at time t.

        Args:
            clusters (Dict[int, Set[str]]): Mapping of local community ID to its set of members.
            t (int): Current time window index.

        Returns:
            Tuple:
                - node_to_gid (Dict[str, int]): Node-to-global-community-ID mapping.
                - births (List[int]): List of new community IDs created at this step.
                - deaths (List[int]): List of community IDs considered dead at this step.
        """
        # Match current clusters to existing global communities
        assignments = {}       # local_id → global_id
        used_ids = set()
        births = []

        # Process larger communities first to ensure stable assignments
        for lid, nodes in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            best_gid, best_ov = None, 0.0
            for gid, info in self.active.items():
                if gid in used_ids:
                    continue
                ov = overlap(nodes, info["members"])
                if ov > best_ov:
                    best_gid, best_ov = gid, ov
            if best_ov >= self.sim_tol:
                assignments[lid] = best_gid
                used_ids.add(best_gid)
            else:
                # New community (birth)
                gid = self._new_id()
                assignments[lid] = gid
                births.append(gid)

        # Update active communities
        still_alive = {}
        for lid, gid in assignments.items():
            nodes = clusters[lid]
            still_alive[gid] = {
                "members": nodes,
                "last_seen": t,
                "age": self.active.get(gid, {}).get("age", 0) + 1,
                "miss": 0
            }

        # Track communities that disappeared
        deaths = []
        for gid, info in self.active.items():
            if gid not in still_alive:
                info["miss"] += 1
                if info["miss"] > self.max_miss:
                    deaths.append(gid)
                else:
                    still_alive[gid] = info

        # Save state
        self.active = still_alive

        # Build mapping: node → global ID
        node_to_gid = {
            n: assignments[lid]
            for lid, nodes in clusters.items()
            for n in nodes
        }

        return node_to_gid, births, deaths
