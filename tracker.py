# tracker.py  ────────────────────────────────────────────────────────────
from collections import defaultdict
from typing import Dict, Set, Tuple, List

def overlap(a: set, b: set) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))

class CommunityTracker:
    def __init__(self, sim_tol=0.3, max_miss=5):
        self.sim_tol   = sim_tol
        self.max_miss  = max_miss
        self.next_id   = 0
        self.active: Dict[int, dict] = {}   # id -> {"members", "last_seen", "age", "miss"}
    
    # ------------------------------------------------------------------
    def _new_id(self) -> int:
        nid = self.next_id
        self.next_id += 1
        return nid
    
    # ------------------------------------------------------------------
    def step(self, clusters: Dict[int, Set[str]], t: int
            ) -> Tuple[Dict[str, int], List[int], List[int]]:
        """
        clusters : {local_id: node_set} from Louvain at window t
        returns  : (node→glob_id mapping, births list, deaths list)
        """
        # 1) compute best overlaps
        assignments = {}            # local_id -> glob_id
        used_ids = set()
        births   = []
        # sort larger clusters first (helps stability)
        for lid, nodes in sorted(clusters.items(),
                                 key=lambda kv: -len(kv[1])):
            best_gid, best_ov = None, 0.0
            for gid, info in self.active.items():
                if gid in used_ids:           # one‑to‑one
                    continue
                ov = overlap(nodes, info["members"])
                if ov > best_ov:
                    best_gid, best_ov = gid, ov
            if best_ov >= self.sim_tol:
                assignments[lid] = best_gid
                used_ids.add(best_gid)
            else:
                # birth
                gid = self._new_id()
                assignments[lid] = gid
                births.append(gid)
        
        # 2) update active table & mark misses
        still_alive = {}
        for lid, gid in assignments.items():
            nodes = clusters[lid]
            still_alive[gid] = {
                "members"  : nodes,
                "last_seen": t,
                "age"      : self.active.get(gid, {}).get("age", 0) + 1,
                "miss"     : 0
            }
        
        # increment miss counters for unmatched old communities
        deaths = []
        for gid, info in self.active.items():
            if gid not in still_alive:
                info["miss"] += 1
                if info["miss"] > self.max_miss:
                    deaths.append(gid)
                else:
                    still_alive[gid] = info
        
        self.active = still_alive
        
        # 3) node→gid mapping for this window
        node_to_gid = {n: assignments[lid]
                       for lid, nodes in clusters.items()
                       for n in nodes}
        return node_to_gid, births, deaths
