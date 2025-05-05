import json, logging, random
from collections import defaultdict
from pathlib import Path
from typing   import Dict, Set, DefaultDict

import numpy as np
import pandas as pd
import tqdm
import yaml

from data      import get_returns
from utils     import correlation_matrix, distance_matrix, mst_from_distance, metrics
from community_tools import (
    louvain_partition, select_nodes,
    markowitz_meanvar, markowitz_meanvar_full,
    plot_partition, save_partition,
)
from tracker   import CommunityTracker

# ───────── configuration & folders ──────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)s │ %(message)s")

CFG: Dict = yaml.safe_load(Path("config.yml").read_text())

OUT_ROOT = Path(CFG["output"]["root"]); OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_COMM = OUT_ROOT / "communities";   OUT_COMM.mkdir(exist_ok=True)

# data & rolling‑window parameters
returns = get_returns(Path(CFG["data"]["file"]))
L, H    = CFG["window"]["length"], CFG["window"]["step"]

# community / selection params
K          = CFG["community"]["k_peripheral"]
PICK_MODE  = CFG["community"].get("pick_mode", "peripheral")

# portfolio params (optional block)
PORT      = CFG.get("portfolio", {})
LAM       = float(PORT.get("lambda", 5.0))
MAX_W     = float(PORT.get("max_w", 0.10))

# misc
VAR_TH = 1e-6
RIDGE  = 1e-3
rng    = random.Random(42)

# ───────── trackers ────────────────────────────────────────────────────
tracker = CommunityTracker(sim_tol=0.55, max_miss=5)
perf_log      = []
portfolios    = {}

cum_comm = cum_eq = cum_full = 1.0
prev_assets, prev_w_comm = None, None

daily_eq_comm = []
daily_eq_eq   = []
daily_eq_full = []

# ───────── rolling loop ────────────────────────────────────────────────
for end in tqdm.tqdm(range(L, len(returns), H), desc="rolling windows"):
    date     = returns.index[end-1]
    date_str = date.strftime("%Y-%m-%d")

    win = returns.iloc[end-L:end]
    win = win.loc[:, win.var(0) > VAR_TH]
    if win.shape[1] < 2:
        continue

    C   = correlation_matrix(win)
    D   = distance_matrix(C)
    mst = mst_from_distance(D)
    for _, _, d in mst.edges(data=True):
        d["weight"] = 1.0 / (d["weight"] + 1e-9)

    local_part: DefaultDict[int, Set[str]] = defaultdict(set)
    for n, cid in louvain_partition(mst).items():
        local_part[cid].add(n)

    node_gid, births, deaths = tracker.step(local_part, t=len(perf_log))
    gid_part = {n: gid for n, gid in node_gid.items()}

    save_partition(gid_part, OUT_COMM / f"{date_str}.json")
    plot_partition (mst, gid_part, OUT_COMM / f"{date_str}.png")

    rebalance = (prev_assets is None) or births or deaths
    if rebalance:
        assets_comm = select_nodes(mst, gid_part, K, mode=PICK_MODE, rng=rng)
        w_comm = markowitz_meanvar(win, assets_comm, lam=LAM, max_w=MAX_W)
        portfolios[date_str] = w_comm.to_dict()
        prev_assets, prev_w_comm = assets_comm, w_comm
    else:
        assets_comm, w_comm = prev_assets, prev_w_comm

    w_eq   = pd.Series(1 / len(assets_comm), index=assets_comm)
    w_full = markowitz_meanvar_full(win, lam=LAM, max_w=MAX_W)

    fwd = returns.loc[date:].iloc[1:H+1]
    def port_ret(w):
        if fwd.empty:
            return 0.0
        gross = (1 + fwd[w.index]).prod()
        return float((gross.values - 1) @ w.values)

    r_comm = port_ret(w_comm)
    r_eq   = port_ret(w_eq)
    r_full = port_ret(w_full)

    if H > 0:
        daily_eq_comm.append((1 + r_comm) ** (1 / H) - 1)
        daily_eq_eq.append((1 + r_eq)     ** (1 / H) - 1)
        daily_eq_full.append((1 + r_full) ** (1 / H) - 1)

    cum_comm *= 1 + r_comm
    cum_eq   *= 1 + r_eq
    cum_full *= 1 + r_full

    perf_log.append({
        "date": date_str,
        "births": births, "deaths": deaths,
        "rebalance": rebalance,
        "n_comm": len(tracker.active),
        "r_comm": r_comm, "r_eq": r_eq, "r_full": r_full,
        "cum_comm": cum_comm-1, "cum_eq": cum_eq-1, "cum_full": cum_full-1,
        "act_vs_eq":   cum_comm - cum_eq,
        "act_vs_full": cum_comm - cum_full,
    })

# ───────── metrics summary ─────────────────────────────────────────────
stats_comm = metrics(daily_eq_comm)
stats_eq   = metrics(daily_eq_eq)
stats_full = metrics(daily_eq_full)

print("\n\u2500\u2500\u2500\u2500 Summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
for name, stats in [
    ("Community",   stats_comm),
    ("Eq‑same",     stats_eq),
    ("Full‑MVP",    stats_full),
]:
    print(f"{name:<10} | AnnRet {stats['annual_return']*100:6.2f}%  | "
          f"AnnVol {stats['annual_vol']*100:5.2f}%  | Sharpe {stats['sharpe']:4.2f}")

# ───────── persist ─────────────────────────────────────────────────────
pd.DataFrame(perf_log).to_csv(OUT_ROOT / "performance.csv", index=False)
with open(OUT_ROOT / "portfolios.json", "w") as fp:
    json.dump(portfolios, fp, indent=2)

logging.info(
    "Final cumulative returns:\n"
    "  Community : %.2f%% | Eq‑same : %.2f%% | Full‑MVP : %.2f%%\n"
    "Annualised metrics:\n"
    "  Community : Ret %.2f%% | Vol %.2f%% | Sharpe %.2f\n"
    "  Eq‑same   : Ret %.2f%% | Vol %.2f%% | Sharpe %.2f\n"
    "  Full‑MVP  : Ret %.2f%% | Vol %.2f%% | Sharpe %.2f",
    100 * (cum_comm - 1), 100 * (cum_eq - 1), 100 * (cum_full - 1),
    100 * stats_comm["annual_return"], 100 * stats_comm["annual_vol"], stats_comm["sharpe"],
    100 * stats_eq["annual_return"],   100 * stats_eq["annual_vol"],   stats_eq["sharpe"],
    100 * stats_full["annual_return"], 100 * stats_full["annual_vol"], stats_full["sharpe"],
)
