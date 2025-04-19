#!/usr/bin/env python3
"""
Driver: run Sybil‑detection experiments and dump results.

* Logs every Nth step (set by --sample) to avoid long‑runtime slow‑downs.
* Writes:
    results/<name>.csv          – final snapshot
    results/<name>_ts.csv       – timeline (sampled)
    figs/<name>.png             – quick histogram
"""
from __future__ import annotations
import argparse, csv, pathlib
from typing import Dict, Callable
import matplotlib.pyplot as plt

from src.modules.trust_score   import TrustScoreNetwork
from src.modules.eigentrust    import EigenTrustNetwork
from src.modules.random_walks  import RandomWalkDetection


# --------------------------------------------------------------------------- #
def build_network(method: str, honest: int, sybil: int, seed: int):
    if method == "bayes":
        return TrustScoreNetwork(num_nodes=honest, num_sybil_nodes=sybil, seed=seed)
    if method == "eigen":
        return EigenTrustNetwork(num_nodes=honest, num_sybil_nodes=sybil, seed=seed)
    if method == "walk":
        return RandomWalkDetection(num_nodes=honest, num_sybil_nodes=sybil, seed=seed)
    raise ValueError(f"unknown method {method}")


def safe_bayes(node) -> float:
    if not node.trust_scores:
        return 0.5
    return max(v["alpha"] / (v["alpha"] + v["beta"])
               for v in node.trust_scores.values())


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Sybil‑detection experiment runner")
    ap.add_argument("--method",   required=True, choices=["bayes", "eigen", "walk"])
    ap.add_argument("--nodes",    type=int, default=50)
    ap.add_argument("--sybil",    type=int, default=10)
    ap.add_argument("--runtime",  type=int, default=30)
    ap.add_argument("--sample",   type=int, default=1,
                    help="log every N steps (≥1) to speed up long runs")
    ap.add_argument("--walks",    type=int, default=10_000)
    ap.add_argument("--weighted", action="store_true",
                    help="trust‑weighted edges for random walks")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--export",   choices=["none", "csv", "png", "all"], default="all")
    args = ap.parse_args()
    if args.sample < 1:
        raise ValueError("--sample must be ≥1")

    suffix = "w" if args.method == "walk" and args.weighted else ""
    res_dir = pathlib.Path("results"); res_dir.mkdir(exist_ok=True)
    fig_dir = pathlib.Path("figs");    fig_dir.mkdir(exist_ok=True)

    csv_fin = res_dir / f"{args.method}_{args.nodes}_{args.sybil}{suffix}.csv"
    csv_ts  = res_dir / f"{args.method}_{args.nodes}_{args.sybil}{suffix}_ts.csv"
    png_out = fig_dir / f"{args.method}_{args.nodes}_{args.sybil}{suffix}.png"

    net = build_network(args.method, args.nodes, args.sybil, args.seed)

    # ------------- score lambdas ------------- #
    def make_score() -> Callable[[str], float]:
        if args.method == "bayes":
            return lambda nid: safe_bayes(net.nodes[nid])

        if args.method == "eigen":
            def score(nid: str, cache: Dict[str, Dict[str, float]] = {}) -> float:
                if "vec" not in cache:
                    cache["vec"] = net.compute_global_trust_scores()
                return cache["vec"][nid]
            return score

        def score(nid: str, cache: Dict[str, Dict[str, float]] = {}) -> float:
            if "probs" not in cache:
                start = next(n for n, obj in net.nodes.items() if not obj.is_sybil)
                cache["probs"] = net.random_walk_visits(
                    start_node=start, steps=args.walks, use_weights=args.weighted
                )
            return cache["probs"].get(nid, 0.0)
        return score

    score_fn = make_score()

    # ------------- timeline CSV header ------------- #
    if args.export in ("csv", "all"):
        f_ts = csv_ts.open("w", newline=""); w_ts = csv.writer(f_ts)
        w_ts.writerow(["t", "node", "is_sybil", "score"])

    # ------------- simulation loop ------------- #
    next_dump = 0
    while next_dump <= args.runtime:
        if args.export in ("csv", "all"):
            for nid, node in net.nodes.items():
                w_ts.writerow([next_dump, nid, node.is_sybil, score_fn(nid)])

        step_to = min(next_dump + args.sample, args.runtime)
        if step_to > next_dump:
            net.run_simulation(until=step_to)  # prints once per chunk
        next_dump = step_to
        score_fn = make_score()               # reset caches

    if args.export in ("csv", "all"):
        f_ts.close()

    # ------------- final snapshot CSV ------------- #
    if args.export in ("csv", "all"):
        with csv_fin.open("w", newline="") as f_fin:
            w_fin = csv.writer(f_fin)
            w_fin.writerow(["node", "is_sybil", "score"])
            for nid, node in net.nodes.items():
                w_fin.writerow([nid, node.is_sybil, score_fn(nid)])

    # ------------- quick histogram ------------- #
    if args.export in ("png", "all"):
        scores = {nid: score_fn(nid) for nid in net.nodes}
        plt.figure()
        honest = [s for nid, s in scores.items() if not net.nodes[nid].is_sybil]
        sybil  = [s for nid, s in scores.items() if     net.nodes[nid].is_sybil]
        plt.hist([honest, sybil], bins=25, density=True, alpha=.7,
                 label=["Honest", "Sybil"])
        plt.xlabel("Detector score"); plt.ylabel("PDF")
        plt.title(f"{args.method.capitalize()} distribution "
                  f"({args.nodes}H+{args.sybil}S){' (weighted)' if suffix else ''}")
        plt.legend(); plt.tight_layout(); plt.savefig(png_out, dpi=300); plt.close()


# --------------------------------------------------------------------------- #
# Cheat‑sheet commands  (IDs E1 – E6)
# --------------------------------------------------------------------------- #
"""
E1 – Bayesian  (runtime 15, log every step)
python -m src.modules.driver --method bayes --nodes 10  --sybil 3  \
       --runtime 15 --sample 1 --export all

E2 – EigenTrust (runtime 30, log every 5)
python -m src.modules.driver --method eigen --nodes 50  --sybil 10 \
       --runtime 30 --sample 5 --export all

E3 – Random‑walk UNweighted (runtime 50, sample 5)
python -m src.modules.driver --method walk  --nodes 100 --sybil 30 \
       --runtime 50 --walks 10000 --sample 5 --export all

E4 – Random‑walk TRUST‑weighted (runtime 50, sample 5)
python -m src.modules.driver --method walk  --nodes 100 --sybil 30 \
       --runtime 50 --walks 10000 --weighted --sample 5 --export all

E5 – Bayesian scalability (runtime 75, sample 10)
python -m src.modules.driver --method bayes --nodes 250 --sybil 50 \
       --runtime 75 --sample 10 --export all

E6 – EigenTrust large net (runtime 100, sample 10)
python -m src.modules.driver --method eigen --nodes 500 --sybil 150 \
       --runtime 100 --sample 10 --export all
"""
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
