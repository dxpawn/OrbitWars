"""Diagnose WHERE v6 loses a 2p game vs adv_heuristic1000.

Tracks each player's planet count / total production / total ships over time,
averaged across several seeds (v6 = seat 0). If v6 trails on production/planets
by midgame -> expansion/economy problem. If it keeps pace but trails on ships /
loses planets late -> combat/tactics problem.

  python -m eval.diag_2p --games 6
"""
from __future__ import annotations

import argparse
import logging
import warnings
from collections import defaultdict

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from eval._quiet import make  # noqa: E402
from eval.match import resolve_agent  # noqa: E402

SNAPS = [25, 50, 75, 100, 150, 200, 250, 300, 400]


def trajectory(a, b, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([resolve_agent(a), resolve_agent(b)])
    rows = {}
    for idx, step in enumerate(env.steps):
        obs = step[0].observation
        planets = defaultdict(int)
        prod = defaultdict(int)
        ships = defaultdict(int)
        for p in obs["planets"]:
            o = p[1]
            if o != -1:
                planets[o] += 1
                ships[o] += p[5]
                prod[o] += p[6]
        for f in obs["fleets"]:
            ships[f[1]] += f[6]
        rows[idx] = (planets, prod, ships)
    winner_scores = env.steps[-1]
    return rows, len(env.steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="heuristic_v6")
    ap.add_argument("--b", default="adv_heuristic1000")
    ap.add_argument("--games", type=int, default=6)
    args = ap.parse_args()

    # accumulate per-snapshot averages
    agg = {s: {"pl0": 0.0, "pl1": 0.0, "pr0": 0.0, "pr1": 0.0, "sh0": 0.0, "sh1": 0.0, "n": 0}
           for s in SNAPS}
    for g in range(args.games):
        rows, nsteps = trajectory(args.a, args.b, seed=4000 + g)
        for s in SNAPS:
            if s >= nsteps:
                continue
            planets, prod, ships = rows[s]
            agg[s]["pl0"] += planets.get(0, 0); agg[s]["pl1"] += planets.get(1, 0)
            agg[s]["pr0"] += prod.get(0, 0); agg[s]["pr1"] += prod.get(1, 0)
            agg[s]["sh0"] += ships.get(0, 0); agg[s]["sh1"] += ships.get(1, 0)
            agg[s]["n"] += 1

    print(f"=== 2p trajectory: {args.a}(seat0) vs {args.b}(seat1), {args.games} games ===")
    print(f"{'step':>5} | {'planets a/b':>12} | {'prod a/b':>12} | {'ships a/b':>15}")
    for s in SNAPS:
        d = agg[s]
        if d["n"] == 0:
            continue
        n = d["n"]
        print(f"{s:>5} | {d['pl0']/n:5.1f}/{d['pl1']/n:<5.1f} | "
              f"{d['pr0']/n:5.1f}/{d['pr1']/n:<5.1f} | "
              f"{d['sh0']/n:6.0f}/{d['sh1']/n:<6.0f}  (n={n})")


if __name__ == "__main__":
    main()
