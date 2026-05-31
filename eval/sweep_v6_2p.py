"""2p-aggression sweep for v6, measured by head-to-head win-rate vs a strong
opponent (default adv_heuristic1000 — the agent we're chasing in 2p), paired by
seed (alternating seats). The diagnostic (eval.diag_2p) showed v6 stalls in the
2p MIDGAME (steps 75-150); these knobs test fixes: longer reach, more aggression.

SCREEN only. Confirm any winner vs a DIFFERENT opponent (e.g. --opp heuristic_v5)
to ensure it's a real 2p improvement, not overfit to one opponent.

  python -m eval.sweep_v6_2p --games 160 --opp adv_heuristic1000
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from eval.ffa4 import wilson95  # noqa: E402

V6_KEYS = ["V6_FWD_HORIZON", "V6_EMIT_FRAC", "V6_PLANET_W", "V6_PROD_W",
           "V6_MIN_GAIN", "V6_MAX_ACTIONS", "V6_MAXDIST_2P",
           "V6_OVERSEND_2P", "V6_PRESS_2P", "V6_PRESS_MAX", "V6_PRESS_MIN_PROD",
           "V6_DEF_FRAC", "V6_HAMMER", "V6_HAMMER_OVERKILL", "V6_HAMMER_PROD_LEAD",
           "V6_HAMMER_MIN_CONTRIB", "V6_HAMMER_MAX_TRAVEL"]

# Round 1 (N=60) — cheap tactical knobs are dead: oversend +0, press +0 (inert),
# def_frac -3. Round 2 = the persistent STAGGERED HAMMER (V6_HAMMER): multi-turn
# plans that land a combined fleet on ONE turn (cross-turn memory). Test the
# default + a couple of aggression/selectivity variants, paired vs H1000.
CONFIGS: list[tuple[str, dict]] = [
    ("baseline (v6 2p)", {}),
    ("hammer", {"V6_HAMMER": "1"}),
    ("hammer ovk1.5", {"V6_HAMMER": "1", "V6_HAMMER_OVERKILL": "1.5"}),
    ("hammer lead4", {"V6_HAMMER": "1", "V6_HAMMER_PROD_LEAD": "4"}),
    ("hammer contrib20", {"V6_HAMMER": "1", "V6_HAMMER_MIN_CONTRIB": "20"}),
]

_OPP = "adv_heuristic1000"


def _w(args):
    a, b, i = args
    from eval.match import run_match
    if i % 2 == 0:
        r = run_match(a, b, seed=30_000 + i)
        return (r.winner == 0)
    r = run_match(b, a, seed=30_000 + i)
    return (r.winner == 1)


def run_h2h(games, workers):
    jobs = [("heuristic_v6", _OPP, i) for i in range(games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        return p.map(_w, jobs)


def _clear():
    for k in V6_KEYS:
        os.environ.pop(k, None)


def main():
    global _OPP
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=160)
    ap.add_argument("--opp", default="adv_heuristic1000")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    _OPP = args.opp

    base_vec = None
    rows = []
    for label, ov in CONFIGS:
        _clear()
        for k, v in ov.items():
            os.environ[k] = v
        vec = run_h2h(args.games, args.workers)
        wins = sum(vec)
        lo, hi = wilson95(wins, args.games)
        if label.startswith("baseline"):
            base_vec = vec
            paired = "(base)"
        else:
            only = sum(1 for c, b in zip(vec, base_vec) if c and not b)
            bonly = sum(1 for c, b in zip(vec, base_vec) if b and not c)
            paired = f"+{only}/-{bonly} (net {only - bonly:+d})"
        rows.append((label, wins, args.games, lo, hi, paired))
        print(f"  done: {label:18s} {wins}/{args.games} = {wins/args.games:.1%}  paired {paired}", flush=True)

    print(f"\n=== v6 2p-aggression sweep vs {args.opp} (paired by seed) ===")
    base_wr = rows[0][1] / rows[0][2]
    for label, wins, n, lo, hi, paired in rows:
        wr = wins / n
        flag = ""
        if not label.startswith("baseline"):
            if wr > base_wr + 0.05:
                flag = "  <== promising"
            elif wr < base_wr - 0.05:
                flag = "  (worse)"
        print(f"{label:18s} {wr:>6.1%}  [{lo:>5.1%},{hi:>5.1%}]  {paired}{flag}")
    print(f"\nbaseline {base_wr:.1%} vs {args.opp}. Confirm winners vs a DIFFERENT --opp.")


if __name__ == "__main__":
    main()
