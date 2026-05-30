"""General 2p head-to-head runner (alternating seats, paired by index).

  python -m eval.h2h --a heuristic_v6 --b heuristic_v5 --games 100
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from eval.ffa4 import wilson95  # noqa: E402


def _w(args):
    a, b, i = args
    from eval.match import run_match
    if i % 2 == 0:
        r = run_match(a, b, seed=30_000 + i)
        a_won = (r.winner == 0)
    else:
        r = run_match(b, a, seed=30_000 + i)
        a_won = (r.winner == 1)
    return a_won, (r.winner == -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    jobs = [(args.a, args.b, i) for i in range(args.games)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as p:
        res = p.map(_w, jobs)
    w = sum(1 for a_won, _ in res if a_won)
    nw = sum(1 for _, n in res if n)
    lo, hi = wilson95(w, args.games)
    print(f"{args.a} vs {args.b} (2p, alt seats): {args.a} {w}/{args.games} = "
          f"{w/args.games:.1%}  CI=[{lo:.1%},{hi:.1%}]  no-winner={nw}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
