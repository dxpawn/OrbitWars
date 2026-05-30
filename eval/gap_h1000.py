"""Quantify the strength gap between adv_heuristic1000 (user reports 1000-1100 LB)
and our shipped heuristic_v5 (Kaggle 970), in our own reproducible harness.

Two complementary measurements:
  1. 4p FFA win-share, paired: each hero plays the SAME games vs the DEFAULT_POOL
     (format Kaggle scores). Higher win-share = stronger.
  2. 2p head-to-head, alternating seats: direct A-vs-B.

Usage:
  python -m eval.gap_h1000 --games 200 --workers 48 --offset 0
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from eval.ffa4 import run_ffa, wilson95  # noqa: E402

A = "adv_heuristic1000"
B = "heuristic_v5"


def _h2h_worker(args):
    a, b, idx = args
    from eval.match import run_match
    # Alternate seats so neither agent gets a fixed positional advantage.
    if idx % 2 == 0:
        r = run_match(a, b, seed=20_000 + idx)
        a_won = (r.winner == 0)
    else:
        r = run_match(b, a, seed=20_000 + idx)
        a_won = (r.winner == 1)
    return a_won, (r.winner == -1)


def run_h2h(a, b, games, workers):
    jobs = [(a, b, i) for i in range(games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        return p.map(_h2h_worker, jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    print("=== 4p FFA win-share | identical games | DEFAULT_POOL ===", flush=True)
    ffa = {}
    for hero in (B, A):
        t = time.time()
        vec = run_ffa(hero, args.games, workers=args.workers, offset=args.offset)
        ffa[hero] = vec
        w = sum(vec)
        lo, hi = wilson95(w, args.games)
        print(f"  {hero:22s} {w:3d}/{args.games} = {w/args.games:5.1%}  "
              f"CI=[{lo:.1%},{hi:.1%}]  ({time.time()-t:.0f}s)", flush=True)
    # paired on identical games
    a_only = sum(1 for x, y in zip(ffa[A], ffa[B]) if x and not y)
    b_only = sum(1 for x, y in zip(ffa[A], ffa[B]) if y and not x)
    print(f"  paired (same games): {A} won {a_only} that {B} lost; "
          f"{B} won {b_only} that {A} lost; net {a_only - b_only:+d}", flush=True)

    print("\n=== 2p head-to-head | alternating seats ===", flush=True)
    t = time.time()
    res = run_h2h(A, B, args.games, args.workers)
    w = sum(1 for a_won, _ in res if a_won)
    nw = sum(1 for _, n in res if n)
    lo, hi = wilson95(w, args.games)
    print(f"  {A} beats {B}: {w}/{args.games} = {w/args.games:.1%}  "
          f"CI=[{lo:.1%},{hi:.1%}]  (no-winner: {nw})  ({time.time()-t:.0f}s)", flush=True)
    print(f"\n(If {A} >> 50% h2h AND higher FFA win-share, the LB gap is real "
          f"and {A} is the stronger base.)")


if __name__ == "__main__":
    main()
