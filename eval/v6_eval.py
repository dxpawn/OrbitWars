"""Decision-grade evaluation of heuristic_v6 (forward-sim brain port).

Compares v6 against our previous best (v5) and the target (adv_heuristic1000):
  - 2p head-to-head: v6 vs v5, v6 vs adv_heuristic1000
  - 4p FFA win-share at a HELD-OUT offset: v6, v5, adv_heuristic1000 on identical
    games (paired), vs DEFAULT_POOL.

  python -m eval.v6_eval --games 200 --offset 300000 --workers 48
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


def _h2h_worker(args):
    a, b, i = args
    from eval.match import run_match
    if i % 2 == 0:
        r = run_match(a, b, seed=30_000 + i)
        return (r.winner == 0), (r.winner == -1)
    r = run_match(b, a, seed=30_000 + i)
    return (r.winner == 1), (r.winner == -1)


def h2h(a, b, games, workers):
    jobs = [(a, b, i) for i in range(games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        res = p.map(_h2h_worker, jobs)
    w = sum(1 for x, _ in res if x)
    nw = sum(1 for _, n in res if n)
    lo, hi = wilson95(w, games)
    print(f"  {a} vs {b}: {a} {w}/{games} = {w/games:.1%}  "
          f"CI=[{lo:.1%},{hi:.1%}]  no-winner={nw}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--offset", type=int, default=300_000)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    g, o, w = args.games, args.offset, args.workers

    print("=== 2p head-to-head (alt seats) ===", flush=True)
    t = time.time()
    h2h("heuristic_v6", "heuristic_v5", g, w)
    h2h("heuristic_v6", "adv_heuristic1000", g, w)
    print(f"  (2p block {time.time()-t:.0f}s)", flush=True)

    print(f"\n=== 4p FFA win-share | held-out offset {o} | identical games ===", flush=True)
    vecs = {}
    for hero in ("heuristic_v5", "heuristic_v6", "adv_heuristic1000"):
        t = time.time()
        vec = run_ffa(hero, g, workers=w, offset=o)
        vecs[hero] = vec
        wins = sum(vec)
        lo, hi = wilson95(wins, g)
        print(f"  {hero:22s} {wins:3d}/{g} = {wins/g:5.1%}  "
              f"CI=[{lo:.1%},{hi:.1%}]  ({time.time()-t:.0f}s)", flush=True)
    # paired v6 vs v5
    a6, a5 = vecs["heuristic_v6"], vecs["heuristic_v5"]
    only6 = sum(1 for x, y in zip(a6, a5) if x and not y)
    only5 = sum(1 for x, y in zip(a6, a5) if y and not x)
    print(f"  paired v6-vs-v5: v6-only {only6}, v5-only {only5}, net {only6-only5:+d}", flush=True)


if __name__ == "__main__":
    main()
