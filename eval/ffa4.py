"""4-player FFA evaluator (parallel).

Measures a hero agent's win-share in 4-player free-for-all against a diverse
opponent pool — the format the Kaggle competition actually scores. Only the
single highest-score player wins (ties => no winner).

Reproducible: each game's opponent lineup + hero seat are derived purely from
the game index, so two different heroes evaluated with the same --games and
--pool play IDENTICAL conditions. That lets you A/B a candidate vs v2 fairly.

Usage:
  python -m eval.ffa4 --hero heuristic_v2 --games 200
  python -m eval.ffa4 --hero heuristic_v2 --games 200 --pool adv_hellburner adv_ver16 adv_proto_v15 adv_lb958 adv_in_progress
"""
from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import os
import random
import time
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

DEFAULT_POOL = [
    "adv_hellburner",
    "adv_ver16",
    "adv_proto_v15",
    "adv_lb958",
    "adv_in_progress",
]


def _lineup(game_idx: int, pool: list[str]) -> tuple[int, list[str]]:
    """Deterministic (hero_seat, [3 opponents]) for this game index."""
    rng = random.Random(game_idx * 7919 + 13)
    seat = rng.randrange(4)
    opps = rng.sample(pool, 3)
    return seat, opps


def _worker(args):
    hero, game_idx, pool = args
    from eval.match import run_match

    seat, opps = _lineup(game_idx, pool)
    four = list(opps)
    four.insert(seat, hero)  # hero at `seat`, opponents fill the rest
    r = run_match(four[0], four[1], seed=10_000 + game_idx, extra_agents=four[2:])
    hero_won = (r.winner == seat)
    # also report whether hero was eliminated to 0 (winner==-1 doesn't tell us)
    return hero_won, r.winner == -1


def run_ffa(hero: str, games: int, pool: list[str] | None = None,
            workers: int | None = None, offset: int = 0) -> list[bool]:
    """Run `games` 4p FFA games; return per-game [hero_won] (reproducible by index).

    Lineups/seats/seeds depend only on the game index, so two heroes run with the
    same (games, pool, offset) play IDENTICAL games — enabling a paired A/B.
    `offset` shifts the index range, e.g. offset=10000 gives a held-out set.
    """
    pool = pool or DEFAULT_POOL
    workers = workers or max(1, min(32, (os.cpu_count() or 4) - 2))
    jobs = [(hero, offset + i, pool) for i in range(games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        results = p.map(_worker, jobs)
    return [w for w, _ in results]


def wilson95(wins: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--pool", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    pool = args.pool or DEFAULT_POOL
    workers = args.workers or max(1, min(32, (os.cpu_count() or 4) - 2))

    jobs = [(args.hero, args.offset + i, pool) for i in range(args.games)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        results = p.map(_worker, jobs)
    dt = time.time() - t0

    wins = sum(1 for w, _ in results if w)
    no_winner = sum(1 for _, nw in results if nw)
    lo, hi = wilson95(wins, args.games)
    print(f"=== 4p FFA | hero={args.hero} | games={args.games} | workers={workers} ===")
    print(f"pool: {pool}")
    print(f"hero wins   : {wins}/{args.games} = {wins/args.games:.1%}  CI95=[{lo:.1%},{hi:.1%}]")
    print(f"no-winner   : {no_winner}/{args.games} = {no_winner/args.games:.1%}")
    print(f"random floor: 25.0%")
    print(f"elapsed     : {dt:.0f}s ({dt/args.games:.2f}s/game)")


if __name__ == "__main__":
    main()
