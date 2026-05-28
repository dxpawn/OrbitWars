"""Parallel benchmark harness.

Runs many matches across cores via multiprocessing. Supports head-to-head
(A vs B with both seatings) and round-robin tournaments across an agent list.

Agents are passed by name (resolved via opponents.REGISTRY or env built-ins).
Function references can't be pickled across processes reliably on Windows;
names are.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from eval.match import MatchResult, run_match


@dataclass
class H2HResult:
    a: str
    b: str
    games: int
    a_wins: int
    b_wins: int
    draws: int  # ties OR no-winner (everyone eliminated with score 0)
    avg_duration_s: float
    matches: list[MatchResult] = field(default_factory=list)

    @property
    def a_win_rate(self) -> float:
        return self.a_wins / max(1, self.games)

    @property
    def wilson_95(self) -> tuple[float, float]:
        """Wilson 95% CI on A's win rate (treating draws as 0.5 wins)."""
        n = max(1, self.games)
        p = (self.a_wins + 0.5 * self.draws) / n
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, center - margin), min(1.0, center + margin)


def _worker_one_match(args):
    """Single match (a, b, seed). Used by multiprocessing.Pool.map."""
    a, b, seed, keep_replay = args
    return run_match(a, b, seed=seed, keep_replay=keep_replay)


def head_to_head(
    a: str,
    b: str,
    n_games: int,
    *,
    both_seatings: bool = True,
    workers: int | None = None,
    keep_replays: bool = False,
    seed_offset: int = 0,
) -> H2HResult:
    """Play `n_games` games between agents `a` and `b`.

    If `both_seatings` is True, half the games have A as P0 and half as P1
    (rounded up). Returns aggregate stats from A's perspective.
    """
    if workers is None:
        # Windows multiprocessing's WaitForMultipleObjects is capped at 63
        # handles. Stay well under that limit even on big machines.
        workers = max(1, min(32, (os.cpu_count() or 4) - 1))

    if both_seatings:
        half = n_games // 2
        rest = n_games - half
        jobs = []
        for i in range(half):
            jobs.append((a, b, seed_offset + i, keep_replays))
        for i in range(rest):
            jobs.append((b, a, seed_offset + half + i, keep_replays))
    else:
        jobs = [(a, b, seed_offset + i, keep_replays) for i in range(n_games)]

    t0 = time.time()
    if workers == 1:
        results = [_worker_one_match(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(_worker_one_match, jobs)
    total_duration = time.time() - t0

    a_wins = b_wins = draws = 0
    for (ag_a, ag_b, _seed, _kr), r in zip(jobs, results):
        # In job (ag_a, ag_b, ...), ag_a was seated at index 0.
        # We want to count wins from the *user's* `a` perspective.
        a_is_p0 = (ag_a == a)
        if r.winner == -1:
            draws += 1
            continue
        winner_is_a = (a_is_p0 and r.winner == 0) or (not a_is_p0 and r.winner == 1)
        if winner_is_a:
            a_wins += 1
        else:
            b_wins += 1

    return H2HResult(
        a=a,
        b=b,
        games=n_games,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        avg_duration_s=total_duration / max(1, n_games),
        matches=results if keep_replays else [],
    )


@dataclass
class RRResult:
    agents: list[str]
    h2h: dict[tuple[str, str], H2HResult]


def round_robin(
    agents: Sequence[str],
    games_per_pair: int,
    *,
    workers: int | None = None,
    seed_offset: int = 0,
) -> RRResult:
    """Round-robin: every unordered pair plays `games_per_pair` games."""
    pairings = list(combinations(agents, 2))
    h2h = {}
    base = seed_offset
    for i, (a, b) in enumerate(pairings):
        res = head_to_head(
            a, b, games_per_pair,
            workers=workers, seed_offset=base + i * games_per_pair,
        )
        h2h[(a, b)] = res
    return RRResult(agents=list(agents), h2h=h2h)


def summarize_h2h(r: H2HResult) -> str:
    lo, hi = r.wilson_95
    return (
        f"{r.a:>20s} vs {r.b:<20s}  "
        f"{r.a_wins:>3d}-{r.b_wins:<3d} (D:{r.draws})  "
        f"WR={r.a_win_rate:.1%}  CI95=[{lo:.2f},{hi:.2f}]  "
        f"{r.avg_duration_s:.2f}s/game"
    )


if __name__ == "__main__":
    # Smoke test: nearest_sniper vs random, 10 games
    print("Running 10-game smoke benchmark: nearest_sniper vs random ...")
    r = head_to_head("nearest_sniper", "random", n_games=10)
    print(summarize_h2h(r))
