"""Simple win-rate tracker for the opponent pool.

Persists `{agent_name: {wins, losses, draws, games, opponents: {...}}}` to
ratings.json. Round-robin "strength" is just total win rate against the pool.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from eval.bench import H2HResult


RATINGS_PATH = Path("ratings.json")


def load_ratings(path: Path = RATINGS_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_ratings(data: dict, path: Path = RATINGS_PATH) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _entry() -> dict:
    return {"wins": 0, "losses": 0, "draws": 0, "games": 0, "opponents": {}}


def _ensure(data: dict, name: str) -> dict:
    if name not in data:
        data[name] = _entry()
    return data[name]


def record_h2h(data: dict, r: H2HResult) -> None:
    """Update ratings.json in place with the results from one H2H."""
    a = _ensure(data, r.a)
    b = _ensure(data, r.b)

    a["wins"] += r.a_wins
    a["losses"] += r.b_wins
    a["draws"] += r.draws
    a["games"] += r.games

    b["wins"] += r.b_wins
    b["losses"] += r.a_wins
    b["draws"] += r.draws
    b["games"] += r.games

    # Per-opponent breakdown
    ao = a["opponents"].setdefault(r.b, _entry())
    ao["wins"] += r.a_wins
    ao["losses"] += r.b_wins
    ao["draws"] += r.draws
    ao["games"] += r.games

    bo = b["opponents"].setdefault(r.a, _entry())
    bo["wins"] += r.b_wins
    bo["losses"] += r.a_wins
    bo["draws"] += r.draws
    bo["games"] += r.games


def win_rate(entry: dict) -> float:
    if entry["games"] == 0:
        return 0.0
    return (entry["wins"] + 0.5 * entry["draws"]) / entry["games"]


def leaderboard(data: dict) -> list[tuple[str, dict, float]]:
    rows = [(name, e, win_rate(e)) for name, e in data.items()]
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def best_agent(data: dict) -> str | None:
    """Return the highest win-rate agent, or None if no data."""
    rows = leaderboard(data)
    if not rows:
        return None
    return rows[0][0]


def print_leaderboard(data: dict) -> None:
    rows = leaderboard(data)
    if not rows:
        print("(no ratings yet)")
        return
    print(f"{'Agent':<25s} {'Games':>7s} {'W':>5s} {'L':>5s} {'D':>5s} {'WR':>7s}")
    print("-" * 60)
    for name, e, wr in rows:
        print(f"{name:<25s} {e['games']:>7d} {e['wins']:>5d} {e['losses']:>5d} "
              f"{e['draws']:>5d} {wr:>6.1%}")
