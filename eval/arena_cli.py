"""CLI for the eval arena.

Usage:
  python -m eval.arena_cli h2h --a NAME --b NAME [--games N]
  python -m eval.arena_cli round-robin --games N
  python -m eval.arena_cli leaderboard
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations

from eval.bench import head_to_head, summarize_h2h
from eval.ratings import (
    RATINGS_PATH,
    load_ratings,
    print_leaderboard,
    record_h2h,
    save_ratings,
)


def cmd_h2h(args):
    r = head_to_head(args.a, args.b, args.games, workers=args.workers)
    print(summarize_h2h(r))
    if args.update:
        data = load_ratings()
        record_h2h(data, r)
        save_ratings(data)
        print(f"Updated {RATINGS_PATH}")


def cmd_round_robin(args):
    # Default pool: all registered opponents + any extras
    import opponents
    names = list(opponents.names())
    if args.include:
        for n in args.include:
            if n not in names:
                names.append(n)
    if args.exclude:
        names = [n for n in names if n not in args.exclude]

    print(f"Round-robin over {len(names)} agents, {args.games} games per pairing:")
    print(f"  {names}")
    print()

    data = load_ratings()
    pairings = list(combinations(names, 2))
    for i, (a, b) in enumerate(pairings):
        seed_off = i * args.games
        r = head_to_head(a, b, args.games, workers=args.workers, seed_offset=seed_off)
        print(summarize_h2h(r))
        if args.update:
            record_h2h(data, r)

    if args.update:
        save_ratings(data)
        print(f"\nUpdated {RATINGS_PATH}")
    print()
    print_leaderboard(data)


def cmd_leaderboard(_args):
    data = load_ratings()
    print_leaderboard(data)


def main(argv=None):
    p = argparse.ArgumentParser(description="Orbit Wars eval arena")
    sub = p.add_subparsers(dest="cmd", required=True)

    h2h = sub.add_parser("h2h", help="Head-to-head match")
    h2h.add_argument("--a", required=True)
    h2h.add_argument("--b", required=True)
    h2h.add_argument("--games", type=int, default=20)
    h2h.add_argument("--workers", type=int, default=None)
    h2h.add_argument("--update", action="store_true", help="Persist to ratings.json")
    h2h.set_defaults(func=cmd_h2h)

    rr = sub.add_parser("round-robin", help="Round-robin tournament")
    rr.add_argument("--games", type=int, default=20)
    rr.add_argument("--workers", type=int, default=None)
    rr.add_argument("--include", nargs="*", default=None)
    rr.add_argument("--exclude", nargs="*", default=None)
    rr.add_argument("--update", action="store_true", default=True)
    rr.set_defaults(func=cmd_round_robin)

    lb = sub.add_parser("leaderboard")
    lb.set_defaults(func=cmd_leaderboard)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
