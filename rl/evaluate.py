"""Evaluate a trained checkpoint against the opponent pool.

Runs deterministic-action games against each opponent for N seeds and
reports per-opponent win rates. Use this to gate "best.pt" promotion.

Usage:
  python -m rl.evaluate --checkpoint checkpoints/latest.pt --games 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from agents.rl_inference import make_agent
from eval._quiet import make


def play_one(rl_agent, opponent_spec, seed: int, *, n_players: int = 2, extras=None) -> int:
    """Return 1 for win, 0 for loss, -1 for draw (or non-strict win)."""
    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    agents = [rl_agent] + [opponent_spec] * (n_players - 1)
    if extras is not None:
        idx = 0
        for i in range(1, n_players):
            if idx < len(extras):
                agents[i] = extras[idx]
                idx += 1
    env.run(agents)
    rewards = [s.reward for s in env.steps[-1]]
    if rewards[0] >= 0.99:
        return 1
    if rewards[0] <= -0.99:
        return 0
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--n-players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--opponents", nargs="*", default=None,
                       help="If omitted, evaluate against all registered opponents.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rl_agent = make_agent(Path(args.checkpoint), deterministic=True)

    from opponents import REGISTRY as OPP_REGISTRY
    opp_names = args.opponents or sorted(OPP_REGISTRY.keys())

    results: dict[str, dict] = {}
    print(f"Evaluating {args.checkpoint} ({args.n_players}p) — {args.games} games per opponent")
    print(f"{'Opponent':<22s} {'W':>4s} {'L':>4s} {'D':>4s} {'WR':>7s}  time")
    print("-" * 60)
    t0 = time.time()
    for name in opp_names:
        if name == "do_nothing" and args.n_players == 2:
            # do_nothing is the floor; skip for speed unless 4p
            pass
        spec = OPP_REGISTRY[name]
        extras = None
        if args.n_players == 4:
            # Same opponent fills all 3 enemy seats
            extras = [spec, spec]
        w = l = d = 0
        op_t0 = time.time()
        for s in range(args.games):
            r = play_one(rl_agent, spec, seed=s + 1000, n_players=args.n_players, extras=extras)
            if r == 1:
                w += 1
            elif r == 0:
                l += 1
            else:
                d += 1
        op_t = time.time() - op_t0
        wr = (w + 0.5 * d) / args.games
        results[name] = {"wins": w, "losses": l, "draws": d, "games": args.games, "wr": wr, "duration_s": op_t}
        print(f"{name:<22s} {w:>4d} {l:>4d} {d:>4d} {wr:>6.1%}  {op_t:.1f}s")

    total_w = sum(r["wins"] for r in results.values())
    total_g = sum(r["games"] for r in results.values())
    print("-" * 60)
    print(f"Overall: {total_w}/{total_g} = {total_w/total_g:.1%}  (total {time.time()-t0:.1f}s)")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
