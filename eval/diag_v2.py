"""Diagnostic: surface silent exceptions and per-turn timing for heuristic_v2.

heuristic_v2.agent() wraps main() in `try/except Exception: return []`, so any
exception silently no-ops that turn. And actTimeout is 1.0s — a slow turn
forfeits. This script runs real games with an instrumented agent that:
  - calls Hellburner().main() WITHOUT swallowing exceptions (records them)
  - times every turn (wall clock)
and reports exception counts + timing percentiles. Pure diagnostics; does not
change the shipped agent.

Run single-process so timings reflect one core (matches Kaggle), not a
contended pool.

Usage:
  python -m eval.diag_v2 --games 12 --mode 2p
  python -m eval.diag_v2 --games 12 --mode 4p
"""
from __future__ import annotations

import argparse
import logging
import time
import traceback
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from eval._quiet import make  # noqa: E402
import opponents  # noqa: E402
from agents import heuristic_v2  # noqa: E402

# Collected across the process
TURN_TIMES: list[float] = []
EXCEPTIONS: list[str] = []
EMPTY_TURNS = 0
TOTAL_TURNS = 0


def instrumented_agent(obs, config=None):
    """Mirror heuristic_v2.agent() but record timing + exceptions."""
    global EMPTY_TURNS, TOTAL_TURNS
    TOTAL_TURNS += 1
    inst = heuristic_v2.Hellburner()
    t0 = time.perf_counter()
    try:
        moves = inst.main(obs)
    except Exception:
        EXCEPTIONS.append(traceback.format_exc())
        moves = []
    dt = time.perf_counter() - t0
    TURN_TIMES.append(dt)
    if not moves:
        EMPTY_TURNS += 1
    return moves


def pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--mode", choices=["2p", "4p"], default="2p")
    ap.add_argument("--opp", default="adv_hellburner")
    args = ap.parse_args()

    opp = opponents.get(args.opp)
    extras = None
    if args.mode == "4p":
        extras = [opponents.get("adv_ver16"), opponents.get("adv_proto_v15")]

    for g in range(args.games):
        env = make("orbit_wars", configuration={"seed": 1000 + g}, debug=False)
        agents = [instrumented_agent, opp]
        if extras:
            agents = [instrumented_agent, opp] + extras
        env.run(agents)

    print(f"=== diag heuristic_v2 | mode={args.mode} | games={args.games} | opp={args.opp} ===")
    print(f"total turns (our agent): {TOTAL_TURNS}")
    print(f"empty-move turns       : {EMPTY_TURNS} ({100*EMPTY_TURNS/max(1,TOTAL_TURNS):.1f}%)")
    print(f"exceptions thrown      : {len(EXCEPTIONS)}")
    if TURN_TIMES:
        print(f"turn time ms: mean={1000*sum(TURN_TIMES)/len(TURN_TIMES):.1f} "
              f"p50={1000*pct(TURN_TIMES,0.50):.1f} "
              f"p95={1000*pct(TURN_TIMES,0.95):.1f} "
              f"p99={1000*pct(TURN_TIMES,0.99):.1f} "
              f"max={1000*max(TURN_TIMES):.1f}")
        over = sum(1 for t in TURN_TIMES if t > 1.0)
        near = sum(1 for t in TURN_TIMES if t > 0.5)
        print(f"turns >1.0s (forfeit) : {over}")
        print(f"turns >0.5s (risky)   : {near}")
    if EXCEPTIONS:
        print("\n--- first exception ---")
        print(EXCEPTIONS[0])


if __name__ == "__main__":
    main()
