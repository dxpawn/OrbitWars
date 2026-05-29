"""One-at-a-time parameter sweep for heuristic_tune, gated on 4p FFA win-share.

heuristic_tune == heuristic_v2 when no HB_* env vars are set. This sweep varies
ONE constant at a time from v2's defaults and measures 4p FFA win-share vs the
strong public pool on IDENTICAL games (reproducible by index), so each config is
a paired A/B against the baseline.

This is a SCREEN, not a verdict. Any apparent winner must be re-confirmed on a
held-out seed range (offset) with more games before it is taken seriously —
small pools overfit (see diary: 3 prior tweaks failed to transfer to Kaggle).

Usage:
  python -m eval.sweep --games 200
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from eval.ffa4 import run_ffa, wilson95  # noqa: E402

HB_KEYS = [
    "HB_EARLY_ROUNDS", "HB_EARLY_LOOK_AHEAD", "HB_MAX_DISTANCE",
    "HB_ROTATION_LOOK_AHEAD", "HB_REINFORCEMENT_SIZE", "HB_GARRISON_SIZE",
]

# (label, {env overrides}). Empty dict = baseline (== v2).
# Confirmation set: refine the two levers the screen flagged (MAX_DISTANCE down,
# EARLY_ROUNDS up) + their combos. Run on a HELD-OUT offset to test for real.
CONFIGS: list[tuple[str, dict]] = [
    ("baseline (v2)", {}),
    ("MAX_DISTANCE=28", {"HB_MAX_DISTANCE": "28"}),
    ("MAX_DISTANCE=30", {"HB_MAX_DISTANCE": "30"}),
    ("MAX_DISTANCE=32", {"HB_MAX_DISTANCE": "32"}),
    ("MAX_DISTANCE=34", {"HB_MAX_DISTANCE": "34"}),
    ("EARLY_ROUNDS=5", {"HB_EARLY_ROUNDS": "5"}),
    ("MD=32+ER=5", {"HB_MAX_DISTANCE": "32", "HB_EARLY_ROUNDS": "5"}),
    ("MD=30+ER=5", {"HB_MAX_DISTANCE": "30", "HB_EARLY_ROUNDS": "5"}),
]


def _clear_env():
    for k in HB_KEYS:
        os.environ.pop(k, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    base_vec = None
    rows = []
    for label, overrides in CONFIGS:
        _clear_env()
        for k, v in overrides.items():
            os.environ[k] = v
        vec = run_ffa("heuristic_tune", args.games, workers=args.workers, offset=args.offset)
        wins = sum(vec)
        lo, hi = wilson95(wins, args.games)
        if label.startswith("baseline"):
            base_vec = vec
            paired = "—"
        else:
            # paired delta vs baseline on identical games
            cfg_only = sum(1 for c, b in zip(vec, base_vec) if c and not b)
            base_only = sum(1 for c, b in zip(vec, base_vec) if b and not c)
            paired = f"+{cfg_only}/-{base_only} (net {cfg_only - base_only:+d})"
        rows.append((label, wins, args.games, lo, hi, paired))
        print(f"  done: {label:24s} {wins}/{args.games} = {wins/args.games:.1%}  paired {paired}", flush=True)

    print("\n=== sweep results (4p FFA, identical games) ===")
    print(f"{'config':24s} {'win%':>7s}  {'CI95':>14s}  paired(vs base)")
    base_wr = rows[0][1] / rows[0][2]
    for label, wins, n, lo, hi, paired in rows:
        wr = wins / n
        flag = ""
        if not label.startswith("baseline"):
            if wr > base_wr + 0.06:
                flag = "  <== promising"
            elif wr < base_wr - 0.06:
                flag = "  (worse)"
        print(f"{label:24s} {wr:>6.1%}  [{lo:>5.1%},{hi:>5.1%}]  {paired}{flag}")
    print(f"\nbaseline win% = {base_wr:.1%}. Promising configs must be CONFIRMED on a "
          f"held-out offset (e.g. --offset 100000) with more games before trusting.")


if __name__ == "__main__":
    main()
