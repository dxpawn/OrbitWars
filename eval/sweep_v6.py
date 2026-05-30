"""Parameter sweep for heuristic_v6's forward-sim brain, gated on 4p FFA win-share.

v6 with no V6_* env vars == its baked defaults. This varies ONE brain knob at a
time and measures 4p FFA win-share vs DEFAULT_POOL on IDENTICAL games (paired
A/B vs baseline). SCREEN only — confirm any winner on a held-out --offset.

  python -m eval.sweep_v6 --games 180 --offset 0
"""
from __future__ import annotations

import argparse
import logging
import os
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from eval.ffa4 import run_ffa, wilson95  # noqa: E402

V6_KEYS = ["V6_FWD_HORIZON", "V6_EMIT_FRAC", "V6_PLANET_W", "V6_PROD_W",
           "V6_MIN_GAIN", "V6_MAX_ACTIONS"]

# PEAK-FIND (3rd seed range, offset 700000). EMIT is the real lever: held-out
# +11 with a monotone 0.20->0.15->0.10 gradient. HORIZON weak + doesn't stack.
# Locate the EMIT optimum and triple-confirm; very low emit may turn naive
# (projection ignores snipe-back), so test below 0.10 too.
CONFIGS: list[tuple[str, dict]] = [
    ("baseline (v6)", {}),
    ("EMIT=0.05", {"V6_EMIT_FRAC": "0.05"}),
    ("EMIT=0.08", {"V6_EMIT_FRAC": "0.08"}),
    ("EMIT=0.10", {"V6_EMIT_FRAC": "0.10"}),
    ("EMIT=0.12", {"V6_EMIT_FRAC": "0.12"}),
]


def _clear():
    for k in V6_KEYS:
        os.environ.pop(k, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=180)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    base_vec = None
    rows = []
    for label, ov in CONFIGS:
        _clear()
        for k, v in ov.items():
            os.environ[k] = v
        vec = run_ffa("heuristic_v6", args.games, workers=args.workers, offset=args.offset)
        wins = sum(vec)
        lo, hi = wilson95(wins, args.games)
        if label.startswith("baseline"):
            base_vec = vec
            paired = "(base)"
        else:
            only = sum(1 for c, b in zip(vec, base_vec) if c and not b)
            bonly = sum(1 for c, b in zip(vec, base_vec) if b and not c)
            paired = f"+{only}/-{bonly} (net {only - bonly:+d})"
        rows.append((label, wins, args.games, lo, hi, paired))
        print(f"  done: {label:18s} {wins}/{args.games} = {wins/args.games:.1%}  paired {paired}", flush=True)

    print("\n=== v6 sweep (4p FFA, identical games) ===")
    base_wr = rows[0][1] / rows[0][2]
    for label, wins, n, lo, hi, paired in rows:
        wr = wins / n
        flag = ""
        if not label.startswith("baseline"):
            if wr > base_wr + 0.06:
                flag = "  <== promising"
            elif wr < base_wr - 0.06:
                flag = "  (worse)"
        print(f"{label:18s} {wr:>6.1%}  [{lo:>5.1%},{hi:>5.1%}]  {paired}{flag}")
    print(f"\nbaseline {base_wr:.1%}. Confirm promising configs on a held-out --offset.")


if __name__ == "__main__":
    main()
