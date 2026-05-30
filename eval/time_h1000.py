"""Per-turn timing + robustness audit for HEURISTIC1000.

It does heavy forward-simulation; a single turn over actTimeout=1.0s is a
forfeit on Kaggle. This wraps its agent to time every call and catch every
exception, then runs a full 2p game and a full 4p game (vs heuristic_v5 to
generate a busy, realistic board) and reports max/mean/p95 per-turn ms.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import time
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from eval.match import run_match  # noqa: E402

_PATH = os.path.join(os.path.dirname(__file__), "..", "other_adversaries", "HEURISTIC1000.py")
_spec = importlib.util.spec_from_file_location("h1000_mod", os.path.abspath(_PATH))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_times: list[float] = []
_excs: list[str] = []


def timed_agent(obs, config=None):
    t = time.perf_counter()
    try:
        r = _mod.agent(obs, config)
    except Exception as e:  # noqa: BLE001
        _excs.append(repr(e))
        r = []
    _times.append((time.perf_counter() - t) * 1000.0)
    return r


def _report(tag):
    if not _times:
        print(f"{tag}: no turns recorded")
        return
    ts = sorted(_times)
    n = len(ts)
    mean = sum(ts) / n
    p95 = ts[int(0.95 * (n - 1))]
    mx = ts[-1]
    over = sum(1 for x in ts if x > 1000.0)
    print(f"{tag}: turns={n} mean={mean:.0f}ms p95={p95:.0f}ms max={mx:.0f}ms "
          f">1000ms={over} exceptions={len(_excs)}")
    if _excs:
        print(f"   first exc: {_excs[0]}")


def main():
    global _times, _excs
    # 2p: timed h1000 vs heuristic_v5
    _times, _excs = [], []
    run_match(timed_agent, "heuristic_v5", seed=12345)
    _report("2p (h1000 vs v5)")

    # 4p: timed h1000 vs 3x heuristic_v5
    _times, _excs = [], []
    run_match(timed_agent, "heuristic_v5", seed=23456,
              extra_agents=["heuristic_v5", "heuristic_v5"])
    _report("4p (h1000 vs 3x v5)")


if __name__ == "__main__":
    main()
