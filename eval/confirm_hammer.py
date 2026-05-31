"""Cross-opponent, held-out confirmation of the default hammer (V6_HAMMER=1) vs the
1017 baseline (hammer off). For each opponent we play the SAME seeds twice — once
with the hammer on, once off, seats alternating — and report the paired net. A real
improvement should be broadly >= 0 across opponents, not just vs a v6 mirror.

  python -m eval.confirm_hammer --games 120 --seedbase 200000
"""
from __future__ import annotations
import argparse, logging, os, warnings
import multiprocessing as mp
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")
from eval.ffa4 import wilson95

OPPS = ["heuristic_v2", "adv_hellburner", "adv_lb958", "adv_proto_v15", "adv_heuristic1000"]

def _w(args):
    hammer, opp, i, seedbase = args
    os.environ["V6_HAMMER"] = "1" if hammer else "0"
    from eval.match import run_match
    seed = seedbase + i
    if i % 2 == 0:
        r = run_match("heuristic_v6", opp, seed=seed); return r.winner == 0
    r = run_match(opp, "heuristic_v6", seed=seed); return r.winner == 1

def run(opp, games, workers, hammer, seedbase):
    jobs = [(hammer, opp, i, seedbase) for i in range(games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        return p.map(_w, jobs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--seedbase", type=int, default=200000)
    args = ap.parse_args()
    print(f"=== hammer(on) vs baseline(off), paired per opponent, seedbase {args.seedbase} ===")
    print(f"{'opponent':20s} {'base_wr':>8s} {'hammer_wr':>10s} {'paired':>16s}")
    for opp in OPPS:
        base = run(opp, args.games, args.workers, False, args.seedbase)
        ham  = run(opp, args.games, args.workers, True,  args.seedbase)
        bw, hw = sum(base), sum(ham)
        only = sum(1 for h, b in zip(ham, base) if h and not b)
        bonly = sum(1 for h, b in zip(ham, base) if b and not h)
        net = only - bonly
        flag = "  <== +" if net > 0 else ("  (worse)" if net < 0 else "")
        print(f"{opp:20s} {bw/args.games:>7.1%} {hw/args.games:>9.1%}  +{only}/-{bonly} (net {net:+d}){flag}", flush=True)

if __name__ == "__main__":
    main()
