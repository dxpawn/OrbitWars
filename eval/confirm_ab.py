"""Generic held-out A/B of a v6 env-config ('treatment') vs the 1017 baseline
(all knobs off), paired per opponent across a field. The discipline that caught
the hammer false-positive: a real win must hold vs DIFFERENT opponents on a
held-out seed base, not just vs a v6 mirror.

  python -m eval.confirm_ab --on "V6_SNAP_WEIGHT=1" --games 120 --seedbase 200000
  python -m eval.confirm_ab --on "V6_SNAP_WEIGHT=1,V6_ARR_DECAY=0.97"
"""
from __future__ import annotations
import argparse, logging, os, warnings
import multiprocessing as mp
logging.disable(logging.CRITICAL); warnings.filterwarnings("ignore")

OPPS = ["heuristic_v2", "adv_hellburner", "adv_lb958", "adv_proto_v15", "adv_heuristic1000"]

def _w(args):
    # treat is passed THROUGH the tuple (pickled to the spawned worker). Relying on
    # a module global set in main() would be EMPTY in spawn workers (main never runs
    # there) => silent no-op A/B. This was the +0/-0 bug.
    on, opp, i, seedbase, treat = args
    for k in treat:
        os.environ.pop(k, None)
    if on:
        for k, v in treat.items():
            os.environ[k] = v
    from eval.match import run_match
    seed = seedbase + i
    if i % 2 == 0:
        r = run_match("heuristic_v6", opp, seed=seed); return r.winner == 0
    r = run_match(opp, "heuristic_v6", seed=seed); return r.winner == 1

def run(opp, games, workers, on, seedbase, treat):
    jobs = [(on, opp, i, seedbase, treat) for i in range(games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        return p.map(_w, jobs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="comma list KEY=VAL for the treatment")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--seedbase", type=int, default=200000)
    ap.add_argument("--opps", default=",".join(OPPS))
    args = ap.parse_args()
    treat = dict(kv.split("=", 1) for kv in args.on.split(","))
    opps = args.opps.split(",")
    print(f"=== treatment {treat} vs baseline(off), paired per opp, seedbase {args.seedbase} ===")
    print(f"{'opponent':20s} {'base_wr':>8s} {'treat_wr':>9s} {'paired':>17s}")
    nets = []
    for opp in opps:
        base = run(opp, args.games, args.workers, False, args.seedbase, treat)
        treat_res = run(opp, args.games, args.workers, True, args.seedbase, treat)
        bw, tw = sum(base), sum(treat_res)
        only = sum(1 for t, b in zip(treat_res, base) if t and not b)
        bonly = sum(1 for t, b in zip(treat_res, base) if b and not t)
        net = only - bonly; nets.append(net)
        flag = "  <== +" if net > 0 else ("  (worse)" if net < 0 else "")
        print(f"{opp:20s} {bw/args.games:>7.1%} {tw/args.games:>8.1%}  +{only}/-{bonly} (net {net:+d}){flag}", flush=True)
    print(f"\nSUM net across {len(opps)} opps: {sum(nets):+d}  (>0 broadly helps; submit only if broadly >= 0)")

if __name__ == "__main__":
    main()
