"""Grouped distillation collector (Edge A: context-aware student).

Same as distill_collect but logs per score_many CALL as a GROUP: every candidate
set his scorer ranked together gets a shared group id. This lets us add set-context
features (how a candidate compares to the others in its set) -> approximate his
transformer's cross-candidate attention with a still-pointwise-at-inference model.

Saves shards with X(N,46), y(N,), gid(N,) for 2p and 4p (gid groups rows of one call).

  python -m rl.distill_collect_grouped --games-per-worker 4 --workers 64 --out rl/distill_grouped
"""
import os
import sys
import argparse
import glob
import random
import multiprocessing as mp

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
FRIEND_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "other_adversaries",
    "submission_feature46_transformer_v2_late_recapture_2p_v1"))
POOL = ["heuristic_v6", "adv_hellburner", "adv_proto_v15", "adv_lb958"]

_X2, _y2, _g2 = [], [], []
_X4, _y4, _g4 = [], [], []
_gid = [0]


def _patch_and_load_friend():
    if FRIEND_DIR not in sys.path:
        sys.path.insert(0, FRIEND_DIR)
    import feature46_weights_2p as w2p
    import feature46_weights_4p as w4p
    _o2, _o4 = w2p.score_many, w4p.score_many

    def w2(rows):
        sc = _o2(rows)
        g = _gid[0]; _gid[0] += 1
        for r, s in zip(rows, sc):
            _X2.append(tuple(float(v) for v in r)); _y2.append(float(s)); _g2.append(g)
        return sc

    def w4(rows):
        sc = _o4(rows)
        g = _gid[0]; _gid[0] += 1
        for r, s in zip(rows, sc):
            _X4.append(tuple(float(v) for v in r)); _y4.append(float(s)); _g4.append(g)
        return sc

    w2p.score_many, w4p.score_many = w2, w4
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "friend_main_grouped", os.path.join(FRIEND_DIR, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.agent


def _worker(args):
    wid, n_games, seedbase, out_dir = args
    for lst in (_X2, _y2, _g2, _X4, _y4, _g4):
        lst.clear()
    _gid[0] = 0
    agent = _patch_and_load_friend()
    from eval.match import run_match
    rng = random.Random(seedbase * 100003 + wid)
    for g in range(n_games):
        seed = seedbase * 1_000_000 + wid * 1000 + g
        if rng.random() < 0.5:
            run_match(agent, rng.choice(POOL), seed=seed)
        else:
            opps = rng.sample(POOL, 3)
            run_match(agent, opps[0], seed=seed, extra_agents=opps[1:])
    shard = os.path.join(out_dir, f"shard_{seedbase}_{wid:03d}.npz")
    np.savez_compressed(
        shard,
        X2=np.array(_X2, np.float32).reshape(-1, 46), y2=np.array(_y2, np.float32), g2=np.array(_g2, np.int64),
        X4=np.array(_X4, np.float32).reshape(-1, 46), y4=np.array(_y4, np.float32), g4=np.array(_g4, np.int64))
    return wid, len(_y2), len(_y4)


def collect(games_per_worker, workers, seedbase, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(w, games_per_worker, seedbase, out_dir) for w in range(workers)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        results = p.map(_worker, jobs)
    print(f"collected 2p rows={sum(r[1] for r in results)} 4p rows={sum(r[2] for r in results)}", flush=True)


def aggregate(out_dir):
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    out = {"X2": [], "y2": [], "g2": [], "X4": [], "y4": [], "g4": []}
    base = 0
    for si, s in enumerate(shards):
        d = np.load(s)
        off = si * 100_000_000  # keep gids unique across shards
        for k in ("X2", "y2", "X4", "y4"):
            if len(d[k]):
                out[k].append(d[k])
        for k in ("g2", "g4"):
            if len(d[k]):
                out[k].append(d[k] + off)
    res = {}
    for k in out:
        res[k] = np.concatenate(out[k]) if out[k] else (
            np.zeros((0, 46), np.float32) if k.startswith("X") else np.zeros((0,), np.int64 if k[0] == "g" else np.float32))
    outf = os.path.join(out_dir, "dataset.npz")
    np.savez_compressed(outf, **res)
    n2g = len(np.unique(res["g2"])); n4g = len(np.unique(res["g4"]))
    print(f"aggregated {len(shards)} shards -> {outf}")
    print(f"  2p: rows={res['X2'].shape[0]} groups={n2g}   4p: rows={res['X4'].shape[0]} groups={n4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-worker", type=int, default=4)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seedbase", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(_HERE, "distill_grouped"))
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.out)
    else:
        collect(args.games_per_worker, args.workers, args.seedbase, args.out)


if __name__ == "__main__":
    main()
