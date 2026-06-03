"""Distillation data collector.

Logs every (46-feature row -> his RAW score) pair the friend's transformer emits
as his agent plays. We REUSE his feature extractor + scorer (the oracle/teacher,
full permission); the collected dataset trains OUR OWN re-ranker (the deliverable).

On-policy: his agent plays 2p + 4p games vs a diverse pool, so the feature/state
distribution matches what his hull (which we'll reuse at inference) actually visits.
Parallel across workers; each writes a .npz shard {X2,y2,X4,y4}.

  python -m rl.distill_collect --games-per-worker 3 --workers 64 --out rl/distill_data --seedbase 1000

Then aggregate with --aggregate.
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

# Fast-ish, diverse opponents (avoid the heaviest heuristic1000 to keep games quick;
# his own transformer scoring is the dominant cost anyway).
POOL = ["heuristic_v6", "adv_hellburner", "adv_proto_v15", "adv_lb958"]

_LOG2: list = []
_LOG4: list = []


def _patch_and_load_friend():
    """Put his folder on sys.path, monkey-patch score_many to log (row, score),
    then load his agent (which calls the patched scorer at play time)."""
    if FRIEND_DIR not in sys.path:
        sys.path.insert(0, FRIEND_DIR)
    import feature46_weights_2p as w2p
    import feature46_weights_4p as w4p
    _o2, _o4 = w2p.score_many, w4p.score_many

    def w2(rows):
        sc = _o2(rows)
        for r, s in zip(rows, sc):
            _LOG2.append((tuple(float(v) for v in r), float(s)))
        return sc

    def w4(rows):
        sc = _o4(rows)
        for r, s in zip(rows, sc):
            _LOG4.append((tuple(float(v) for v in r), float(s)))
        return sc

    w2p.score_many, w4p.score_many = w2, w4
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "friend_main_collect", os.path.join(FRIEND_DIR, "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.agent


def _to_arrays(log):
    if not log:
        return np.zeros((0, 46), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.array([r for r, _ in log], dtype=np.float32)
    y = np.array([s for _, s in log], dtype=np.float32)
    return X, y


def _worker(args):
    wid, n_games, seedbase, out_dir = args
    _LOG2.clear()
    _LOG4.clear()
    agent = _patch_and_load_friend()
    from eval.match import run_match
    rng = random.Random(seedbase * 100003 + wid)
    for g in range(n_games):
        seed = seedbase * 1_000_000 + wid * 1000 + g
        if rng.random() < 0.5:
            opp = rng.choice(POOL)
            run_match(agent, opp, seed=seed)                       # 2p
        else:
            opps = rng.sample(POOL, 3)
            run_match(agent, opps[0], seed=seed, extra_agents=opps[1:])  # 4p
    X2, y2 = _to_arrays(_LOG2)
    X4, y4 = _to_arrays(_LOG4)
    shard = os.path.join(out_dir, f"shard_{seedbase}_{wid:03d}.npz")
    np.savez_compressed(shard, X2=X2, y2=y2, X4=X4, y4=y4)
    return wid, len(y2), len(y4)


def collect(games_per_worker, workers, seedbase, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(w, games_per_worker, seedbase, out_dir) for w in range(workers)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        results = p.map(_worker, jobs)
    n2 = sum(r[1] for r in results)
    n4 = sum(r[2] for r in results)
    print(f"collected: 2p rows={n2}  4p rows={n4}  across {workers} workers x {games_per_worker} games", flush=True)


def aggregate(out_dir):
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    X2s, y2s, X4s, y4s = [], [], [], []
    for s in shards:
        d = np.load(s)
        if len(d["y2"]): X2s.append(d["X2"]); y2s.append(d["y2"])
        if len(d["y4"]): X4s.append(d["X4"]); y4s.append(d["y4"])
    X2 = np.concatenate(X2s) if X2s else np.zeros((0, 46), np.float32)
    y2 = np.concatenate(y2s) if y2s else np.zeros((0,), np.float32)
    X4 = np.concatenate(X4s) if X4s else np.zeros((0, 46), np.float32)
    y4 = np.concatenate(y4s) if y4s else np.zeros((0,), np.float32)
    out = os.path.join(out_dir, "dataset.npz")
    np.savez_compressed(out, X2=X2, y2=y2, X4=X4, y4=y4)
    print(f"aggregated {len(shards)} shards -> {out}")
    print(f"  2p: X{X2.shape} y{y2.shape}   4p: X{X4.shape} y{y4.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-worker", type=int, default=3)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seedbase", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(_HERE, "distill_data"))
    ap.add_argument("--aggregate", action="store_true", help="just aggregate existing shards")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.out)
    else:
        collect(args.games_per_worker, args.workers, args.seedbase, args.out)


if __name__ == "__main__":
    main()
