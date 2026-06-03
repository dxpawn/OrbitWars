"""Feasibility probe for the distillation plan.

1. Can we call the friend's transformer scorer as a STANDALONE ORACLE?
   (feature row -> score). This is the linchpin of distillation.
2. Does his full agent load + run in our harness? Benchmark v6 vs him (2p).
"""
import os, sys, random, importlib.util

FRIEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "other_adversaries",
    "submission_feature46_transformer_v2_late_recapture_2p_v1",
)


def _load_friend_module(modname):
    """Import a module from the friend's folder under a namespaced name,
    with the folder on sys.path so its sibling imports resolve."""
    if FRIEND_DIR not in sys.path:
        sys.path.insert(0, FRIEND_DIR)
    return importlib.import_module(modname)


def probe_oracle():
    print("=== ORACLE PROBE: friend's scorer as standalone teacher ===")
    w2p = _load_friend_module("feature46_weights_2p")
    nfeat = len(getattr(w2p, "FEATURE_NAMES", []))
    print(f"feature dim: {nfeat}")
    # surface any architecture constants
    arch = {k: getattr(w2p, k) for k in dir(w2p)
            if k.isupper() and isinstance(getattr(w2p, k), (int, float))}
    print(f"scalar arch consts: {arch}")
    mean = getattr(w2p, "MEAN", None)
    # build a few feature rows: the dataset mean (=> ~neutral), and random rows
    rows = []
    if mean is not None:
        rows.append(list(mean))
    rng = random.Random(0)
    for _ in range(4):
        rows.append([rng.uniform(0.0, 1.0) for _ in range(nfeat)])
    scores = w2p.score_many(rows)
    print(f"score_many({len(rows)} rows) -> {[round(float(s),4) for s in scores]}")
    print(f"finite: {all(s==s and abs(float(s))<1e9 for s in scores)}")
    # 4p scorer too
    w4p = _load_friend_module("feature46_weights_4p")
    s4 = w4p.score_many(rows)
    print(f"4p scorer OK: {[round(float(s),4) for s in s4][:3]} ...")
    print("ORACLE: USABLE\n")
    return nfeat


def probe_agent_and_benchmark(games=6):
    print(f"=== AGENT LOAD + BENCHMARK: heuristic_v6 vs friend (2p, {games} games) ===")
    friend_main = _load_friend_module("main")  # his folder's main.py
    friend_agent = friend_main.agent
    from eval.match import run_match
    v6_wins = friend_wins = 0
    for i in range(games):
        seed = 900000 + i
        if i % 2 == 0:
            r = run_match("heuristic_v6", friend_agent, seed=seed)
            if r.winner == 0: v6_wins += 1
            elif r.winner == 1: friend_wins += 1
        else:
            r = run_match(friend_agent, "heuristic_v6", seed=seed)
            if r.winner == 1: v6_wins += 1
            elif r.winner == 0: friend_wins += 1
        print(f"  game {i}: steps={r.steps} winner_seat={r.winner} dur={r.duration_s:.1f}s")
    print(f"v6 {v6_wins} - {friend_wins} friend  (out of {games})")


if __name__ == "__main__":
    probe_oracle()
    probe_agent_and_benchmark()
