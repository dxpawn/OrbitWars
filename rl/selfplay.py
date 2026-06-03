"""Self-play foundation for Stage-2 finetuning.

make_agent(student_dir): build a distilled agent whose scorer comes from the given
student dir (his hull + features reused; OUR re-ranker swapped in). Multiple agents
with DIFFERENT weights coexist in one process (sys.modules save/restore + unique
module names), so we can run A-vs-B self-play in a single game.

winrate(A_dir, B_dir, games, workers): A's win-rate with A as hero vs B filling the
other seats (2p + 4p mix). Fast (our re-ranker is a tiny MLP -> no timeout).

  python -m rl.selfplay --a rl/student --b rl/student --games 64      # sanity ~50%
  python -m rl.selfplay --a rl/student --b-name heuristic_v6 --games 64
"""
import os
import sys
import argparse
import importlib.util
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
FRIEND_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "other_adversaries",
    "submission_feature46_transformer_v2_late_recapture_2p_v1"))

_uid = [0]


def _load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_agent(student_dir):
    """Return an agent callable using OUR re-ranker from student_dir on his hull."""
    if FRIEND_DIR not in sys.path:
        sys.path.insert(0, FRIEND_DIR)
    _uid[0] += 1
    n = _uid[0]
    sw2 = _load_file(os.path.join(student_dir, "student_weights_2p.py"), f"_sw2_{n}")
    sw4 = _load_file(os.path.join(student_dir, "student_weights_4p.py"), f"_sw4_{n}")
    keys = ("feature46_weights_2p", "feature46_weights_4p")
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules["feature46_weights_2p"] = sw2
    sys.modules["feature46_weights_4p"] = sw4
    try:
        m = _load_file(os.path.join(FRIEND_DIR, "main.py"), f"_main_{n}")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return m.agent


def _resolve(spec):
    """spec is a student dir (path) -> distilled agent, or a registry name -> that agent."""
    if os.path.isdir(spec):
        return make_agent(spec)
    import opponents
    return opponents.REGISTRY[spec]


def _worker(args):
    a_spec, b_spec, i, seedbase = args
    from eval.match import run_match
    A = _resolve(a_spec)
    B = _resolve(b_spec)
    seed = seedbase + i
    # alternate: 50% 2p, 50% 4p; rotate A's seat by parity
    if i % 2 == 0:  # 2p
        if (i // 2) % 2 == 0:
            r = run_match(A, B, seed=seed); return r.winner == 0
        r = run_match(B, A, seed=seed); return r.winner == 1
    else:  # 4p: A vs three B-copies (fresh B instances each fill a seat)
        seat = (i // 2) % 4
        agents = [B, B, B]
        agents.insert(seat, A)
        r = run_match(agents[0], agents[1], seed=seed, extra_agents=agents[2:])
        return r.winner == seat


def winrate(a_spec, b_spec, games, workers, seedbase=500000):
    jobs = [(a_spec, b_spec, i, seedbase) for i in range(games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        res = p.map(_worker, jobs)
    w = sum(1 for x in res if x)
    return w, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(_HERE, "student"), help="A student dir")
    ap.add_argument("--b", default=None, help="B student dir")
    ap.add_argument("--b-name", default=None, help="B registry name instead of a dir")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seedbase", type=int, default=500000)
    args = ap.parse_args()
    b_spec = args.b_name if args.b_name else (args.b or os.path.join(_HERE, "student"))
    w, n = winrate(args.a, b_spec, args.games, args.workers, args.seedbase)
    print(f"A={args.a} vs B={b_spec}: A wins {w}/{n} = {w/n:.1%}", flush=True)


if __name__ == "__main__":
    main()
