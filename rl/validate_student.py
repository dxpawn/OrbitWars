"""Fast distillation-fidelity check: per candidate-set, does OUR student rank targets
like his teacher? Instruments his score_many while his agent plays; for each scoring
call compares argmax(his) vs argmax(our) (the pick that actually drives re-ranking)
plus mean Spearman. Far faster + more direct than full-game win-rate for fidelity.

  python -m rl.validate_student --games-per-worker 1 --workers 16
"""
import os, sys, argparse
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
FRIEND_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "other_adversaries",
    "submission_feature46_transformer_v2_late_recapture_2p_v1"))
STUDENT_DIR = os.environ.get("STUDENT_DIR", os.path.join(_HERE, "student"))
POOL = ["heuristic_v6", "adv_hellburner", "adv_proto_v15", "adv_lb958"]

# per-mode accumulators: [calls, top1_agree, top3_agree, spearman_sum]
_ACC = {"2p": [0, 0, 0, 0.0], "4p": [0, 0, 0, 0.0]}


def _spearman(a, b):
    n = len(a)
    if n < 2:
        return 1.0
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0] * n
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 1.0


def _record(mode, his, our):
    if len(his) < 2:
        return
    acc = _ACC[mode]
    his_top = max(range(len(his)), key=lambda i: his[i])
    our_order = sorted(range(len(our)), key=lambda i: our[i], reverse=True)
    acc[0] += 1
    if our_order[0] == his_top:
        acc[1] += 1
    if his_top in our_order[:3]:
        acc[2] += 1
    acc[3] += _spearman(his, our)


def _worker(args):
    wid, n_games, seedbase = args
    for k in _ACC:
        _ACC[k] = [0, 0, 0, 0.0]
    if FRIEND_DIR not in sys.path: sys.path.insert(0, FRIEND_DIR)
    if STUDENT_DIR not in sys.path: sys.path.insert(0, STUDENT_DIR)
    import feature46_weights_2p as hw2, feature46_weights_4p as hw4
    import student_weights_2p as sw2, student_weights_4p as sw4
    ho2, ho4 = hw2.score_many, hw4.score_many

    def mk(orig, student, mode):
        def wrapped(rows):
            his = orig(rows)
            try:
                our = student.score_many(rows)
                _record(mode, [float(x) for x in his], [float(x) for x in our])
            except Exception:
                pass
            return his
        return wrapped
    hw2.score_many = mk(ho2, sw2, "2p")
    hw4.score_many = mk(ho4, sw4, "4p")

    import importlib.util
    spec = importlib.util.spec_from_file_location("friend_main_val", os.path.join(FRIEND_DIR, "main.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    agent = m.agent

    from eval.match import run_match
    import random
    rng = random.Random(seedbase + wid)
    for g in range(n_games):
        seed = seedbase * 1000 + wid * 100 + g
        if rng.random() < 0.5:
            run_match(agent, rng.choice(POOL), seed=seed)
        else:
            opps = rng.sample(POOL, 3)
            run_match(agent, opps[0], seed=seed, extra_agents=opps[1:])
    return {k: list(v) for k, v in _ACC.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-worker", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seedbase", type=int, default=77)
    args = ap.parse_args()
    jobs = [(w, args.games_per_worker, args.seedbase) for w in range(args.workers)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as p:
        results = p.map(_worker, jobs)
    for mode in ("2p", "4p"):
        c = t1 = t3 = 0; sp = 0.0
        for r in results:
            c += r[mode][0]; t1 += r[mode][1]; t3 += r[mode][2]; sp += r[mode][3]
        if c:
            print(f"[{mode}] calls={c}  top1_agree={t1/c:.1%}  top3_agree={t3/c:.1%}  mean_spearman={sp/c:.3f}", flush=True)
        else:
            print(f"[{mode}] no calls")


if __name__ == "__main__":
    main()
