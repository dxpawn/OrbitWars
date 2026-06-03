"""Bundle the distilled agent into a SINGLE submission .py for Kaggle.

Embeds (base64) his hull `orbit_base.py` + his feature extractor `main.py` + OUR
distilled `student_weights_{2p,4p}.py` (renamed to the names his main imports:
feature46_weights_2p/4p). Each is materialized as a real module in sys.modules so
his `import orbit_base` / `import feature46_weights_2p` resolve and the
agent/_nearest_targets name collisions are avoided. His 7.6 MB transformer is NOT
included — only our small model ships.

  python -m scripts.build_distilled_submission --student rl/student --out submission_distilled.py
"""
import os
import sys
import base64
import argparse
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRIEND_DIR = os.path.join(
    _HERE, "other_adversaries", "submission_feature46_transformer_v2_late_recapture_2p_v1")


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build(student_dir, out_path):
    parts = {
        "orbit_base": os.path.join(FRIEND_DIR, "orbit_base.py"),
        "feature46_weights_2p": os.path.join(student_dir, "student_weights_2p.py"),
        "feature46_weights_4p": os.path.join(student_dir, "student_weights_4p.py"),
        "main": os.path.join(FRIEND_DIR, "main.py"),
    }
    enc = {name: _b64(p) for name, p in parts.items()}
    lines = [
        '"""Distilled re-ranker submission: friend\'s hull + features (reused), OUR model.',
        'Single-file bundle; his transformer is NOT included. Built by scripts/build_distilled_submission.py."""',
        "import sys, types, base64",
        "",
        "_SRC = {",
    ]
    # emit in dependency order: orbit_base, weights, then main (which imports them)
    for name in ("orbit_base", "feature46_weights_2p", "feature46_weights_4p", "main"):
        lines.append(f"  {name!r}: {enc[name]!r},")
    lines.append("}")
    lines.append("""
def _mk(name):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__name__ = name
    sys.modules[name] = m
    exec(compile(base64.b64decode(_SRC[name]).decode("utf-8-sig"), name + ".py", "exec"), m.__dict__)
    return m

# dependency order: hull + scorers first, then main (its imports resolve from sys.modules)
_mk("orbit_base")
_mk("feature46_weights_2p")
_mk("feature46_weights_4p")
_main = _mk("main")


def agent(obs, config=None):
    return _main.agent(obs, config)
""")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    size = os.path.getsize(out_path)
    print(f"built {out_path}  ({size/1024:.0f} KB)")


def selftest(out_path):
    """Import the bundle in isolation and run one 4p game to confirm it works."""
    spec = importlib.util.spec_from_file_location("submission_distilled_test", out_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from eval.match import run_match
    r = run_match(m.agent, "adv_hellburner", seed=900222, extra_agents=["adv_proto_v15", "adv_lb958"])
    print(f"selftest 4p game: steps={r.steps} winner={r.winner} scores={[round(x) for x in r.scores]}")
    # confirm his transformer is NOT loaded
    print("his transformer feature46 NOT bundled:", not any(
        "feature46_transformer" in str(getattr(mod, '__file__', '')) for mod in sys.modules.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=os.path.join(_HERE, "rl", "student"))
    ap.add_argument("--out", default=os.path.join(_HERE, "submission_distilled.py"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    build(args.student, args.out)
    if args.selftest:
        selftest(args.out)


if __name__ == "__main__":
    main()
