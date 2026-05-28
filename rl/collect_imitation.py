"""Collect imitation-learning data by having adversaries play each other.

For each game, log (obs, moves) per player at every turn. Save as pickled
list-of-tuples per game.

The collected data is then used by rl/imitation_train.py to supervised-train
the policy net to mimic the strong adversaries.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import random
import time
from itertools import combinations
from pathlib import Path

from eval._quiet import make


# Which agents are worth imitating. Heaviest opponents win games, so we mimic them.
IMITATION_TARGETS = (
    "adv_distance",
    "adv_lbmax",
    "adv_structured",
    "adv_rf_v0",
    "adv_rf_v1",
    "adv_rf_v2",
)


def _snapshot_obs(obs):
    """Make a deep enough copy of fields we care about, so we don't store
    mutating references."""
    if isinstance(obs, dict):
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        player = obs.get("player", 0)
        step = obs.get("step", 0)
        omega = obs.get("angular_velocity", 0.03)
        initial = obs.get("initial_planets") or []
        comets = obs.get("comets") or []
        comet_ids = obs.get("comet_planet_ids") or []
    else:
        planets = getattr(obs, "planets", None) or []
        fleets = getattr(obs, "fleets", None) or []
        player = getattr(obs, "player", 0)
        step = getattr(obs, "step", 0)
        omega = getattr(obs, "angular_velocity", 0.03)
        initial = getattr(obs, "initial_planets", None) or []
        comets = getattr(obs, "comets", None) or []
        comet_ids = getattr(obs, "comet_planet_ids", None) or []
    return {
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
        "initial_planets": [list(p) for p in initial],
        "comets": [
            {
                "planet_ids": list(c.get("planet_ids", [])),
                "paths": [list(p) for p in c.get("paths", [])],
                "path_index": c.get("path_index", 0),
            }
            for c in comets
        ],
        "comet_planet_ids": list(comet_ids),
        "player": int(player),
        "step": int(step),
        "angular_velocity": float(omega),
    }


def _wrap_agent(agent_callable, log):
    """Wrap an agent so each (obs, returned_moves) pair is logged."""
    def wrapped(obs):
        snap = _snapshot_obs(obs)
        moves = agent_callable(obs)
        # Make sure we deep-copy moves too (defensive)
        log.append((snap, [list(m) for m in (moves or [])]))
        return moves
    return wrapped


def _resolve_callable(spec):
    """Resolve a registry name or file path into a callable agent."""
    from opponents import REGISTRY as OPP_REGISTRY
    if callable(spec):
        return spec
    if isinstance(spec, str) and spec in OPP_REGISTRY:
        spec = OPP_REGISTRY[spec]
    if callable(spec):
        return spec
    # File path — load as module
    import importlib.util
    spec_obj = importlib.util.spec_from_file_location("_loaded_agent", spec)
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)
    return mod.agent


def play_and_log(args):
    """Worker function. Args: (agent_a_name, agent_b_name, seed, out_dir)."""
    a_name, b_name, seed, out_dir = args
    a_fn = _resolve_callable(a_name)
    b_fn = _resolve_callable(b_name)

    log_a, log_b = [], []
    wrapped_a = _wrap_agent(a_fn, log_a)
    wrapped_b = _wrap_agent(b_fn, log_b)

    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    t0 = time.time()
    env.run([wrapped_a, wrapped_b])
    duration = time.time() - t0

    final = env.steps[-1]
    rewards = [s.reward for s in final]

    # Save winner's trajectory (we only mimic players who won)
    winner_idx = 0 if rewards[0] >= 0.99 else (1 if rewards[1] >= 0.99 else -1)

    out_path = Path(out_dir) / f"{a_name}_vs_{b_name}_seed{seed}.pkl"
    payload = {
        "a_name": a_name,
        "b_name": b_name,
        "seed": seed,
        "rewards": tuple(rewards),
        "winner_idx": winner_idx,
        "log_a": log_a,
        "log_b": log_b,
        "duration_s": duration,
        "steps": len(env.steps),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"path": str(out_path), "winner": winner_idx, "rewards": rewards, "duration": duration}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="state/imitation_data")
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=10000)
    parser.add_argument("--targets", nargs="*", default=list(IMITATION_TARGETS))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairings = list(combinations(args.targets, 2))
    print(f"Pairings: {len(pairings)} pairs × {args.games_per_pair} games = {len(pairings)*args.games_per_pair} games total")

    jobs = []
    for i, (a, b) in enumerate(pairings):
        for s in range(args.games_per_pair):
            seed = args.seed_offset + i * args.games_per_pair + s
            out_path = out_dir / f"{a}_vs_{b}_seed{seed}.pkl"
            if args.skip_existing and out_path.exists():
                continue
            jobs.append((a, b, seed, str(out_dir)))

    print(f"Running {len(jobs)} games (skip-existing={args.skip_existing}) with {args.workers} workers")
    t0 = time.time()
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(play_and_log, jobs)):
            if (i + 1) % 5 == 0 or i == len(jobs) - 1:
                elapsed = time.time() - t0
                print(f"[{i+1}/{len(jobs)}] {result['path']} winner={result['winner']} rewards={result['rewards']} ({result['duration']:.1f}s) | total {elapsed:.0f}s", flush=True)

    print(f"Done. Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
