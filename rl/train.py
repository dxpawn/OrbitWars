"""Main PPO training loop with multiprocessing rollout workers."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from rl.league import League, default_league
from rl.policy import OrbitWarsPolicy
from rl.ppo import flatten_trajectories, ppo_update
from rl.rollout_worker import rollout_episode


def _trim_state_dict(state_dict: dict) -> dict:
    """Convert state_dict tensors to CPU so they pickle cheaply."""
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def _save_checkpoint(policy, optimizer, step: int, path: Path, league: League | None = None):
    payload = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "league_stats": league.stats() if league else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(policy, optimizer, path: Path) -> int:
    payload = torch.load(path, map_location="cpu")
    policy.load_state_dict(payload["policy"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))


def collect_rollouts(state_cpu, league, n_episodes, n_players, rng, max_entities, shape_coef,
                     d_model, n_heads, n_layers, workers, pool=None):
    """Collect n_episodes trajectories. Uses the given multiprocessing.Pool if provided."""
    jobs = []
    chosen_opponents = []
    for _ in range(n_episodes):
        opp_name = league.sample(rng=rng)
        chosen_opponents.append(opp_name)
        from opponents import REGISTRY as OPP_REGISTRY
        opp_spec = OPP_REGISTRY[opp_name]
        seed = rng.randint(0, 2**31 - 1)
        extras = None
        if n_players == 4:
            # Fill the other two seats with two random other agents from the league
            others = [n for n in league.members if n != opp_name]
            picks = rng.sample(others, k=min(2, len(others))) if others else [opp_name, opp_name]
            extras = [OPP_REGISTRY[p] for p in picks]
        jobs.append((state_cpu, opp_spec, seed, n_players, extras, d_model, n_heads, n_layers, max_entities, shape_coef))

    if pool is None:
        results = [_worker_fn(j) for j in jobs]
    else:
        results = pool.map(_worker_fn, jobs)

    return results, chosen_opponents


def _worker_fn(args):
    state_cpu, opp_spec, seed, n_players, extras, d_model, n_heads, n_layers, max_entities, shape_coef = args
    return rollout_episode(
        state_cpu, opp_spec, seed,
        n_players=n_players,
        extras=extras,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_entities=max_entities,
        shape_coef=shape_coef,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--episodes-per-iter", type=int, default=16)
    parser.add_argument("--total-iters", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-players", type=int, default=2, choices=(2, 4))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--max-entities", type=int, default=96)
    parser.add_argument("--shape-coef-start", type=float, default=1.0)
    parser.add_argument("--shape-coef-end", type=float, default=0.05)
    parser.add_argument("--shape-anneal-iters", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--mix-4p-prob", type=float, default=0.25,
                       help="Probability of running a 4-player episode each iter (rest are 2p).")
    args = parser.parse_args(argv)

    ckpt_dir = Path(args.checkpoints_dir)
    state_dir = Path(args.state_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    policy = OrbitWarsPolicy(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        start_step = _load_checkpoint(policy, optimizer, Path(args.resume))
        print(f"Resumed from {args.resume} at step {start_step}")

    league = default_league()
    print(f"League members: {sorted(league.members)}")

    # Persist league stats next to checkpoints
    league_path = state_dir / "league.json"

    # Multiprocessing setup
    import multiprocessing as mp
    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    pool = ctx.Pool(args.workers) if args.workers > 1 else None

    try:
        for it in range(start_step // args.episodes_per_iter, args.total_iters):
            t0 = time.time()
            # Anneal shape coef
            t = min(1.0, it / max(1, args.shape_anneal_iters))
            shape_coef = args.shape_coef_start + t * (args.shape_coef_end - args.shape_coef_start)

            # Decide format: 2p or 4p for this iter
            n_players = 4 if rng.random() < args.mix_4p_prob else 2

            state_cpu = _trim_state_dict(policy.state_dict())
            results, chosen_opps = collect_rollouts(
                state_cpu, league, args.episodes_per_iter, n_players, rng,
                args.max_entities, shape_coef,
                args.d_model, args.n_heads, args.n_layers, args.workers,
                pool=pool,
            )
            rollout_t = time.time() - t0

            # Update league stats from terminal rewards
            wins = 0
            draws = 0
            for tj, opp in zip(results, chosen_opps):
                tr = tj["terminal_reward"]
                if tr >= 0.99:
                    league.record(opp, win=True)
                    wins += 1
                elif tr <= -0.99:
                    league.record(opp, win=False)
                else:
                    league.record(opp, win=False, draw=True)
                    draws += 1

            # PPO update
            t1 = time.time()
            batch = flatten_trajectories(results)
            if batch is not None:
                metrics = ppo_update(policy, optimizer, batch, device=device)
            else:
                metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
            update_t = time.time() - t1

            avg_steps = float(np.mean([r["steps"] for r in results]))
            avg_T = float(np.mean([r["T"] for r in results]))
            avg_reward = float(np.mean([r["terminal_reward"] for r in results]))

            step_count = (it + 1) * args.episodes_per_iter
            if it % args.log_every == 0:
                print(
                    f"[iter {it:>5d}] {n_players}p win={wins}/{args.episodes_per_iter} (draw={draws}) "
                    f"rew={avg_reward:+.2f} steps={avg_steps:.0f} T={avg_T:.0f} "
                    f"loss={metrics['loss']:.3f} v={metrics['value_loss']:.3f} "
                    f"ent={metrics['entropy']:.2f} kl={metrics['approx_kl']:.4f} "
                    f"shape={shape_coef:.2f} t(roll={rollout_t:.1f}s upd={update_t:.1f}s)",
                    flush=True,
                )

            if it % args.save_every == 0 and it > 0:
                _save_checkpoint(
                    policy, optimizer, step_count,
                    ckpt_dir / f"step_{step_count:08d}.pt",
                    league=league,
                )
                # Also update "latest" symlink/copy
                _save_checkpoint(policy, optimizer, step_count, ckpt_dir / "latest.pt", league=league)
                # Save league stats
                league_path.write_text(json.dumps(league.stats(), indent=2, sort_keys=True))

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Final save
    _save_checkpoint(policy, optimizer, args.total_iters * args.episodes_per_iter,
                     ckpt_dir / "final.pt", league=league)


if __name__ == "__main__":
    main()
