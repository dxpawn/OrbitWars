"""Rollout worker: play one episode, return a trajectory.

The env doesn't expose step()/reset() so we collect (obs, action, value,
reward) tuples INSIDE the agent_fn closure that env.run drives. After the
episode ends, compute returns + advantages externally.

A trajectory is a dict of CPU tensors / numpy arrays. Suitable for sending
back through multiprocessing.Queue (pickle-friendly).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from eval._quiet import make
from rl.action_space import sample_action
from rl.features import encode
from rl.policy import OrbitWarsPolicy
from rl.reward import compute_step_reward, compute_terminal_reward


# Globals lazily initialized per worker process (so we don't pay re-init cost
# across many episodes within one worker).
_POLICY: OrbitWarsPolicy | None = None
_DEVICE: torch.device = torch.device("cpu")


def _ensure_policy(d_model: int, n_heads: int, n_layers: int):
    global _POLICY
    if _POLICY is None:
        _POLICY = OrbitWarsPolicy(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        _POLICY.eval()
    return _POLICY


def _load_state(policy: OrbitWarsPolicy, state: dict):
    if state is not None:
        policy.load_state_dict(state)


def rollout_episode(
    state_dict: dict | None,
    opponent_spec,             # str name or callable
    seed: int,
    *,
    n_players: int = 2,
    extras=None,
    d_model: int = 96,
    n_heads: int = 4,
    n_layers: int = 3,
    player_index: int = 0,
    deterministic: bool = False,
    shape_coef: float = 1.0,
    max_entities: int = 96,
):
    """Run one episode. Returns trajectory dict.

    Args:
        state_dict: policy weights (CPU). None to use random init.
        opponent_spec: name or callable; passed to env.run for the OTHER seat.
        seed: env seed.
        n_players: 2 or 4.
        extras: extra opponents for 4p (list of name/callable, length 2).
        player_index: which seat our agent sits in (0..n_players-1). Usually 0.
        deterministic: argmax sampling instead of stochastic (eval mode).
        shape_coef: reward shaping coefficient.

    Returns:
        dict with keys:
            entities, mask, globals: stacked input tensors (T, ...)
            launch_sampled, launch_logp, target_idx, target_logp,
              fraction_idx, fraction_logp, my_slots, my_planet_ids,
              n_launches, value, reward: per-step lists/arrays
            terminal_reward: float
            steps: int
            duration_s: float
    """
    policy = _ensure_policy(d_model, n_heads, n_layers)
    _load_state(policy, state_dict)
    policy.eval()

    # Resolve opponent into something env.run accepts
    def _resolve(spec):
        if callable(spec):
            return spec
        if isinstance(spec, str):
            try:
                import opponents
                if spec in opponents.REGISTRY:
                    return opponents.REGISTRY[spec]
            except (ImportError, AttributeError):
                pass
            return spec  # file path or env built-in name
        return spec

    traj_entities: list[np.ndarray] = []
    traj_mask: list[np.ndarray] = []
    traj_globals: list[np.ndarray] = []
    traj_launch_sampled: list[torch.Tensor] = []
    traj_launch_logp: list[torch.Tensor] = []
    traj_target_idx: list[torch.Tensor] = []
    traj_target_logp: list[torch.Tensor] = []
    traj_fraction_idx: list[torch.Tensor] = []
    traj_fraction_logp: list[torch.Tensor] = []
    traj_my_slots: list[torch.Tensor] = []
    traj_my_planet_ids: list[torch.Tensor] = []
    traj_values: list[float] = []
    traj_rewards: list[float] = []

    prev_obs_snapshot = None
    last_step_seen = -1

    def agent_fn(obs):
        nonlocal prev_obs_snapshot, last_step_seen
        # Encode + forward
        enc = encode(obs, max_entities=max_entities)
        # Skip storing if we have no owned planets (no actionable state).
        if len(enc.my_planet_slots) == 0:
            return []

        ent = torch.from_numpy(enc.entities).unsqueeze(0)
        mask = torch.from_numpy(enc.mask).unsqueeze(0)
        gl = torch.from_numpy(enc.globals_).unsqueeze(0)
        with torch.no_grad():
            out = policy(ent, mask, gl)
        moves, record = sample_action(out, enc, obs, deterministic=deterministic)

        # Shaping reward for previous step (computed against prev_obs_snapshot)
        step_reward = (
            compute_step_reward(prev_obs_snapshot, obs, player_index, shape_coef=shape_coef)
            if prev_obs_snapshot is not None
            else 0.0
        )
        if traj_rewards:
            traj_rewards[-1] = step_reward  # update prev step's reward
        # Snapshot current obs as last
        prev_obs_snapshot = _snapshot_obs(obs)

        # Record
        traj_entities.append(enc.entities)
        traj_mask.append(enc.mask)
        traj_globals.append(enc.globals_)
        traj_launch_sampled.append(record["launch_sampled"])
        traj_launch_logp.append(record["launch_logp"])
        traj_target_idx.append(record["target_idx"])
        traj_target_logp.append(record["target_logp"])
        traj_fraction_idx.append(record["fraction_idx"])
        traj_fraction_logp.append(record["fraction_logp"])
        traj_my_slots.append(record["my_slots"])
        traj_my_planet_ids.append(torch.from_numpy(enc.my_planet_ids).long())
        traj_values.append(record["value"])
        traj_rewards.append(0.0)
        return moves

    # Build agent list
    agents = [_resolve(opponent_spec)] * n_players
    agents[player_index] = agent_fn
    if extras is not None:
        # Override the non-player-index seats with extras
        idx = 0
        for i in range(n_players):
            if i == player_index:
                continue
            if idx < len(extras):
                agents[i] = _resolve(extras[idx])
                idx += 1

    env = make("orbit_wars", configuration={"seed": int(seed)}, debug=False)
    t0 = time.time()
    env.run(agents)
    duration = time.time() - t0

    final = env.steps[-1]
    final_obs = final[player_index].observation
    terminal = compute_terminal_reward(final_obs, player_index, n_players)

    # Append terminal reward to the last collected step.
    if traj_rewards:
        traj_rewards[-1] = traj_rewards[-1] + terminal
    else:
        # Edge case: agent never acted (was eliminated before turn 1)
        traj_rewards.append(terminal)
        traj_values.append(0.0)

    T = len(traj_entities)
    return {
        "entities": np.stack(traj_entities, axis=0) if T else np.zeros((0, max_entities, 32), dtype=np.float32),
        "mask": np.stack(traj_mask, axis=0) if T else np.zeros((0, max_entities), dtype=bool),
        "globals": np.stack(traj_globals, axis=0) if T else np.zeros((0, 12), dtype=np.float32),
        "launch_sampled": traj_launch_sampled,
        "launch_logp": traj_launch_logp,
        "target_idx": traj_target_idx,
        "target_logp": traj_target_logp,
        "fraction_idx": traj_fraction_idx,
        "fraction_logp": traj_fraction_logp,
        "my_slots": traj_my_slots,
        "my_planet_ids": traj_my_planet_ids,
        "value": np.array(traj_values, dtype=np.float32),
        "reward": np.array(traj_rewards, dtype=np.float32),
        "terminal_reward": float(terminal),
        "steps": int(len(env.steps)),
        "duration_s": float(duration),
        "T": T,
    }


def _snapshot_obs(obs):
    """Make a shallow snapshot of fields we need for shaping rewards.
    The observation gets MUTATED across turns (the engine reuses the dict),
    so we must copy out the values we care about.
    """
    if isinstance(obs, dict):
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
    else:
        planets = getattr(obs, "planets", None) or []
        fleets = getattr(obs, "fleets", None) or []
    return {
        "planets": [list(p) for p in planets],
        "fleets": [list(f) for f in fleets],
    }
