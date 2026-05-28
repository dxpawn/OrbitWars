"""Action sampling + log-prob accounting for the policy net.

The policy outputs per-entity (launch, target, fraction) factors. For each
of OUR owned planets we sample independently:
  1. Bernoulli(launch_logit[slot]) — emit a fleet from this planet?
  2. Categorical(target_logits[slot]) over all other entities (masked).
  3. Categorical(fraction_logits[slot]) over SHIP_FRACTIONS.

Angle is computed deterministically from src → predicted intercept of the
target (we use the target's CURRENT position; the engine handles dynamics).

The log-prob of an action is the sum of independent log-probs across all
launches we took.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from rl.features import EncodedObs, BOARD, CENTER, SUN_R, ROTATION_RADIUS_LIMIT
from rl.policy import SHIP_FRACTIONS, N_FRACTIONS


def _seg_point_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    ls = dx * dx + dy * dy
    if ls < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / ls
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _safe_angle(sx, sy, tx, ty, margin: float = 1.5):
    """Direct angle if sun isn't blocking, else tangent around the sun."""
    direct = math.atan2(ty - sy, tx - sx)
    if _seg_point_dist(CENTER, CENTER, sx, sy, tx, ty) >= SUN_R + margin:
        return direct
    d = math.hypot(sx - CENTER, sy - CENTER)
    if d <= SUN_R + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_R + 2.0) / d))
    to_sun = math.atan2(CENTER - sy, CENTER - sx)
    cw = to_sun + half
    ccw = to_sun - half
    def norm(a):
        while a <= -math.pi:
            a += 2 * math.pi
        while a > math.pi:
            a -= 2 * math.pi
        return a
    return cw if abs(norm(cw - direct)) < abs(norm(ccw - direct)) else ccw


def sample_action(net_out: dict, enc: EncodedObs, raw_obs, *, deterministic: bool = False):
    """Sample one action (set of moves) and return (moves, action_record).

    Args:
        net_out: dict of (B=1, ...) tensors from OrbitWarsPolicy.forward.
        enc: the EncodedObs that produced the input.
        raw_obs: the original observation (needed for current ship counts).
        deterministic: if True, take argmax/threshold at 0.5 instead of sampling.

    Returns:
        moves: list of [from_id, angle, ships] for env consumption.
        record: dict storing tensors needed for PPO updates:
            - my_slots: (K,) long, planet slots we owned
            - launch_sampled: (K,) bool
            - launch_logp: (K,) float
            - target_idx: (K,) long  (-1 if no launch)
            - target_logp: (K,) float
            - fraction_idx: (K,) long
            - fraction_logp: (K,) float
            - value: scalar
            - n_launches: int
    """
    launch_l = net_out["launch_logit"][0]      # (N,)
    target_l = net_out["target_logits"][0]     # (N, N)
    fraction_l = net_out["fraction_logits"][0] # (N, F)
    value = net_out["value"][0].item()

    my_slots = enc.my_planet_slots
    my_ids = enc.my_planet_ids
    if len(my_slots) == 0:
        return [], {
            "my_slots": torch.empty(0, dtype=torch.long),
            "launch_sampled": torch.empty(0, dtype=torch.bool),
            "launch_logp": torch.empty(0),
            "target_idx": torch.empty(0, dtype=torch.long),
            "target_logp": torch.empty(0),
            "fraction_idx": torch.empty(0, dtype=torch.long),
            "fraction_logp": torch.empty(0),
            "value": value,
            "n_launches": 0,
        }

    moves: list[list] = []
    n_launches = 0

    K = len(my_slots)
    launch_sampled = torch.zeros(K, dtype=torch.bool)
    launch_logp = torch.zeros(K)
    target_idx = torch.full((K,), -1, dtype=torch.long)
    target_logp = torch.zeros(K)
    fraction_idx = torch.zeros(K, dtype=torch.long)
    fraction_logp = torch.zeros(K)

    # Get current ship counts per owned planet (the encoder stores normalized)
    planets_raw = raw_obs["planets"] if isinstance(raw_obs, dict) else raw_obs.planets
    ships_by_pid = {p[0]: p[5] for p in planets_raw}
    pos_by_pid = {p[0]: (float(p[2]), float(p[3])) for p in planets_raw}
    radius_by_pid = {p[0]: float(p[4]) for p in planets_raw}

    for k in range(K):
        slot = int(my_slots[k])
        pid = int(my_ids[k])

        # 1. Launch?
        p_launch = torch.sigmoid(launch_l[slot])
        if deterministic:
            do_launch = bool(p_launch >= 0.5)
        else:
            do_launch = bool(torch.bernoulli(p_launch).item())
        # log-prob of the binary choice
        lp_launch = (
            torch.log(p_launch.clamp_min(1e-8))
            if do_launch
            else torch.log((1.0 - p_launch).clamp_min(1e-8))
        )
        launch_sampled[k] = do_launch
        launch_logp[k] = lp_launch.detach()

        if not do_launch:
            continue

        # 2. Target
        t_logits = target_l[slot]  # (N,)
        # Mask: only valid entities; also exclude own planets that aren't tactical
        # (we still allow targeting own planets so we can REINFORCE; the engine
        # will treat same-owner arrivals as reinforcements). Reasonable.
        valid = torch.from_numpy(enc.mask).to(t_logits.device).clone()
        # Exclude self
        valid[slot] = False
        # Exclude the sun/global tokens
        for ts in (-2, -1):  # last two slots are sun + global by construction
            # find them by mask-trace? Easier: mask via the type onehot
            pass
        # type-based mask: only target planets or fleets (skip sun + global)
        # We can read the type from the entity feature: index 0..3 are type onehots
        ent = torch.from_numpy(enc.entities[:, :4]).to(t_logits.device)  # (N, 4)
        is_planet_or_fleet = (ent[:, 0] + ent[:, 1]) > 0.5
        valid = valid & is_planet_or_fleet
        if not valid.any():
            # Force a no-op for this planet
            launch_sampled[k] = False
            launch_logp[k] = torch.log((1.0 - p_launch).clamp_min(1e-8)).detach()
            continue
        masked_logits = t_logits.masked_fill(~valid, float("-inf"))
        t_probs = F.softmax(masked_logits, dim=-1)
        if deterministic:
            tgt_slot = int(torch.argmax(t_probs).item())
        else:
            tgt_slot = int(torch.multinomial(t_probs, 1).item())
        target_idx[k] = tgt_slot
        target_logp[k] = torch.log(t_probs[tgt_slot].clamp_min(1e-8)).detach()

        # 3. Fraction
        f_logits = fraction_l[slot]  # (F,)
        f_probs = F.softmax(f_logits, dim=-1)
        if deterministic:
            f_idx = int(torch.argmax(f_probs).item())
        else:
            f_idx = int(torch.multinomial(f_probs, 1).item())
        fraction_idx[k] = f_idx
        fraction_logp[k] = torch.log(f_probs[f_idx].clamp_min(1e-8)).detach()

        # ---- Decode into actual move ----
        sx, sy = pos_by_pid[pid]
        sr = radius_by_pid[pid]
        # Look up target slot → could be planet or fleet
        if enc.planet_slot_ids[tgt_slot] >= 0:
            tpid = int(enc.planet_slot_ids[tgt_slot])
            tx, ty = pos_by_pid.get(tpid, (CENTER, CENTER))
        elif enc.fleet_slot_ids[tgt_slot] >= 0:
            # Target a fleet: use its current position (engine will handle it
            # being a moving target — we just shoot at where it is).
            fleets_raw = raw_obs["fleets"] if isinstance(raw_obs, dict) else raw_obs.fleets
            tfid = int(enc.fleet_slot_ids[tgt_slot])
            f_pos = next(((float(f[2]), float(f[3])) for f in fleets_raw if f[0] == tfid), None)
            if f_pos is None:
                continue
            tx, ty = f_pos
        else:
            continue

        angle = _safe_angle(sx, sy, tx, ty)

        # Ships: fraction of current garrison, integer, at least 1
        current_ships = int(ships_by_pid.get(pid, 0))
        if current_ships < 1:
            continue
        send = max(1, int(round(SHIP_FRACTIONS[f_idx] * current_ships)))
        send = min(send, current_ships)

        moves.append([int(pid), float(angle), int(send)])
        n_launches += 1

    record = {
        "my_slots": torch.from_numpy(my_slots).long(),
        "launch_sampled": launch_sampled,
        "launch_logp": launch_logp,
        "target_idx": target_idx,
        "target_logp": target_logp,
        "fraction_idx": fraction_idx,
        "fraction_logp": fraction_logp,
        "value": value,
        "n_launches": n_launches,
    }
    return moves, record
