"""PPO training step for Orbit Wars.

Treats each environment step as one sample. A step's "action" is the joint
choice of (launch, target, fraction) across all owned planets, and its
log-prob is the sum of independent log-probs.

The policy net is re-evaluated on stored (entities, mask, globals) under
the current policy. New log-probs are computed by reading off the entries
chosen by old log-probs (selection indices stored in the trajectory).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float):
    """Generalized Advantage Estimation.

    rewards: (T,)  per-step reward (already includes terminal at last step)
    values:  (T,)  V(s_t)
    Returns:
        adv: (T,) advantage estimates
        ret: (T,) discounted returns (adv + value)
    """
    T = len(rewards)
    if T == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        # Use bootstrap V(s_{t+1}) for t<T-1, else 0 (episode ended)
        next_v = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret


def flatten_trajectories(trajs):
    """Combine multiple trajectories into a single replay batch.

    Each step keeps a *variable-length* list of (planet, factor) decisions.
    We flatten everything per-step but keep the joint log-prob in a 1D array.
    """
    all_entities = []
    all_mask = []
    all_globals = []
    all_old_logp = []
    all_adv = []
    all_ret = []
    # Per-step action records (lists of tensors, since K is variable)
    all_my_slots = []
    all_launch_sampled = []
    all_target_idx = []
    all_fraction_idx = []

    for tj in trajs:
        T = tj["T"]
        if T == 0:
            continue
        rewards = tj["reward"]
        values = tj["value"]
        adv, ret = compute_gae(rewards, values, gamma=0.997, lam=0.95)

        # joint logp per step = sum of launch_logp + target_logp + fraction_logp across launching planets
        for t in range(T):
            ls = tj["launch_sampled"][t]
            llp = tj["launch_logp"][t]
            tlp = tj["target_logp"][t]
            flp = tj["fraction_logp"][t]
            # joint = sum over all owned planets of (launch_logp + cond_target_logp + cond_frac_logp)
            # Target and fraction logp are 0 if no launch happened (set above)
            joint = llp.sum() + tlp.sum() + flp.sum()
            all_old_logp.append(joint.item())

        all_entities.append(tj["entities"])
        all_mask.append(tj["mask"])
        all_globals.append(tj["globals"])
        all_adv.append(adv)
        all_ret.append(ret)
        all_my_slots.extend(tj["my_slots"])
        all_launch_sampled.extend(tj["launch_sampled"])
        all_target_idx.extend(tj["target_idx"])
        all_fraction_idx.extend(tj["fraction_idx"])

    if not all_entities:
        return None

    return {
        "entities": np.concatenate(all_entities, axis=0),
        "mask": np.concatenate(all_mask, axis=0),
        "globals": np.concatenate(all_globals, axis=0),
        "old_logp": np.array(all_old_logp, dtype=np.float32),
        "adv": np.concatenate(all_adv, axis=0),
        "ret": np.concatenate(all_ret, axis=0),
        "my_slots": all_my_slots,
        "launch_sampled": all_launch_sampled,
        "target_idx": all_target_idx,
        "fraction_idx": all_fraction_idx,
    }


def _step_logp_entropy(net_out: dict, my_slots, launch_sampled, target_idx, fraction_idx):
    """Compute joint logp + entropy for ONE batched step.

    net_out: forward pass output of (B=1, ...) tensors.
    my_slots: tensor (K,)
    launch_sampled: tensor (K,) bool
    target_idx: tensor (K,) long (-1 if no launch)
    fraction_idx: tensor (K,) long

    Returns: scalar logp, scalar entropy (per-step).
    """
    launch_l = net_out["launch_logit"][0]      # (N,)
    target_l = net_out["target_logits"][0]     # (N, N)
    fraction_l = net_out["fraction_logits"][0] # (N, F)

    if my_slots.numel() == 0:
        return torch.zeros((), device=launch_l.device), torch.zeros((), device=launch_l.device)

    sel_launch = launch_l[my_slots]  # (K,)
    p_launch = torch.sigmoid(sel_launch)
    # Bernoulli logp
    sampled_f = launch_sampled.float().to(p_launch.device)
    logp_launch = (
        sampled_f * torch.log(p_launch.clamp_min(1e-8))
        + (1.0 - sampled_f) * torch.log((1.0 - p_launch).clamp_min(1e-8))
    )
    # Entropy of Bernoulli
    ent_launch = -(
        p_launch * torch.log(p_launch.clamp_min(1e-8))
        + (1.0 - p_launch) * torch.log((1.0 - p_launch).clamp_min(1e-8))
    )

    # Target & fraction logps only count for planets that launched
    logp_target = torch.zeros_like(logp_launch)
    logp_frac = torch.zeros_like(logp_launch)
    ent_target = torch.zeros_like(logp_launch)
    ent_frac = torch.zeros_like(logp_launch)

    K = my_slots.shape[0]
    for k in range(K):
        if not launch_sampled[k]:
            continue
        slot = my_slots[k]
        # Target
        t_logits = target_l[slot]  # (N,)
        # The same mask used during sampling: valid mask was based on slot ent type
        # but since we stored target_idx, just use softmax over masked logits.
        # For simplicity here, we recompute log-softmax with -inf at padded keys.
        t_log_softmax = F.log_softmax(t_logits, dim=-1)
        t_probs = t_log_softmax.exp()
        idx = target_idx[k].clamp_min(0)
        if 0 <= int(target_idx[k]) < t_logits.shape[0]:
            logp_target[k] = t_log_softmax[idx]
            ent_target[k] = -(t_probs * t_log_softmax).sum()
        # Fraction
        f_logits = fraction_l[slot]
        f_log_softmax = F.log_softmax(f_logits, dim=-1)
        f_probs = f_log_softmax.exp()
        logp_frac[k] = f_log_softmax[fraction_idx[k]]
        ent_frac[k] = -(f_probs * f_log_softmax).sum()

    return (
        (logp_launch + logp_target + logp_frac).sum(),
        (ent_launch + ent_target + ent_frac).sum(),
    )


def ppo_update(
    policy,
    optimizer,
    batch,
    *,
    device,
    epochs: int = 4,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
    minibatch_size: int = 64,
):
    """Run PPO update over `batch` (output of flatten_trajectories)."""
    T = len(batch["old_logp"])
    if T == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

    entities = torch.from_numpy(batch["entities"]).to(device)
    mask = torch.from_numpy(batch["mask"]).to(device)
    gl = torch.from_numpy(batch["globals"]).to(device)
    old_logp = torch.from_numpy(batch["old_logp"]).to(device)
    adv = torch.from_numpy(batch["adv"]).to(device)
    ret = torch.from_numpy(batch["ret"]).to(device)

    # Normalize advantages
    adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))

    indices = torch.arange(T)
    metrics_acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    n_updates = 0

    for _ in range(epochs):
        shuffled = indices[torch.randperm(T)]
        for mb_start in range(0, T, minibatch_size):
            mb_idx = shuffled[mb_start: mb_start + minibatch_size]
            if len(mb_idx) == 0:
                continue

            # Forward pass on the minibatch
            net_out = policy(
                entities[mb_idx], mask[mb_idx], gl[mb_idx]
            )

            # Compute new joint logp + entropy step-by-step
            new_logps = []
            entropies = []
            for j, t_idx in enumerate(mb_idx.tolist()):
                # Reconstruct per-step out
                step_out = {
                    "launch_logit": net_out["launch_logit"][j: j + 1],
                    "target_logits": net_out["target_logits"][j: j + 1],
                    "fraction_logits": net_out["fraction_logits"][j: j + 1],
                    "value": net_out["value"][j: j + 1],
                }
                lp, ent = _step_logp_entropy(
                    step_out,
                    batch["my_slots"][t_idx].to(device),
                    batch["launch_sampled"][t_idx].to(device),
                    batch["target_idx"][t_idx].to(device),
                    batch["fraction_idx"][t_idx].to(device),
                )
                new_logps.append(lp)
                entropies.append(ent)
            new_logp = torch.stack(new_logps)
            ent = torch.stack(entropies)
            value = net_out["value"]

            old_lp_mb = old_logp[mb_idx]
            adv_mb = adv[mb_idx]
            ret_mb = ret[mb_idx]

            ratio = torch.exp(new_logp - old_lp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret_mb)
            entropy = ent.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            approx_kl = (old_lp_mb - new_logp).mean().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            metrics_acc["loss"] += float(loss.item())
            metrics_acc["policy_loss"] += float(policy_loss.item())
            metrics_acc["value_loss"] += float(value_loss.item())
            metrics_acc["entropy"] += float(entropy.item())
            metrics_acc["approx_kl"] += float(approx_kl)
            n_updates += 1

    n_updates = max(1, n_updates)
    return {k: v / n_updates for k, v in metrics_acc.items()}
