"""Supervised pretraining: clone the winning adversaries' policy.

For each step (obs, moves) in collected games, we:
  1. Encode obs to entity tensor.
  2. Convert moves → per-owned-planet (launch, target_slot, fraction_bin) labels.
  3. Cross-entropy loss on launch (Bernoulli), target (categorical over entities),
     fraction (categorical over 5 bins).

The result is a policy net that approximately reproduces strong-adversary
behavior. Use it as PPO warm-start.
"""

from __future__ import annotations

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from rl.features import (
    ENTITY_DIM, GLOBAL_DIM, MAX_ENTITIES, encode,
)
from rl.policy import OrbitWarsPolicy, SHIP_FRACTIONS, N_FRACTIONS


def _angle_diff(a, b):
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d = 2 * math.pi - d
    return abs(d)


def _label_move(move, src_planet, enc):
    """Given a move [from_id, angle, ships] from src_planet, find the most
    likely (target_slot, fraction_idx) the adversary was aiming at.

    Returns (target_slot, fraction_idx) or (None, None) if can't infer.
    """
    _, angle, ships = move
    sx, sy = src_planet[2], src_planet[3]

    # For each candidate (planet or fleet), compute the angle from src to it
    best_slot, best_delta = None, math.pi  # 180° max
    for slot in range(MAX_ENTITIES):
        if not enc.mask[slot]:
            continue
        # Get entity type from encoded features
        is_planet = enc.entities[slot, 0] > 0.5
        is_fleet = enc.entities[slot, 1] > 0.5
        if not (is_planet or is_fleet):
            continue
        # Get position
        x = enc.entities[slot, 9] * 100.0
        y = enc.entities[slot, 10] * 100.0
        # Skip self
        if is_planet and enc.planet_slot_ids[slot] == src_planet[0]:
            continue
        target_angle = math.atan2(y - sy, x - sx)
        delta = _angle_diff(angle, target_angle)
        if delta < best_delta:
            best_delta = delta
            best_slot = slot

    if best_slot is None or best_delta > math.radians(30):
        # Too far off — adversary's intercept solver may have used predicted position
        # that doesn't match current. Still pick best slot but flag.
        if best_slot is None:
            return None, None

    # Fraction: ships / src.ships at this moment
    src_ships = max(1, int(src_planet[5]))
    frac = ships / src_ships
    # Find nearest bin
    diffs = [abs(frac - f) for f in SHIP_FRACTIONS]
    frac_idx = int(np.argmin(diffs))
    return best_slot, frac_idx


def step_to_labels(obs_dict, moves):
    """Convert one (obs, moves) pair into supervised labels.

    Returns dict with:
        entities: (N, E) float32
        mask: (N,) bool
        globals: (G,) float32
        my_slots: (K,) int
        launch_label: (K,) bool  — did the adversary launch from this planet this turn?
        target_label: (K,) int   — target_slot if launched, -1 otherwise
        fraction_label: (K,) int — fraction bin if launched, -1 otherwise
    """
    enc = encode(obs_dict)
    K = len(enc.my_planet_slots)
    if K == 0:
        return None

    planet_by_id = {p[0]: p for p in obs_dict["planets"]}

    # Map planet_id → first move from that planet
    move_by_pid = {}
    for m in moves:
        if len(m) != 3:
            continue
        pid = int(m[0])
        if pid in move_by_pid:
            continue
        move_by_pid[pid] = m

    launch_label = np.zeros(K, dtype=bool)
    target_label = np.full(K, -1, dtype=np.int64)
    fraction_label = np.full(K, -1, dtype=np.int64)

    for k, (slot, pid) in enumerate(zip(enc.my_planet_slots.tolist(), enc.my_planet_ids.tolist())):
        if pid not in move_by_pid:
            continue
        src = planet_by_id.get(pid)
        if src is None:
            continue
        tgt_slot, frac_idx = _label_move(move_by_pid[pid], src, enc)
        if tgt_slot is None:
            continue
        launch_label[k] = True
        target_label[k] = tgt_slot
        fraction_label[k] = frac_idx

    return {
        "entities": enc.entities,
        "mask": enc.mask,
        "globals": enc.globals_,
        "my_slots": np.array(enc.my_planet_slots, dtype=np.int64),
        "launch_label": launch_label,
        "target_label": target_label,
        "fraction_label": fraction_label,
    }


class ImitationDataset(IterableDataset):
    """Yields (obs_encoded, labels) tuples from a pile of pickled trajectories."""

    def __init__(self, data_dir: Path, *, only_winners: bool = True, max_files: int | None = None):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pkl"))
        if max_files is not None:
            self.files = self.files[:max_files]
        self.only_winners = only_winners

    def _iter_pairs(self, payload):
        if self.only_winners:
            if payload["winner_idx"] == 0:
                yield from payload["log_a"]
            elif payload["winner_idx"] == 1:
                yield from payload["log_b"]
            # if draw/no-winner, skip both
        else:
            yield from payload["log_a"]
            yield from payload["log_b"]

    def __iter__(self):
        # Shuffle file order each epoch
        order = list(range(len(self.files)))
        np.random.shuffle(order)
        for i in order:
            try:
                with open(self.files[i], "rb") as f:
                    payload = pickle.load(f)
            except (OSError, EOFError, pickle.UnpicklingError):
                continue
            for obs, moves in self._iter_pairs(payload):
                sample = step_to_labels(obs, moves)
                if sample is None or len(sample["my_slots"]) == 0:
                    continue
                yield sample


def collate(batch):
    """Variable-K per sample. Stack obs tensors, keep labels as lists."""
    entities = torch.from_numpy(np.stack([b["entities"] for b in batch]))
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch]))
    gl = torch.from_numpy(np.stack([b["globals"] for b in batch]))
    my_slots = [torch.from_numpy(b["my_slots"]) for b in batch]
    launch_label = [torch.from_numpy(b["launch_label"]) for b in batch]
    target_label = [torch.from_numpy(b["target_label"]) for b in batch]
    fraction_label = [torch.from_numpy(b["fraction_label"]) for b in batch]
    return entities, mask, gl, my_slots, launch_label, target_label, fraction_label


def imitation_loss(net_out, my_slots, launch_label, target_label, fraction_label):
    """Compute supervised loss for one batch."""
    B = net_out["launch_logit"].shape[0]
    launch_l = net_out["launch_logit"]
    target_l = net_out["target_logits"]
    fraction_l = net_out["fraction_logits"]

    total_launch_loss = 0.0
    total_target_loss = 0.0
    total_fraction_loss = 0.0
    n_launch = 0
    n_target = 0

    for b in range(B):
        slots_b = my_slots[b].to(launch_l.device)
        ll_b = launch_label[b].to(launch_l.device).float()
        tl_b = target_label[b].to(launch_l.device)
        fl_b = fraction_label[b].to(launch_l.device)
        if slots_b.numel() == 0:
            continue

        # Launch: BCE on each slot
        sel_launch = launch_l[b, slots_b]
        loss_launch = F.binary_cross_entropy_with_logits(sel_launch, ll_b, reduction="sum")
        total_launch_loss = total_launch_loss + loss_launch
        n_launch += slots_b.numel()

        # Target + fraction: only on slots where launch_label is True
        launched_mask = ll_b > 0.5
        if launched_mask.any():
            launched_slots = slots_b[launched_mask]
            launched_tl = tl_b[launched_mask]
            launched_fl = fl_b[launched_mask]
            # Target: cross-entropy over N entities for each launched planet
            sel_target_logits = target_l[b, launched_slots]  # (k', N)
            loss_target = F.cross_entropy(sel_target_logits, launched_tl, reduction="sum")
            total_target_loss = total_target_loss + loss_target
            # Fraction
            sel_frac_logits = fraction_l[b, launched_slots]
            loss_frac = F.cross_entropy(sel_frac_logits, launched_fl, reduction="sum")
            total_fraction_loss = total_fraction_loss + loss_frac
            n_target += launched_slots.numel()

    if n_launch == 0:
        return None, {}

    loss_launch_avg = total_launch_loss / max(1, n_launch)
    loss_target_avg = total_target_loss / max(1, n_target) if n_target > 0 else torch.tensor(0.0, device=launch_l.device)
    loss_frac_avg = total_fraction_loss / max(1, n_target) if n_target > 0 else torch.tensor(0.0, device=launch_l.device)
    total = loss_launch_avg + loss_target_avg + loss_frac_avg
    return total, {
        "launch": float(loss_launch_avg),
        "target": float(loss_target_avg) if n_target > 0 else 0.0,
        "fraction": float(loss_frac_avg) if n_target > 0 else 0.0,
        "n_launch": n_launch,
        "n_target": n_target,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="state/imitation_data")
    parser.add_argument("--checkpoint", default="checkpoints/imitation.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    policy = OrbitWarsPolicy(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    ds = ImitationDataset(args.data_dir)
    print(f"Found {len(ds.files)} game files in {args.data_dir}", flush=True)

    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in dl:
            entities, mask, gl, my_slots, launch_label, target_label, fraction_label = batch
            entities = entities.to(device)
            mask = mask.to(device)
            gl = gl.to(device)

            out = policy(entities, mask, gl)
            loss, metrics = imitation_loss(out, my_slots, launch_label, target_label, fraction_label)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"epoch {epoch} step {step:>5d}  loss={float(loss):.4f}  L_launch={metrics['launch']:.3f}  L_target={metrics['target']:.3f}  L_frac={metrics['fraction']:.3f}  n_launch={metrics['n_launch']}  ({el:.0f}s)", flush=True)

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "step": step,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
    }, ckpt_path)
    print(f"Saved {ckpt_path} (step={step}, wall={time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
