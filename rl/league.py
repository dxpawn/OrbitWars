"""Opponent league: pool of agents to sample from during training.

Mix of:
  - Real Kaggle adversaries (file paths) from other_adversaries/
  - In-house heuristics from opponents/
  - Past snapshots of self ("frozen" checkpoints — TODO)

Sampling strategy: weighted by inverse of recent win rate against us so that
hard opponents get more attention.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

from opponents import REGISTRY as OPP_REGISTRY


@dataclass
class LeagueMember:
    name: str
    weight: float = 1.0
    wins: int = 0   # our wins against this opponent
    losses: int = 0
    draws: int = 0


@dataclass
class League:
    members: dict[str, LeagueMember] = field(default_factory=dict)

    def add(self, name: str, weight: float = 1.0):
        if name in self.members:
            return
        if name not in OPP_REGISTRY:
            raise KeyError(f"{name!r} not in opponents.REGISTRY")
        self.members[name] = LeagueMember(name=name, weight=weight)

    def record(self, name: str, win: bool, draw: bool = False):
        m = self.members.get(name)
        if m is None:
            return
        if draw:
            m.draws += 1
        elif win:
            m.wins += 1
        else:
            m.losses += 1

    def sample(self, *, prioritize_hard: bool = True, rng: random.Random | None = None) -> str:
        rng = rng or random
        if not self.members:
            raise RuntimeError("League is empty")
        names = list(self.members)
        if not prioritize_hard:
            return rng.choices(names, weights=[m.weight for m in self.members.values()], k=1)[0]
        # Prioritize opponents we lose against most often
        weights = []
        for name in names:
            m = self.members[name]
            games = m.wins + m.losses + m.draws
            if games < 5:
                # Sample warmup more often
                w = m.weight * 2.0
            else:
                loss_rate = m.losses / max(1, games)
                # Higher loss_rate → higher weight, but floor at base weight
                w = m.weight * (0.5 + 1.5 * loss_rate)
            weights.append(w)
        return rng.choices(names, weights=weights, k=1)[0]

    def stats(self) -> dict[str, dict]:
        return {
            name: {
                "wins": m.wins,
                "losses": m.losses,
                "draws": m.draws,
                "games": m.wins + m.losses + m.draws,
                "weight": m.weight,
            }
            for name, m in self.members.items()
        }


def default_league() -> League:
    """Build a sensible starting league: all real adversaries + a few weak baselines."""
    L = League()
    # Real adversaries — these are the BAR.
    for name in ("adv_distance", "adv_lbmax", "adv_structured", "adv_rf_v0", "adv_rf_v1", "adv_rf_v2"):
        if name in OPP_REGISTRY:
            L.add(name, weight=1.5)
    # In-house heuristics — easy wins to build confidence + provide variety.
    for name in ("heuristic_v1", "nearest_sniper", "rusher", "defender", "random"):
        if name in OPP_REGISTRY:
            L.add(name, weight=1.0)
    return L
