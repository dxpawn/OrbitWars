"""Single-match runner. Wraps kaggle_environments.make + env.run."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from eval._quiet import make  # quiet import wrapper


AgentSpec = str | Callable
"""An agent is either a Python callable (obs -> moves) or a registry name."""


@dataclass
class MatchResult:
    seed: int
    agents: tuple[str, ...]
    rewards: tuple[float, ...]
    scores: tuple[float, ...]
    winner: int  # -1 if no winner (tie or everyone eliminated)
    winners: tuple[int, ...]  # all players tied for first
    steps: int
    duration_s: float
    replay: dict | None = None


def resolve_agent(spec: AgentSpec) -> Callable:
    """Convert a name or callable into an agent function.

    Names are first looked up in the opponents registry, then tried as
    kaggle_environments built-ins (e.g. 'random', 'starter').
    """
    if callable(spec):
        return spec
    if isinstance(spec, str):
        try:
            opponents = importlib.import_module("opponents")
            if spec in opponents.REGISTRY:
                return opponents.REGISTRY[spec]
        except (ImportError, AttributeError):
            pass
        # Fall back to env built-ins: pass the string through; env.run accepts it.
        return spec
    raise TypeError(f"Cannot resolve agent from {spec!r}")


def _agent_name(spec: AgentSpec) -> str:
    if callable(spec):
        return getattr(spec, "__module__", "callable").split(".")[-1]
    return str(spec)


def run_match(
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    seed: int = 0,
    *,
    keep_replay: bool = False,
    extra_agents: Sequence[AgentSpec] | None = None,
    episode_steps: int | None = None,
) -> MatchResult:
    """Play one match.

    Args:
        agent_a, agent_b: the two main agents (P0, P1).
        seed: RNG seed for the env.
        keep_replay: if True, attach the full env.toJSON() replay to the result.
        extra_agents: for 4-player games, supply 2 more agents.
        episode_steps: override episode length (default: env default of 500).

    Returns:
        MatchResult with rewards, scores, winner index, steps, duration.
    """
    config = {"seed": seed}
    if episode_steps is not None:
        config["episodeSteps"] = episode_steps
    env = make("orbit_wars", configuration=config, debug=False)
    agents = [resolve_agent(agent_a), resolve_agent(agent_b)]
    if extra_agents:
        agents.extend(resolve_agent(a) for a in extra_agents)

    t0 = time.time()
    env.run(agents)
    duration = time.time() - t0

    final = env.steps[-1]
    rewards = tuple(s.reward for s in final)

    obs0 = final[0].observation
    n = len(rewards)
    scores = [0.0] * n
    for p in obs0.planets:
        if p[1] != -1 and p[1] < n:
            scores[p[1]] += p[5]
    for f in obs0.fleets:
        if f[1] < n:
            scores[f[1]] += f[6]

    max_score = max(scores) if scores else 0
    winners = tuple(i for i, s in enumerate(scores) if s == max_score and max_score > 0)
    winner = winners[0] if len(winners) == 1 else -1

    names = tuple(_agent_name(s) for s in ([agent_a, agent_b] + list(extra_agents or [])))

    replay = env.toJSON() if keep_replay else None

    return MatchResult(
        seed=seed,
        agents=names,
        rewards=rewards,
        scores=tuple(scores),
        winner=winner,
        winners=winners,
        steps=len(env.steps),
        duration_s=duration,
        replay=replay,
    )


if __name__ == "__main__":
    # Quick smoke test
    r = run_match("nearest_sniper", "random", seed=42)
    print(f"Result: winner={r.winner}, scores={r.scores}, "
          f"steps={r.steps}, duration={r.duration_s:.2f}s")
