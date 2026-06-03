"""Heuristic v6 — heuristic_v5 + a forward-projection "brain".

This is our own reimplementation of the decision core that separates the public
~1000-1100 LB agent (other_adversaries/HEURISTIC1000.py) from our 970 hellburner
lineage. We keep ALL of v5's proven machinery — obs parsing, orbital geometry,
intercept/aim, sun-blocking, the per-planet combat sim (simulate_planet_timeline),
candidate generation (evaluate_frontline_strategy), the early-game DFS, and the
mode-aware reach (MAX_DISTANCE 38 in 1v1 / 30 in FFA) — and replace ONLY the
mid-game decision rule.

v5's mid-game was greedy: pick the single reachable planet of highest production
we can win, commit, repeat. That is myopic — no notion of the resulting board
position, and it ignores who is winning.

v6 instead does a 1-ply search over a global value function:
  1. forward_project(): project EVERY planet's (owner, ships) forward FWD_HORIZON
     turns at once — production growth, in-flight fleet arrivals resolved with the
     engine's simultaneous-combat math, plus "phantom" opponent launches (each
     live planet periodically flings a fraction of its surplus at its nearest
     non-friendly target) so we don't grab planets that get instantly sniped back.
  2. forward_score(): score a projected board LEADER-RELATIVE — our advantage over
     the single strongest opponent in ships + 5*planets + 8*production. This
     matches Kaggle's win condition (single highest score wins), which v5's
     absolute production heuristic does not.
  3. plan_midgame(): for each candidate capture/defense (concrete fleet orders from
     v5's evaluate_frontline_strategy), project the board WITH that action applied
     and keep the action with the best score gain vs doing nothing. Commit, repeat
     until no positive-gain action remains or the per-turn time budget is hit.

Everything is bounded by a soft deadline (SEARCH_SOFT_BUDGET) so the heavier sim
never risks the 1.0s actTimeout. v5's reinforcement pass still runs afterward.

v5/v2's bug-fixes (inherited) vs the original other_adversaries/hellburner.py:
  1. Removed the `viz.add_text(...)` debug call inside run_early_game.
  2. Removed the dangling `elapsed_ms = (time.perf_counter() - _t0) * 1000`.
"""

import math
import os
import time
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Fleet, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
    distance, point_to_segment_distance
)

# --- persistent hammer state (survives across turns; agent() re-instantiates the
# Hellburner each turn, so cross-turn memory must live at module scope). Keyed by
# obs['player'] so two seats sharing this module in local self-play never collide;
# reset per-player when a new game restarts the step counter (see main()). ---
_HAMMER_PLANS: dict = {}       # player -> active plan dict (or absent)
_HAMMER_LAST_STEP: dict = {}   # player -> last scene_step seen (for game-restart detection)

class HPlanet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.radius = radius; self.ships = ships; self.production = production
        self.reinforcement_target: 'HPlanet | None' = None  # nearest owned planet on shortest path to front

# HPlanet -> (orbital_radius, initial_angle) if the planet orbits the sun, else None
OrbitalInfo = dict[HPlanet, tuple[float, float] | None]
# HPlanets rotated by ROTATION_LOOK_AHEAD
FuturePos = dict[HPlanet, tuple[float, float]]
# dst -> [(src, travel_steps)]: directed graph; src departs now, dst is its intercept position
ProximityGraph = dict[HPlanet, list[tuple[HPlanet, float]]]
# HPlanet -> [(owner, ships, travel_time, src_x, src_y, arrival_x, arrival_y)]
DestinationList = dict[HPlanet, list[tuple[int, float, float, float, float, float, float]]]
# [planet_id, angle, ships]
FleetOrders = list[list]
# (intercept_x, intercept_y, travel_steps)
Intercept = tuple[float, float, float]
# (target planet, heuristic value, fleet orders, intercepts)
# intercepts is parallel to fleet_orders: list of (ix, iy, travel) pre-computed at plan time
MoveOrders = tuple[HPlanet | None, int, FleetOrders, list[Intercept]]

@dataclass(slots=True)
class EarlyGameFleet:
    source_id: int
    destination_id: int
    fleet_size: int
    garrison_on_arrival: int
    arrival_turn: int
    is_capture: bool

@dataclass(slots=True)
class EarlyGameState:
    turn: int
    garrison: dict
    production: dict
    owned: set
    fleets: list = field(default_factory=list)


class Hellburner:
    SHIP_SPEED_MAX: float = 6.0
    EARLY_ROUNDS: int = 3
    EARLY_LOOK_AHEAD: int = 33
    MAX_DISTANCE: int = 38       # 1v1 reach (v2's value; used when n_sides <= 2)
    MAX_DISTANCE_MP: int = 38    # 3p/4p reach. REVERTED 30->38: v5's reach-30 won
                                 # local 4p FFA but REGRESSED the real ladder
                                 # (v5=918 < v2=970). Reach 38 everywhere = v2's
                                 # proven base, so the ONLY diff from v2 is the brain.
    ROTATION_LOOK_AHEAD: int = 10
    REINFORCEMENT_SIZE: int = 17
    GARRISON_SIZE: int = 11

    # --- forward-projection brain (v6) ---
    FWD_HORIZON: int = 18                       # turns to project the whole board
    FWD_SNAPSHOT_TURNS: tuple = (4, 8, 13, 18)  # score is averaged over these horizons
    FWD_EMIT_FRAC: float = 0.10                 # phantom-launch surplus fraction; swept +
                                                # held-out confirmed (3 seed ranges: +7/+11/+8
                                                # net vs 0.20). 0.10 is the peak; lower turns
                                                # snipe-blind, higher is over-pessimistic.
    VAL_PLANET_W: float = 5.0                   # value of a planet-count lead (in ships)
    VAL_PROD_W: float = 8.0                     # value of a production lead (in ships)
    FWD_SELF_EMIT: float = 0.5                  # phantom self-launch rate vs opponents' full
                                                # rate. 0.5 (default/1017) = the brain models
                                                # ITSELF launching at half the opponents' rate
                                                # → asymmetric pessimism that makes projected
                                                # captures look snipe-able and FREEZES the
                                                # midgame (the diagnosed 2p stall). Higher =
                                                # less self-pessimistic. Structural, untested.
    SEARCH_SOFT_BUDGET: float = 0.85            # s; per-turn deadline (actTimeout is 1.0)
    SEARCH_MAX_ACTIONS: int = 8                 # cap committed actions per turn
    SEARCH_MAX_ACTIONS_2P: int = 8              # 2p-only commit cap (default 8 = byte-identical off).
                                                # PORT of exp30 (LB~1072) = HEURISTIC1000 with its
                                                # SEARCH_MAX_ACTIONS_TO_PICK_2P raised 7->9 (the ONLY
                                                # diff, +~30-70 LB on the real ladder). Targets v6's
                                                # diagnosed 2p midgame stall: let it commit MORE
                                                # captures/turn in 2p. 4p path untouched (n_sides!=2).
    SEARCH_MIN_GAIN: float = 1e-6               # only commit actions with positive score gain

    # --- 2p tactical layer (ported from HEURISTIC1000), env-gated, default OFF ---
    # The brain (forward-projection + leader-relative search) beats v2 by +105 on
    # the ladder, but loses 2p h2h to H1000: diag_2p showed v6 stalls and bleeds
    # planets in the midgame (steps 75-150). These two levers target exactly that,
    # gated to 2p (n_sides==2) so 4p behaviour is byte-identical to the 1017 agent.
    OVERSEND_2P: int = 0    # 2p: skip the capture-fleet trim → land full force (holds vs snipe-back)
    PRESS_2P: int = 0       # 2p: after value search, press hold-able high-prod captures (anti-stall)
    PRESS_2P_MAX: int = 3   # max extra captures the press pass may commit per turn
    PRESS_2P_MIN_PROD: int = 2  # ignore prod<this junk targets in the press pass
    DEF_PRESSURE_FRAC: float = 0.5  # 2p: share of adjacent enemy strength treated as
                                    # counterattack pressure when sizing a source's
                                    # held-back garrison (0.5 = v6 default; higher = more defensive)

    # --- persistent staggered HAMMER (ported from HEURISTIC1000), env-gated, default OFF ---
    # v6's brain fires every contributing fleet in ONE turn, so fleets from
    # different-distance sources arrive on DIFFERENT turns and a reinforcing
    # defender beats them piecemeal — v6's structural 2p weakness. The hammer
    # commits a MULTI-TURN plan: it picks a high-prod enemy target and a set of
    # stockpiles, then staggers each source's launch turn so the whole combined
    # fleet LANDS ON ONE TURN, overwhelming the defender (forecast at arrival).
    # Requires cross-turn memory (module globals below), which v6 lacks by default.
    HAMMER_2P: int = 0          # master toggle (env V6_HAMMER); OFF => byte-identical to 1017 agent
    HAMMER_MIN_PROD: int = 2    # only hammer enemy planets with production >= this
    HAMMER_OVERKILL: float = 1.25   # commit >= forecast-defender-at-arrival * this
    HAMMER_MAX_TRAVEL: int = 40     # ignore sources whose travel to target exceeds this (turns)
    HAMMER_MIN_CONTRIB: int = 12    # a stockpile must contribute >= this many ships
    HAMMER_MAX_SOURCES: int = 4     # cap contributors per plan
    HAMMER_ABORT_RATIO: float = 1.1  # abort plan if defender_at_arrival > committed/this
    HAMMER_PROD_LEAD: int = 0   # only open a plan when (my_prod - leader_prod) >= this

    # --- LB1050 "council" value/search refinements (env-gated, default OFF) ---
    # Ported from the public 1050 agent, which shares v6's EXACT brain + value
    # function — these are pure selection tweaks, not the aggression mechanics that
    # failed. SNAP_WEIGHT: weight near-term snapshots by 1/t (late projections carry
    # more cascaded error). ARR_DECAY: in 2p, discount a capture's score gain by
    # decay**arrival_turn so sooner-landing captures are preferred.
    SNAP_WEIGHT: int = 0        # 0 = equal weight (1017); 1 = 1/t weighting (LB1050)
    ARR_DECAY: float = 1.0      # 1.0 = off (1017); LB1050 uses 0.97 in 2p

    # --- LB1050 DEPTH-2 counter-response penalty (env-gated, default OFF) ---
    # The substantive LB1050 lever. For the top-K candidate captures, project the
    # nearest strong enemies' counterattack at the captured planet; if they retake
    # it, penalise the action by their strength. A pure SELECTION refinement that
    # targets v6's diagnosed "bleed" (captures sniped back) — not aggression.
    DEPTH2: int = 0             # master toggle (env V6_DEPTH2); 2p-only
    DEPTH2_TOPK: int = 3        # how many top candidates to counter-check per search step
    DEPTH2_OPPS: int = 2        # how many nearest enemies to test per candidate
    DEPTH2_RADIUS: float = 30.0  # only enemies within this distance can counter in time

    def __init__(self):
        self._start_time: float = 0.0
        # Brain knobs: env overrides for sweeping. Unset => class defaults
        # (identical behaviour). eval/sweep_v6.py sets these to explore.
        self.FWD_HORIZON = _envi("V6_FWD_HORIZON", Hellburner.FWD_HORIZON)
        self.FWD_EMIT_FRAC = _envf("V6_EMIT_FRAC", Hellburner.FWD_EMIT_FRAC)
        self.VAL_PLANET_W = _envf("V6_PLANET_W", Hellburner.VAL_PLANET_W)
        self.VAL_PROD_W = _envf("V6_PROD_W", Hellburner.VAL_PROD_W)
        self.FWD_SELF_EMIT = _envf("V6_SELF_EMIT", Hellburner.FWD_SELF_EMIT)
        self.SEARCH_MIN_GAIN = _envf("V6_MIN_GAIN", Hellburner.SEARCH_MIN_GAIN)
        self.SEARCH_MAX_ACTIONS = _envi("V6_MAX_ACTIONS", Hellburner.SEARCH_MAX_ACTIONS)
        self.SEARCH_MAX_ACTIONS_2P = _envi("V6_MAX_ACTIONS_2P", Hellburner.SEARCH_MAX_ACTIONS_2P)
        # 2p tactical knobs (default off => identical to the 1017 ladder agent).
        self.OVERSEND_2P = _envi("V6_OVERSEND_2P", Hellburner.OVERSEND_2P)
        self.PRESS_2P = _envi("V6_PRESS_2P", Hellburner.PRESS_2P)
        self.PRESS_2P_MAX = _envi("V6_PRESS_MAX", Hellburner.PRESS_2P_MAX)
        self.PRESS_2P_MIN_PROD = _envi("V6_PRESS_MIN_PROD", Hellburner.PRESS_2P_MIN_PROD)
        self.DEF_PRESSURE_FRAC = _envf("V6_DEF_FRAC", Hellburner.DEF_PRESSURE_FRAC)
        # persistent hammer knobs (default off => identical to the 1017 ladder agent).
        self.HAMMER_2P = _envi("V6_HAMMER", Hellburner.HAMMER_2P)
        self.HAMMER_MIN_PROD = _envi("V6_HAMMER_MIN_PROD", Hellburner.HAMMER_MIN_PROD)
        self.HAMMER_OVERKILL = _envf("V6_HAMMER_OVERKILL", Hellburner.HAMMER_OVERKILL)
        self.HAMMER_MAX_TRAVEL = _envi("V6_HAMMER_MAX_TRAVEL", Hellburner.HAMMER_MAX_TRAVEL)
        self.HAMMER_MIN_CONTRIB = _envi("V6_HAMMER_MIN_CONTRIB", Hellburner.HAMMER_MIN_CONTRIB)
        self.HAMMER_MAX_SOURCES = _envi("V6_HAMMER_MAX_SOURCES", Hellburner.HAMMER_MAX_SOURCES)
        self.HAMMER_ABORT_RATIO = _envf("V6_HAMMER_ABORT", Hellburner.HAMMER_ABORT_RATIO)
        self.HAMMER_PROD_LEAD = _envi("V6_HAMMER_PROD_LEAD", Hellburner.HAMMER_PROD_LEAD)
        # LB1050 value/search refinements (default off => identical to 1017 agent).
        self.SNAP_WEIGHT = _envi("V6_SNAP_WEIGHT", Hellburner.SNAP_WEIGHT)
        self.ARR_DECAY = _envf("V6_ARR_DECAY", Hellburner.ARR_DECAY)
        self.DEPTH2 = _envi("V6_DEPTH2", Hellburner.DEPTH2)
        self.DEPTH2_TOPK = _envi("V6_DEPTH2_TOPK", Hellburner.DEPTH2_TOPK)
        self.DEPTH2_OPPS = _envi("V6_DEPTH2_OPPS", Hellburner.DEPTH2_OPPS)
        self.DEPTH2_RADIUS = _envf("V6_DEPTH2_RADIUS", Hellburner.DEPTH2_RADIUS)
        # source planet ids reserved for future hammer launches; skipped by the
        # value search and reinforcement pass so the stockpile stays intact.
        self._reserved_ids: set = set()
        self.planet_by_id: dict = {}
        # 1v1 reach (used when n_sides <= 2). v5 kept 38 here; H1000 reaches far
        # further (up to ~52) in 2p. Tunable to test the 2p-midgame-stall fix.
        self.MAXDIST_2P = _envi("V6_MAXDIST_2P", Hellburner.MAX_DISTANCE)
        self.player: int = 0
        self.scene_step: int = 0
        self.angular_velocity: float = 0.0
        self.planets: list[HPlanet] = []
        self.owned_planets: list[HPlanet] = []
        self.enemy_planets: list[HPlanet] = []
        self.fleets: list[Fleet] = []
        self.orbital_info: OrbitalInfo = {}
        self.inbound_edges: ProximityGraph = {}
        self.outbound_edges: ProximityGraph = {}
        self.future_pos: FuturePos = {}
        self.destination_list: DestinationList = {}
        # Number of active sides (us + distinct enemy owners). 2 in 2p. Set per
        # turn in main(); selects 1v1 vs multiplayer reach.
        self.n_sides: int = 2

    def fleet_speed(self, ships: int | float) -> float:
        return min(self.SHIP_SPEED_MAX, 1.0 + (self.SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)

    def build_orbital_info(self, initial_planets: list[Any]) -> None:
        """Return dict mapping Planet -> (r, initial_angle) if orbiting, else None."""
        cx = cy = CENTER
        ip_by_id = {ip[0]: ip for ip in initial_planets}
        self.orbital_info = {}
        for p in self.planets:
            r = distance((p.x, p.y), (cx, cy))
            if r + p.radius < ROTATION_RADIUS_LIMIT and p.id in ip_by_id:
                ip = ip_by_id[p.id]
                self.orbital_info[p] = (r, math.atan2(ip[3] - cy, ip[2] - cx))
            else:
                self.orbital_info[p] = None

    def build_proximity_graph(self) -> None:
        """Build directed adjacency list: dst -> [(src, travel_steps)].

        Directed because:
        - src departs from its current position immediately
        - dst is rotated into the future to its intercept position
        So travel from A->B and B->A may differ and one direction may exceed MAX_DISTANCE.

        future_pos stores each planet's current position (source frame).
        intercept_pos stores each planet's arrival position given a shot from the center
        (used only for visualization; actual intercepts are computed per-src in evaluate_frontline_strategy).
        """
        cx = cy = CENTER
        self.future_pos = {}
        for p in self.planets:
            orb = self.orbital_info[p]
            if orb is not None:
                r, ia = orb
                a = ia + self.angular_velocity * (self.scene_step + 1 + self.ROTATION_LOOK_AHEAD)
                self.future_pos[p] = (cx + r * math.cos(a), cy + r * math.sin(a))
            else:
                self.future_pos[p] = (p.x, p.y)

        self.inbound_edges = {p: [] for p in self.planets}
        for src in self.planets:
            for dst in self.planets:
                if dst is src:
                    continue
                travel = distance((src.x, src.y), self.future_pos[dst])
                if travel <= self.MAX_DISTANCE:
                    self.inbound_edges[dst].append((src, travel))

        # self.outbound_edges[p] = [(dst, travel)] — keyed by source, complement of the inbound-keyed inbound_edges.
        self.outbound_edges = {p: [] for p in self.planets}
        for dst, inbound in self.inbound_edges.items():
            for src, travel in inbound:
                self.outbound_edges[src].append((dst, travel))

    def build_reinforcement_targets(self) -> None:
        front_line = {
            p for p in self.owned_planets
            if any(src.owner != self.player for src, _ in self.inbound_edges[p])
            or any(dst.owner != self.player for dst, _ in self.outbound_edges[p])
        }

        # BFS hop-distance from every owned node to nearest frontline planet,
        # traversing only owned-planet edges (frontline nodes are sinks, not sources).
        hops_to_front: dict[HPlanet, int] = {p: 0 for p in front_line}
        queue: list[HPlanet] = list(front_line)
        head = 0
        while head < len(queue):
            node = queue[head]; head += 1
            for src, _ in self.inbound_edges[node]:
                if src.owner != self.player or src in hops_to_front:
                    continue
                hops_to_front[src] = hops_to_front[node] + 1
                queue.append(src)

        for p in self.owned_planets:
            p.reinforcement_target = None
            if p in front_line:
                continue

            direct_front = [
                dst for dst, _ in self.outbound_edges[p]
                if dst in front_line
            ]
            if direct_front:
                p.reinforcement_target = min(direct_front, key=lambda d: d.ships)
                continue

            # No direct edge to a frontline planet: pick the direct neighbor with
            # fewest hops to the front, breaking ties by fewest ships at destination.
            reachable = [
                dst for dst, _ in self.outbound_edges[p]
                if dst.owner == self.player and dst not in front_line and dst in hops_to_front
            ]
            if reachable:
                p.reinforcement_target = min(reachable, key=lambda d: (hops_to_front[d], d.ships))

    def intercept_planet(
        self,
        sx: float, sy: float, target: HPlanet, ships: int | float,
        tol: float = 1e-6, max_iters: int = 30,
    ) -> tuple[float, float, float, float]:
        """Aim angle from (sx, sy) toward where target will be when a fleet arrives.
        Returns (angle, intercept_x, intercept_y, travel_steps).
        """
        speed = self.fleet_speed(ships)
        orb = self.orbital_info[target]
        if orb is None:
            tx, ty = target.x, target.y
            travel = distance((sx, sy), (tx, ty)) / speed
        else:
            cx = cy = CENTER
            r, ia = orb
            # Seed: straight-line travel time to the planet's current position.
            travel = distance((sx, sy), (target.x, target.y)) / speed
            for _ in range(max_iters):
                a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
                new_tx, new_ty = cx + r * math.cos(a), cy + r * math.sin(a)
                new_travel = distance((sx, sy), (new_tx, new_ty)) / speed
                # Damp update: average old and new travel to suppress oscillation.
                new_travel = 0.5 * (travel + new_travel - 0.5)
                if abs(new_travel - travel) < tol:
                    travel = new_travel
                    break
                travel = new_travel
            else:
                # Diverged: fleet too slow to catch this planet's orbital speed.
                return 0.0, target.x, target.y, math.inf
            # Recompute final position from converged travel so tx/ty/angle are consistent.
            a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
            tx, ty = cx + r * math.cos(a), cy + r * math.sin(a)
        angle = math.atan2(ty - sy, tx - sx)
        return angle, tx, ty, travel

    def first_planet_hit(self, sx: float, sy: float, angle: float, ships: int | float, source: HPlanet) -> HPlanet | None:
        """Return the first planet a fleet launched from (sx, sy) at `angle` would hit, or None.
        Returns None if the path crosses the sun before any planet is hit."""
        best = None
        best_t = float('inf')
        for planet in self.planets:
            if planet is source:
                continue
            needed_angle, px, py, travel = self.intercept_planet(sx, sy, planet, ships)
            dist = distance((sx, sy), (px, py))
            if dist < planet.radius:
                half_cone = math.pi
            else:
                half_cone = math.asin(min(1.0, planet.radius / dist))
            delta = abs(math.atan2(math.sin(angle - needed_angle), math.cos(angle - needed_angle)))
            if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                best_t = travel
                best = planet
        if best is None:
            return None
        # Check if the sun blocks the path to the first planet hit.
        ex, ey = sx + best_t * self.fleet_speed(ships) * math.cos(angle), sy + best_t * self.fleet_speed(ships) * math.sin(angle)
        if point_to_segment_distance((CENTER, CENTER), (sx, sy), (ex, ey)) <= SUN_RADIUS:
            return None
        return best

    def build_destination_list(self) -> None:
        """For each fleet, find the first planet it is on an interception course for.
        Populates self.destination_list: Planet -> list of (owner, ships, t, src_x, src_y, arrival_x, arrival_y).
        t is continuous time in turns.
        """
        self.destination_list = defaultdict(list)
        for fleet in self.fleets:
            best = None
            best_t = float('inf')
            for planet in self.planets:
                needed_angle, px, py, travel = self.intercept_planet(
                    fleet.x, fleet.y, planet, fleet.ships
                )
                dist = distance((fleet.x, fleet.y), (px, py))
                if dist < planet.radius:
                    half_cone = math.pi
                else:
                    half_cone = math.asin(min(1.0, planet.radius / dist))
                delta = abs(math.atan2(math.sin(fleet.angle - needed_angle),
                                       math.cos(fleet.angle - needed_angle)))
                if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                    best_t = travel
                    best = (planet, travel, px, py)
            if best is not None:
                planet, travel, px, py = best
                self.destination_list[planet].append((fleet.owner, fleet.ships, travel, fleet.x, fleet.y, px, py))

    def simulate_planet_timeline(self, planet: HPlanet, destination_list: DestinationList) -> tuple[int, float]:
        """Simulate planet ownership/production over time given a list of inbound fleets.
        All arrivals at the same integer turn are resolved simultaneously (highest stack wins).
        Returns (final_owner, excess_ships) where excess_ships is the surplus in the last entry in destination_list.
        """
        cur_owner = planet.owner
        entries = destination_list.get(planet)
        if not bool(entries):
            return cur_owner, 0

        buckets = defaultdict(list)
        for owner, ships, t, _, _, _, _ in entries:
            turn = max(1, math.ceil(t))
            buckets[turn].append((owner, ships))

        last_ships, last_t = entries[-1][1], entries[-1][2]
        last_turn = max(1, math.ceil(last_t))

        cur_ships = float(planet.ships)
        prod = planet.production
        cur_t = 0
        # minimum margin by which the player survived each fight after the last entry landed
        excess_ships = float('inf')

        for turn in sorted(buckets):
            elapsed = turn - cur_t
            if elapsed > 0:
                if cur_owner == self.player:
                    cur_ships += prod * elapsed
                elif cur_owner != -1:
                    cur_ships += prod * elapsed
            cur_t = turn

            owner_ships = defaultdict(float)
            for owner, ships in buckets[turn]:
                owner_ships[owner] += ships

            if owner_ships:
                sorted_owners = sorted(owner_ships.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_owners) == 1:
                    survivor_owner, survivor_ships = sorted_owners[0]
                else:
                    top_owner, top_ships = sorted_owners[0]
                    second_ships = sorted_owners[1][1]
                    survivor_ships = top_ships - second_ships
                    survivor_owner = top_owner if survivor_ships > 0 else -1

                if survivor_ships > 0:
                    if survivor_owner == cur_owner:
                        cur_ships += survivor_ships
                    else:
                        cur_ships -= survivor_ships
                        if cur_ships < 0:
                            cur_owner = survivor_owner
                            cur_ships = abs(cur_ships)

            if turn >= last_turn:
                # track the narrowest margin by which we stayed in control
                margin = cur_ships if cur_owner == self.player else 0.0
                excess_ships = min(excess_ships, margin)

        if excess_ships == float('inf'):
            excess_ships = 0.0
        # excess can't exceed what the last entry actually sent
        excess_ships = min(excess_ships, last_ships)

        return cur_owner, excess_ships

    def evaluate_frontline_strategy(self, target: HPlanet) -> tuple[FleetOrders, list[Intercept], bool]:
        """Find the set of nearby ships needed to attack or reinforce a target.
        Returns (fleet_orders, intercepts, battle_won).
        intercepts is parallel to fleet_orders: list of (ix, iy, travel) pre-computed at plan time.
        """
        possible_origins = sorted(
            [(src, travel) for src, travel in self.inbound_edges.get(target, [])
                if src.owner == self.player], key=lambda x: x[1])

        fleet_orders: FleetOrders = []
        intercepts: list[Intercept] = []
        # Scale enemy ships at the target down to 50% — only half may actually threaten us.
        trial_destination_list = {}
        for _p, _entries in self.destination_list.items():
            if _p is target:
                trial_destination_list[_p] = [
                    (o, int(s * 0.5) if o != self.player else s, t, x, y, bx, by)
                    for o, s, t, x, y, bx, by in _entries
                ]
            else:
                trial_destination_list[_p] = list(_entries)
        trial_destination_list.setdefault(target, [])
        battle_won = False

        # If an enemy fleet is already inbound to this target (attacking its current owner,
        # who is not us), don't arrive until after that battle resolves.
        second_enemy_arrival = None
        if target.owner != self.player:
            for owner, _, t, _, _, _, _ in self.destination_list.get(target, []):
                if owner != self.player and owner != target.owner:
                    turn = math.ceil(t)
                    if second_enemy_arrival is None or turn < second_enemy_arrival:
                        second_enemy_arrival = turn

        for neighbor, _ in possible_origins:
            if neighbor.ships == 0 or neighbor.id in self._reserved_ids:
                continue  # reserved => committed to a pending hammer launch

            # Cost/benefit: is exposing neighbor to worst-case enemy pressure worthwhile?
            # Only relevant when neighbor survives in the baseline — if it's already doomed, send freely.
            # Assume all enemy planets connected via inbound_edges attack neighbor simultaneously.
            ships_to_send = int(neighbor.ships)
            baseline_owner, _ = self.simulate_planet_timeline(neighbor, self.destination_list)
            not_doomed = baseline_owner == self.player
            if not_doomed:
                worst_case_dl = {k: list(v) for k, v in self.destination_list.items()}
                worst_case_dl.setdefault(neighbor, [])
                half_pressure = 0
                # DEF_2P: in 1v1, model a larger share of adjacent enemy strength as
                # counterattack pressure (default 0.5). Higher => the source holds
                # back a bigger garrison => fewer planets bled to H1000's snipes.
                pres_frac = self.DEF_PRESSURE_FRAC if self.n_sides == 2 else 0.5
                for attacker, _ in self.inbound_edges.get(neighbor, []):
                    if attacker.owner == self.player or attacker.owner == -1 or attacker.ships == 0:
                        continue
                    _, ax, ay, atk_travel = self.intercept_planet(attacker.x, attacker.y, neighbor, attacker.ships)
                    if not math.isfinite(atk_travel):
                        continue
                    half_ships = max(1, int(attacker.ships * pres_frac))
                    worst_case_dl[neighbor].append((attacker.owner, half_ships, atk_travel, attacker.x, attacker.y, ax, ay))
                    half_pressure += half_ships

                saved_ships = neighbor.ships
                neighbor.ships = 0
                exposed_owner, _ = self.simulate_planet_timeline(neighbor, worst_case_dl)
                neighbor.ships = saved_ships

                if exposed_owner != self.player:
                    # Neighbor would fall under worst-case pressure; skip unless target production offsets the loss.
                    if target.production <= neighbor.production:
                        continue
                    # Knowingly sacrificing neighbor — send all ships.
                else:
                    # Neighbor holds worst-case; keep 50% of enemy pressure as a garrison buffer.
                    ships_to_send = max(0, int(neighbor.ships) - half_pressure)
                    if ships_to_send == 0:
                        continue

            # Cannot reach or is blocked.
            angle, ix, iy, travel = self.intercept_planet(neighbor.x, neighbor.y, target, ships_to_send)
            if not math.isfinite(travel):
                continue
            if self.first_planet_hit(neighbor.x, neighbor.y, angle, ships_to_send, neighbor) is not target:
                continue

            # second enemy arrival handling
            # + 1 for tolerance in swept collision handling
            if second_enemy_arrival is not None and math.ceil(travel) <= second_enemy_arrival + 1:
                continue

            trial_destination_list[target].append((self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy))
            fleet_orders.append([neighbor.id, angle, ships_to_send])
            intercepts.append((ix, iy, travel))
            trial_end_owner, excess_ships = self.simulate_planet_timeline(target, trial_destination_list)
            if trial_end_owner == self.player:
                battle_won = True

                # OVERSEND_2P: in 1v1, skip the trim so the capture lands at full
                # force — a thicker garrison survives the opponent's snipe-back
                # that otherwise causes v6's midgame "bleed".
                oversend = self.OVERSEND_2P and self.n_sides == 2
                if not_doomed and not oversend:
                    # Try leaving half the excess ships behind; re-simulate to confirm still winning.
                    # Never trim below 10 ships (small fleets move slowly and may miss the battle window).
                    keep = int(excess_ships // 2)
                    trimmed = max(10, ships_to_send - keep)
                    if trimmed < ships_to_send:
                        t_angle, t_ix, t_iy, t_travel = self.intercept_planet(neighbor.x, neighbor.y, target, trimmed)
                        if math.isfinite(t_travel):
                            trial_destination_list[target][-1] = (self.player, trimmed, t_travel, neighbor.x, neighbor.y, t_ix, t_iy)
                            if self.simulate_planet_timeline(target, trial_destination_list)[0] == self.player:
                                ships_to_send, angle, ix, iy, travel = trimmed, t_angle, t_ix, t_iy, t_travel
                            else:
                                # Trim would lose the battle; revert entry and keep original fleet.
                                trial_destination_list[target][-1] = (self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy)
                        # If t_travel is not finite, trimmed fleet can't reach target — keep original.
                    fleet_orders[-1] = [neighbor.id, angle, ships_to_send]
                    intercepts[-1] = (ix, iy, travel)
                break

        return fleet_orders, intercepts, battle_won

    def evaluate_move_orders(self) -> MoveOrders:
        """Score every reachable planet and pick the best destination."""
        best_move_orders: MoveOrders = (None, -65535, [], [])

        for target in sorted(self.planets, key=lambda p: p.ships, reverse=True):
            if not bool(self.inbound_edges.get(target)):
                continue # effectively unreachable

            # is owned
            if (target.owner == self.player):
                if not bool(self.destination_list.get(target)):
                    continue # no incoming

                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                threatened = (end_owner != self.player)
                if not threatened:
                    continue

                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)

                if not battle_won:
                    continue  # can't save it; skip for now

                value = target.production
                _, best_value, best_orders, _ = best_move_orders
                if (value > best_value or
                        (value == best_value and len(fleet_orders) < len(best_orders))):
                    best_move_orders = (target, value, fleet_orders, intercepts)

            # not owned
            else:
                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                if end_owner == self.player:
                    continue  # already won by in-flight fleets

                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)
                if not battle_won:
                    continue

                value = target.production
                if (target.owner == -1):
                    value = value - 1

                _, best_value, best_orders, _ = best_move_orders
                if (value > best_value or
                        (value == best_value and len(fleet_orders) < len(best_orders))):
                    best_move_orders = (target, value, fleet_orders, intercepts)

        return best_move_orders

    def send_reinforcements(self) -> FleetOrders:
        """ Allows sending by an intermediate planet if in the way. """
        orders: FleetOrders = []
        for p in self.owned_planets:
            if p.reinforcement_target is None or p.id in self._reserved_ids:
                continue
            if p.ships < (self.REINFORCEMENT_SIZE + self.GARRISON_SIZE):
                continue
            has_enemy_incoming = any(
                src.owner != self.player
                for src, _ in self.inbound_edges.get(p, []) )
            if has_enemy_incoming:
                continue
            target = p.reinforcement_target
            ships = int(p.ships - self.GARRISON_SIZE)
            angle, ix, iy, travel = self.intercept_planet(p.x, p.y, target, ships)
            if not math.isfinite(travel):
                continue
            orders.append([p.id, angle, ships])
        return orders

    def commit_move_orders(self, move: MoveOrders) -> None:
        target, _, fleet_orders, intercepts = move

        for (from_id, _, ships), (ix, iy, travel) in zip(fleet_orders, intercepts):
            src = next((p for p in self.planets if p.id == from_id), None)
            if src is None:
                continue
            src.ships = max(0, src.ships - ships)
            self.destination_list.setdefault(target, [])
            self.destination_list[target].append((self.player, ships, travel, src.x, src.y, ix, iy))

    # ------------------------------------------------------------------
    # Early game optimizer

    def early_game_compute_travel_turns(self, source_id: int, target: HPlanet, fleet_size: int, launch_turn: int) -> float:
        src = next(p for p in self.planets if p.id == source_id)
        orb = self.orbital_info.get(src)
        if orb is not None:
            cx = cy = CENTER
            r, ia = orb
            a = ia + self.angular_velocity * (launch_turn - 0.5)
            sx, sy = cx + r * math.cos(a), cy + r * math.sin(a)
        else:
            sx, sy = src.x, src.y
        _, _, _, travel = self.intercept_planet(sx, sy, target, fleet_size)
        return travel

    def early_game_find_capture_turn(self, state: EarlyGameState, target: HPlanet) -> float:
        """Return the earliest turn any single owned source can deliver > garrison ships."""
        garrison_size = target.ships
        horizon = state.turn + self.EARLY_LOOK_AHEAD
        best = math.inf
        for source in state.owned:
            current_ships = state.garrison[source]
            production_rate = state.production[source]
            for wait_turns in range(self.EARLY_LOOK_AHEAD):
                fleet_size = int(current_ships + production_rate * wait_turns)
                if fleet_size <= garrison_size:
                    continue
                launch_turn = state.turn + wait_turns
                if launch_turn >= horizon:
                    break
                travel_turns = self.early_game_compute_travel_turns(source, target, fleet_size, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= horizon:
                    best = min(best, arrival_turn)
                    break  # larger fleets from this source arrive no earlier
        return best

    def early_game_assign_fleets(self, state: EarlyGameState, target: HPlanet, capture_turn: int) -> dict:
        """Pick the single best source: earliest arrival with fleet > garrison."""
        garrison_size = target.ships
        best_source = None
        best_entry = None
        best_arrival = math.inf
        for source in state.owned:
            current_ships = state.garrison[source]
            production_rate = state.production[source]
            for wait_turns in range(capture_turn - state.turn):
                fleet_size = int(current_ships + production_rate * wait_turns)
                if fleet_size <= garrison_size:
                    continue
                launch_turn = state.turn + wait_turns
                travel_turns = self.early_game_compute_travel_turns(source, target, fleet_size, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= capture_turn and arrival_turn < best_arrival:
                    best_arrival = arrival_turn
                    best_source = source
                    best_entry = (fleet_size, launch_turn, arrival_turn)
                break  # larger fleets from this source arrive no earlier
        if best_source is None:
            return {}
        return {best_source: best_entry}

    def early_game_advance(self, state: EarlyGameState, from_turn: int, to_turn: int) -> EarlyGameState:
        for current_turn in range(from_turn + 1, to_turn + 1):
            for fleet in list(state.fleets):
                if fleet.arrival_turn == current_turn:
                    if fleet.is_capture:
                        state.garrison[fleet.destination_id] = fleet.garrison_on_arrival
                        state.owned.add(fleet.destination_id)
                        if fleet.destination_id not in state.production:
                            state.production[fleet.destination_id] = self.early_game_production(fleet.destination_id)
                    else:
                        state.garrison[fleet.destination_id] += fleet.garrison_on_arrival
                    state.fleets.remove(fleet)
            for planet_id in state.owned:
                state.garrison[planet_id] += state.production[planet_id]
        return state

    def early_game_execute_attack(self, state: EarlyGameState, target: HPlanet, fleet_assignment: dict, capture_turn: int) -> EarlyGameState:
        garrison_size = target.ships
        total_fleet = sum(fs for fs, _, _ in fleet_assignment.values())

        current_turn = state.turn
        for source, (fleet_size, launch_turn, _) in sorted(fleet_assignment.items(), key=lambda se: se[1][1]):
            state = self.early_game_advance(state, current_turn, launch_turn)
            current_turn = launch_turn
            state.garrison[source] -= fleet_size

        state.fleets.append(EarlyGameFleet(
            source_id=-1,
            destination_id=target.id,
            fleet_size=total_fleet,
            garrison_on_arrival=total_fleet - garrison_size,
            arrival_turn=capture_turn,
            is_capture=True,
        ))
        state = self.early_game_advance(state, current_turn, capture_turn)
        return state

    def early_game_score(self, state: EarlyGameState) -> int:
        horizon = state.turn + self.EARLY_LOOK_AHEAD
        total = 0
        for planet_id in state.owned:
            total += state.garrison[planet_id] + state.production[planet_id] * (horizon - state.turn)
        for fleet in state.fleets:
            total += fleet.garrison_on_arrival
            if fleet.is_capture:
                total += self.early_game_production(fleet.destination_id) * max(0, horizon - fleet.arrival_turn)
        return total

    def early_game_production(self, planet_id: int) -> int:
        p = next((pl for pl in self.planets if pl.id == planet_id), None)
        return p.production if p else 0

    def run_early_game(self) -> list:
        owned_ids = {p.id for p in self.owned_planets}
        neutral_candidates = [
            p for p in self.planets
            if p.owner == -1 and any(src.id in owned_ids for src, _ in self.inbound_edges.get(p, []))
        ]

        # Populate in-flight friendly fleets from destination_list so the optimizer
        # knows about already-committed ships and won't double-assign the same target.
        in_flight: list[EarlyGameFleet] = []
        for dest_planet, arrivals in self.destination_list.items():
            for owner, ships, t, _, _, _, _ in arrivals:
                if owner != self.player:
                    continue
                arrival = self.scene_step + math.ceil(t)
                is_cap = dest_planet.owner != self.player
                surplus = ships - dest_planet.ships
                in_flight.append(EarlyGameFleet(
                    source_id=-1,
                    destination_id=dest_planet.id,
                    fleet_size=int(ships),
                    garrison_on_arrival=int(surplus) if is_cap else int(ships),
                    arrival_turn=arrival,
                    is_capture=is_cap,
                ))

        initial_state = EarlyGameState(
            turn=self.scene_step,
            garrison={p.id: float(p.ships) for p in self.owned_planets},
            production={p.id: p.production for p in self.owned_planets},
            owned=owned_ids.copy(),
            fleets=in_flight,
        )

        def initial_gain(planet: HPlanet) -> float:
            ct = self.early_game_find_capture_turn(initial_state, planet)
            horizon = initial_state.turn + self.EARLY_LOOK_AHEAD
            return planet.production * (horizon - ct) - planet.ships if math.isfinite(ct) else -math.inf

        candidates = sorted(neutral_candidates, key=initial_gain, reverse=True)
        candidates = [p for p in candidates if initial_gain(p) > 0]
        # BUGFIX: original had `viz.add_text(...)` here referencing an
        # undefined module. Dropping the debug line keeps real early-game logic.

        if not candidates:
            return []

        best = [self.early_game_score(initial_state), []]

        def upper_bound(state, remaining):
            horizon = state.turn + self.EARLY_LOOK_AHEAD
            bound = self.early_game_score(state)
            for planet in remaining:
                ct = self.early_game_find_capture_turn(state, planet)
                gain = planet.production * (horizon - ct) - planet.ships
                if gain > 0:
                    bound += gain
            return bound

        def dfs(state, remaining, sequence):
            current_score = self.early_game_score(state)
            if current_score > best[0]:
                best[0] = current_score
                best[1] = list(sequence)

            if upper_bound(state, remaining) <= best[0]:
                return

            already_targeted = {f.destination_id for f in state.fleets if f.is_capture}
            for index, planet in enumerate(remaining):
                if planet.id in already_targeted:
                    continue
                horizon = state.turn + self.EARLY_LOOK_AHEAD
                ct = self.early_game_find_capture_turn(state, planet)
                if not math.isfinite(ct):
                    continue
                if planet.production * (horizon - ct) - planet.ships <= 0:
                    continue
                fleet_assignment = self.early_game_assign_fleets(state, planet, ct)
                if not fleet_assignment:
                    continue
                next_state = self.early_game_execute_attack(copy.deepcopy(state), planet, fleet_assignment, ct)
                dfs(next_state, remaining[:index] + remaining[index + 1:], sequence + [(planet, fleet_assignment, ct)])

        dfs(initial_state, candidates, [])
        _, best_sequence = best

        if not best_sequence:
            return []

        # Emit only the moves whose launch_turn == current step
        moves: FleetOrders = []
        for target_planet, fleet_assignment, _ in best_sequence:
            for source_id, (fleet_size, launch_turn, _) in fleet_assignment.items():
                if launch_turn != self.scene_step:
                    continue
                src = next((p for p in self.planets if p.id == source_id), None)
                if src is None:
                    continue
                angle, _, _, travel = self.intercept_planet(src.x, src.y, target_planet, fleet_size)
                if not math.isfinite(travel):
                    continue
                hit = self.first_planet_hit(src.x, src.y, angle, fleet_size, src)
                if hit is not target_planet:
                    continue
                moves.append([source_id, angle, fleet_size])

        return moves

    # ------------------------------------------------------------------
    # Forward-projection brain (v6)

    def forward_project(self, extra_arrivals=None, horizon=None,
                        phantom=True, emit_frac=None, snapshot_turns=None):
        """Project every planet's (owner, ships) forward `horizon` turns.

        Resolves in-flight fleet arrivals (from destination_list) plus any
        hypothetical `extra_arrivals` (our planned action) with the engine's
        simultaneous-combat math, accrues production, and — if `phantom` — lets
        each live planet periodically fling FWD_EMIT_FRAC of its surplus at its
        nearest non-friendly target (a cheap opponent model so we don't grab
        planets that get instantly sniped back).

        extra_arrivals: list of (target_pid, eta, owner, ships).
        Returns final {pid: (owner, ships)}, or (final, {t: snapshot}) when
        snapshot_turns is given.
        """
        horizon = self.FWD_HORIZON if horizon is None else horizon
        emit_frac = self.FWD_EMIT_FRAC if emit_frac is None else emit_frac
        # state[pid] = [owner, ships(float), production]
        state = {p.id: [int(p.owner), float(p.ships), int(p.production)] for p in self.planets}
        pos = {p.id: (p.x, p.y) for p in self.planets}

        arrivals = defaultdict(list)  # pid -> [(eta, owner, ships)]
        for planet, entries in self.destination_list.items():
            for entry in entries:
                owner, ships, travel = int(entry[0]), int(entry[1]), entry[2]
                if ships <= 0:
                    continue
                eta = max(1, int(math.ceil(travel)))
                if eta <= horizon:
                    arrivals[planet.id].append((eta, owner, ships))
        if extra_arrivals:
            for pid, eta, owner, ships in extra_arrivals:
                if ships > 0 and 1 <= eta <= horizon:
                    arrivals[pid].append((int(eta), int(owner), int(ships)))

        snap_set = set(snapshot_turns) if snapshot_turns else None
        snapshots = {}
        for t in range(1, horizon + 1):
            # production growth
            for st in state.values():
                if st[0] != -1:
                    st[1] += st[2]
            # phantom launches (opponents at full rate, us at half) every 4 turns
            if phantom and t % 4 == 0:
                for pid, st in state.items():
                    if st[0] == -1 or st[1] < 10:
                        continue
                    owner = st[0]
                    sx, sy = pos[pid]
                    best_d2 = float('inf')
                    best = None
                    for opid, ost in state.items():
                        if opid == pid or ost[0] == owner:
                            continue
                        ox, oy = pos[opid]
                        d2 = (sx - ox) ** 2 + (sy - oy) ** 2
                        if d2 < best_d2:
                            best_d2, best = d2, opid
                    if best is None:
                        continue
                    frac = emit_frac * (self.FWD_SELF_EMIT if owner == self.player else 1.0)
                    emit = int(st[1] * frac)
                    if emit < 5:
                        continue
                    speed = self.fleet_speed(max(2, emit))
                    eta = t + max(1, int(math.ceil(math.sqrt(best_d2) / speed)))
                    if eta <= horizon:
                        arrivals[best].append((eta, owner, emit))
                        st[1] -= emit
            # resolve arrivals landing this turn (simultaneous combat)
            for pid, arrs in arrivals.items():
                this_turn = None
                for eta, o, s in arrs:
                    if eta == t:
                        if this_turn is None:
                            this_turn = defaultdict(float)
                        this_turn[o] += s
                if not this_turn:
                    continue
                st = state[pid]
                d_owner, garrison = st[0], st[1]
                ranked = sorted(this_turn.items(), key=lambda kv: kv[1], reverse=True)
                top_o, top_s = ranked[0]
                if len(ranked) >= 2 and ranked[1][1] == top_s:
                    surv_s, surv_o = 0.0, -1
                elif len(ranked) >= 2:
                    surv_s, surv_o = top_s - ranked[1][1], top_o
                else:
                    surv_s, surv_o = top_s, top_o
                if surv_s > 0:
                    if d_owner == surv_o:
                        st[1] = garrison + surv_s
                    else:
                        ng = garrison - surv_s
                        if ng < 0:
                            st[0] = surv_o
                            st[1] = -ng
                        else:
                            st[1] = ng
            if snap_set is not None and t in snap_set:
                snapshots[t] = {pid: (st[0], st[1]) for pid, st in state.items()}

        final = {pid: (st[0], st[1]) for pid, st in state.items()}
        if snapshot_turns:
            return final, snapshots
        return final

    def forward_score(self, state):
        """Leader-relative value of a projected board from our POV:
        (our ships - leader ships) + 5*(planets lead) + 8*(production lead),
        where 'leader' is the single strongest OPPONENT (per metric). Aligns
        with Kaggle's single-highest-score-wins rule."""
        prod_by_pid = {p.id: int(p.production) for p in self.planets}
        ships = defaultdict(float)
        planets = defaultdict(int)
        prod = defaultdict(int)
        for pid, (o, s) in state.items():
            if o == -1:
                continue
            ships[o] += s
            planets[o] += 1
            prod[o] += prod_by_pid.get(pid, 0)
        me = self.player
        others = [o for o in ships if o != me]
        if not others:
            return ships[me]
        leader_ships = max(ships[o] for o in others)
        leader_planets = max(planets[o] for o in others)
        leader_prod = max(prod[o] for o in others)
        return ((ships[me] - leader_ships)
                + self.VAL_PLANET_W * (planets[me] - leader_planets)
                + self.VAL_PROD_W * (prod[me] - leader_prod))

    def _score_projection(self, extra_arrivals):
        """Average leader-relative score over the snapshot horizons (stabler
        than a single end-of-horizon read)."""
        final, snaps = self.forward_project(
            extra_arrivals=extra_arrivals, snapshot_turns=self.FWD_SNAPSHOT_TURNS)
        total = 0.0
        wsum = 0.0
        wt = self.SNAP_WEIGHT  # 0 => equal weight (1017); 1 => 1/t weighting (LB1050)
        for t in self.FWD_SNAPSHOT_TURNS:
            snap = snaps.get(t)
            if snap is not None:
                w = (1.0 / t) if wt else 1.0
                total += self.forward_score(snap) * w
                wsum += w
        if self.FWD_HORIZON not in self.FWD_SNAPSHOT_TURNS:
            w = (1.0 / self.FWD_HORIZON) if wt else 1.0
            total += self.forward_score(final) * w
            wsum += w
        return total / wsum if wsum else self.forward_score(final)

    def _action_for_target(self, target):
        """Concrete fleet orders to capture/defend `target`, or None. Mirrors
        v5's evaluate_move_orders viability gates, but returns the action so the
        search can SCORE it via projection instead of by raw production."""
        if not bool(self.inbound_edges.get(target)):
            return None
        if target.owner == self.player:
            if not bool(self.destination_list.get(target)):
                return None
            end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
            if end_owner == self.player:
                return None  # not threatened
        else:
            end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
            if end_owner == self.player:
                return None  # already being won by in-flight fleets
        fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)
        if not battle_won or not fleet_orders:
            return None
        return fleet_orders, intercepts

    def _depth2_penalty(self, target, our_extra):
        """LB1050 depth-2: worst-case opponent counterattack at `target` right after
        we capture it. For the nearest strong enemies, project our capture PLUS that
        enemy's counter-launch (real fleets only, no phantom); if the enemy retakes
        the planet, penalise by its strength. Returns a value <= 0 to add to the
        action's gain — demotes captures that immediately bleed back."""
        worst = 0.0
        evaluated = 0
        tx, ty = target.x, target.y
        enemies = sorted(
            (ep for ep in self.planets
             if ep.owner != self.player and ep.owner != -1 and int(ep.ships) >= 9
             and math.hypot(tx - ep.x, ty - ep.y) <= self.DEPTH2_RADIUS),
            key=lambda ep: math.hypot(tx - ep.x, ty - ep.y))
        for ep in enemies:
            opp_ships = max(8, int(ep.ships) - 5)
            d = math.hypot(tx - ep.x, ty - ep.y)
            opp_eta = max(1, int(math.ceil(d / self.fleet_speed(opp_ships))))
            if opp_eta > self.FWD_HORIZON:
                continue
            extra = list(our_extra) + [(target.id, opp_eta, ep.owner, opp_ships)]
            final = self.forward_project(extra_arrivals=extra, phantom=False)
            end_owner, end_ships = final.get(target.id, (-1, 0.0))
            if end_owner != self.player and opp_ships > end_ships:
                worst = min(worst, -float(opp_ships))
            evaluated += 1
            if evaluated >= self.DEPTH2_OPPS:
                break
        return worst

    def plan_midgame(self, deadline):
        """1-ply search: repeatedly commit the capture/defense with the best
        leader-relative projected score gain, until none helps or time is up.
        With DEPTH2 (2p), the top-K candidates are re-ranked by a counter-response
        penalty before committing."""
        moves: FleetOrders = []
        baseline = self._score_projection(None)
        use_d2 = bool(self.DEPTH2) and self.n_sides == 2
        max_actions = self.SEARCH_MAX_ACTIONS_2P if self.n_sides == 2 else self.SEARCH_MAX_ACTIONS
        for _ in range(max_actions):
            if time.perf_counter() >= deadline:
                break
            cands = []  # (gain, target, fleet_orders, intercepts, extra)
            for target in sorted(self.planets, key=lambda p: p.ships, reverse=True):
                if time.perf_counter() >= deadline:
                    break
                action = self._action_for_target(target)
                if action is None:
                    continue
                fleet_orders, intercepts = action
                extra = [
                    (target.id, max(1, int(math.ceil(travel))), self.player, int(ships))
                    for (_sid, _ang, ships), (_ix, _iy, travel) in zip(fleet_orders, intercepts)
                ]
                gain = self._score_projection(extra) - baseline
                # LB1050 2p arrival decay: prefer captures that land sooner (later
                # arrivals are noisier and give the opponent more time to respond).
                if self.ARR_DECAY < 1.0 and self.n_sides == 2 and gain > 0:
                    arrival = max((e[1] for e in extra), default=1)
                    gain *= self.ARR_DECAY ** arrival
                cands.append((gain, target, fleet_orders, intercepts, extra))
            if not cands:
                break
            # Stable sort by -gain preserves the original ship-desc tie-break, so the
            # DEPTH2-off path picks exactly the same action as the incremental argmax.
            cands.sort(key=lambda c: -c[0])
            if use_d2:
                topk = min(self.DEPTH2_TOPK, len(cands))
                for idx in range(topk):
                    if time.perf_counter() >= deadline:
                        break
                    g, tgt, fo, ic, ex = cands[idx]
                    cands[idx] = (g + self._depth2_penalty(tgt, ex), tgt, fo, ic, ex)
                cands.sort(key=lambda c: -c[0])
            best_gain, target, fleet_orders, intercepts, _ = cands[0]
            if best_gain <= self.SEARCH_MIN_GAIN:
                break
            self.commit_move_orders((target, 0, fleet_orders, intercepts))
            moves.extend(fleet_orders)
            baseline = self._score_projection(None)
        return moves

    def plan_pressure_2p(self, deadline):
        """2p anti-stall pass (runs AFTER plan_midgame).

        The value search only commits captures whose leader-relative score gain is
        positive under the *phantom* opponent model — so when that model imagines a
        capture being sniped back, v6 skips it and stalls. This pass presses the
        highest-production enemy planets we can win AND hold against *known*
        (real-fleet, non-phantom) threats, even when the phantom-discounted gain is
        ~0. Paired with OVERSEND_2P the capture lands thick enough to actually hold.
        Mirrors HEURISTIC1000's hammer drive to keep cracking strong-prod targets.
        """
        moves: FleetOrders = []
        if self.n_sides != 2:
            return moves
        committed = 0
        targets = sorted(
            (p for p in self.planets
             if p.owner != self.player and p.owner != -1
             and p.production >= self.PRESS_2P_MIN_PROD),
            key=lambda p: p.production, reverse=True)
        for target in targets:
            if committed >= self.PRESS_2P_MAX or time.perf_counter() >= deadline:
                break
            action = self._action_for_target(target)
            if action is None:
                continue
            fleet_orders, intercepts = action
            extra = [
                (target.id, max(1, int(math.ceil(travel))), self.player, int(ships))
                for (_sid, _ang, ships), (_ix, _iy, travel) in zip(fleet_orders, intercepts)
            ]
            # Hold check vs KNOWN threats only (phantom=False): commit if the
            # capture survives the real in-flight fleets through the horizon. This
            # is the deliberate difference from plan_midgame's phantom-pessimistic
            # gate — it acts where the value search froze.
            final = self.forward_project(extra_arrivals=extra, phantom=False)
            if final.get(target.id, (-1, 0))[0] != self.player:
                continue
            self.commit_move_orders((target, 0, fleet_orders, intercepts))
            moves.extend(fleet_orders)
            committed += 1
        return moves

    # ------------------------------------------------------------------
    # Persistent staggered hammer (v6 + cross-turn memory)

    def _predict_defender(self, target, arrival_rel):
        """Forecast (owner, ships) of `target` `arrival_rel` turns from now using
        ONLY known in-flight fleets (no phantoms) + production growth, resolving
        same-turn arrivals with the engine's simultaneous-combat math. Cheap,
        per-target version of simulate_planet_timeline that reports state at a
        specific future turn (what a hammer must out-muster)."""
        owner = int(target.owner)
        ships = float(target.ships)
        prod = int(target.production)
        buckets = defaultdict(lambda: defaultdict(float))  # turn -> {owner: ships}
        for entry in self.destination_list.get(target, []):
            o, s, travel = int(entry[0]), int(entry[1]), entry[2]
            if s <= 0:
                continue
            eta = max(1, int(math.ceil(travel)))
            if eta <= arrival_rel:
                buckets[eta][o] += s
        for t in range(1, arrival_rel + 1):
            if owner != -1:
                ships += prod
            if t not in buckets:
                continue
            ranked = sorted(buckets[t].items(), key=lambda kv: kv[1], reverse=True)
            top_o, top_s = ranked[0]
            second = ranked[1][1] if len(ranked) >= 2 else 0.0
            surv_s = top_s - second
            surv_o = top_o if surv_s > 0 else -1
            if surv_s > 0:
                if surv_o == owner:
                    ships += surv_s
                else:
                    ships -= surv_s
                    if ships < 0:
                        owner = surv_o
                        ships = -ships
        return owner, ships

    def _build_hammer(self):
        """Pick a high-production enemy target and a set of stockpiles whose
        combined fleet, staggered to land on ONE turn, beats the forecast
        defender * overkill. Returns a plan dict or None."""
        targets = [
            p for p in self.planets
            if p.owner != self.player and p.owner != -1
            and p.production >= self.HAMMER_MIN_PROD
            and self.inbound_edges.get(p)
        ]
        if not targets:
            return None
        stockpiles = [p for p in self.owned_planets if p.ships >= self.HAMMER_MIN_CONTRIB]
        if len(stockpiles) < 2:
            return None
        # Press only when not behind on production (don't over-extend while losing).
        my_prod = sum(p.production for p in self.owned_planets)
        prod_by_owner = defaultdict(int)
        for p in self.planets:
            if p.owner != -1 and p.owner != self.player:
                prod_by_owner[p.owner] += p.production
        leader_prod = max(prod_by_owner.values()) if prod_by_owner else 0
        if my_prod - leader_prod < self.HAMMER_PROD_LEAD:
            return None

        best = None
        for tgt in targets:
            per_src = []
            for src in stockpiles:
                ships = int(src.ships)
                angle, ix, iy, travel = self.intercept_planet(src.x, src.y, tgt, ships)
                if not math.isfinite(travel) or math.ceil(travel) > self.HAMMER_MAX_TRAVEL:
                    continue
                if self.first_planet_hit(src.x, src.y, angle, ships, src) is not tgt:
                    continue
                per_src.append((int(math.ceil(travel)), src, ships))
            if len(per_src) < 2:
                continue
            per_src.sort(key=lambda r: r[0])  # nearest first
            # Provisional arrival = farthest candidate (max lead time to stagger).
            provisional_arrival = per_src[-1][0]
            _, d_ships = self._predict_defender(tgt, provisional_arrival)
            required = int(math.ceil(d_ships * self.HAMMER_OVERKILL)) + 1
            accum = 0
            chosen = []
            for turns, src, ships in per_src:
                chosen.append((turns, src, ships))
                accum += ships
                if accum >= required or len(chosen) >= self.HAMMER_MAX_SOURCES:
                    break
            if len(chosen) < 2:
                continue
            # Tighten: real arrival is the farthest CHOSEN source; re-forecast there.
            arrival = chosen[-1][0]
            d_owner2, d_ships2 = self._predict_defender(tgt, arrival)
            if d_owner2 == self.player:
                continue
            required2 = int(math.ceil(d_ships2 * self.HAMMER_OVERKILL)) + 1
            if accum < required2:
                continue
            score = tgt.production * 10 - arrival  # prefer strong prod, near landing
            if best is None or score > best[0]:
                launches = {}
                for turns, src, ships in chosen:
                    launches[src.id] = {
                        "fire_abs": self.scene_step + (arrival - turns),
                        "ships": int(ships),
                        "fired": False,
                    }
                best = (score, {
                    "target_id": tgt.id,
                    "arrival_abs": self.scene_step + arrival,
                    "committed": int(accum),
                    "launches": launches,
                })
        return best[1] if best else None

    def plan_hammer(self):
        """Per-turn hammer driver. Validates/builds the persistent plan, fires any
        launches whose staggered fire-turn has arrived, and reserves the sources of
        still-pending launches. Returns the fleet orders launched THIS turn."""
        self._reserved_ids = set()
        moves: FleetOrders = []
        if not self.HAMMER_2P or self.n_sides != 2:
            return moves

        plan = _HAMMER_PLANS.get(self.player)
        if plan is not None:
            tgt = self.planet_by_id.get(plan["target_id"])
            arrival_rel = plan["arrival_abs"] - self.scene_step
            invalid = (tgt is None or tgt.owner == self.player or arrival_rel <= 0)
            if not invalid:
                _, d_ships = self._predict_defender(tgt, arrival_rel)
                if d_ships > plan["committed"] / self.HAMMER_ABORT_RATIO:
                    invalid = True  # defender over-reinforced past what we committed
            if not invalid:
                for sid, l in plan["launches"].items():
                    if l["fired"]:
                        continue
                    src = self.planet_by_id.get(sid)
                    if src is None or src.owner != self.player:
                        invalid = True  # lost a pending contributor
                        break
            if invalid:
                _HAMMER_PLANS.pop(self.player, None)
                plan = None

        if plan is None:
            plan = self._build_hammer()
            if plan is not None:
                _HAMMER_PLANS[self.player] = plan
        if plan is None:
            return moves

        tgt = self.planet_by_id.get(plan["target_id"])
        if tgt is None:
            _HAMMER_PLANS.pop(self.player, None)
            return moves

        pending = False
        for sid, l in plan["launches"].items():
            if l["fired"]:
                continue
            src = self.planet_by_id.get(sid)
            if src is None or src.owner != self.player:
                l["fired"] = True
                continue
            if l["fire_abs"] > self.scene_step:
                self._reserved_ids.add(sid)  # hold this stockpile for its launch turn
                pending = True
                continue
            ships = min(int(l["ships"]), int(src.ships))
            if ships < self.HAMMER_MIN_CONTRIB:
                l["fired"] = True
                continue
            angle, ix, iy, travel = self.intercept_planet(src.x, src.y, tgt, ships)
            if not math.isfinite(travel) or \
                    self.first_planet_hit(src.x, src.y, angle, ships, src) is not tgt:
                l["fired"] = True
                continue
            # commit: deduct ships + register the fleet so the brain treats it as in-flight
            src.ships = max(0, src.ships - ships)
            self.destination_list.setdefault(tgt, [])
            self.destination_list[tgt].append(
                (self.player, ships, travel, src.x, src.y, ix, iy))
            moves.append([sid, angle, ships])
            l["fired"] = True

        if not pending or all(l["fired"] for l in plan["launches"].values()):
            _HAMMER_PLANS.pop(self.player, None)
        return moves

    # ------------------------------------------------------------------

    def main(self, obs: dict[str, Any]) -> list[Any]:
        self._start_time = time.perf_counter()
        self.player = obs['player']
        self.scene_step = obs['step'] - 1
        self.angular_velocity = obs['angular_velocity']

        comet_ids = set(obs['comet_planet_ids'])
        planets_and_comets = [HPlanet(*p) for p in obs['planets']]
        self.planets = [p for p in planets_and_comets if p.id not in comet_ids]
        self.owned_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner != self.player]
        self.fleets = [Fleet(*f) for f in obs['fleets']]
        self.planet_by_id = {p.id: p for p in self.planets}
        self._reserved_ids = set()

        # Persistent hammer memory: drop a stale plan when a new game restarts the
        # step counter (or a seat is reused in local self-play). Only touched when
        # the hammer is enabled, so the default path stays identical to the 1017 agent.
        if self.HAMMER_2P:
            prev_step = _HAMMER_LAST_STEP.get(self.player, -1)
            if self.scene_step <= prev_step:
                _HAMMER_PLANS.pop(self.player, None)
            _HAMMER_LAST_STEP[self.player] = self.scene_step

        if not self.enemy_planets:
            return []

        # v5: count active sides (us + distinct enemy owners with a planet or
        # fleet). Use the tuned multiplayer reach in FFA, v2's reach in 1v1.
        # Must be set BEFORE build_proximity_graph (the sole consumer of MAX_DISTANCE).
        active_enemy_owners = {p.owner for p in self.enemy_planets if p.owner != -1}
        active_enemy_owners |= {
            f.owner for f in self.fleets
            if f.owner != self.player and f.owner != -1
        }
        self.n_sides = 1 + len(active_enemy_owners)
        self.MAX_DISTANCE = (
            Hellburner.MAX_DISTANCE_MP if self.n_sides > 2 else self.MAXDIST_2P
        )

        self.build_orbital_info(obs.get('initial_planets', []))
        self.build_proximity_graph()
        self.build_destination_list()

        if self.scene_step < self.EARLY_ROUNDS:
            moves = self.run_early_game()
            # BUGFIX: original had `elapsed_ms = (time.perf_counter() - _t0) * 1000`
            # referencing an undefined `_t0`. Drop the unused diagnostic.
            return moves

        self.build_reinforcement_targets()

        # Persistent staggered hammer FIRST (no-op unless V6_HAMMER set): it fires
        # due launches and reserves the stockpiles of pending ones, so the value
        # search below won't spend ships the hammer is holding for a future turn.
        hammer_moves = self.plan_hammer()

        # v6: replace v5's greedy "highest-production winnable target" loop with a
        # 1-ply search over the leader-relative forward-projection value function.
        deadline = self._start_time + self.SEARCH_SOFT_BUDGET
        moves = hammer_moves + self.plan_midgame(deadline)

        # 2p tactical pass (no-op unless V6_PRESS_2P set; 4p-identical to the 1017 agent)
        if self.PRESS_2P and self.n_sides == 2:
            moves.extend(self.plan_pressure_2p(deadline))

        reinforcement_orders = self.send_reinforcements()
        if reinforcement_orders:
            moves.extend(reinforcement_orders)

        return moves
        
def agent(obs: dict[str, Any]) -> list[Any]:
    _agent = Hellburner()
    try:
        return _agent.main(obs)
    except Exception:
        # Keep the defensive empty-fallback so a single buggy turn never
        # crashes our Kaggle submission, but unlike the original this should
        # now rarely (if ever) fire on the early-game path.
        return []