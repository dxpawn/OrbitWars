# Orbit Wars — Progress Diary

Reverse-chronological log of decisions, setup, training runs, and results. Newest on top.

---

## 2026-05-29 — CORRECTION: scores converged, hellburner WON. Do NOT pivot to ver16.

### The 2026-05-28 conclusion was wrong — it read TrueSkill scores mid-convergence
Kaggle Arena scores keep moving for many hours after submission as games accumulate.
Yesterday's entry below recommended "pivot to ver16, hellburner is capped at ~630."
That was based on transient scores (v2 at 600→761 while still climbing). **The scores
have now fully converged and tell the opposite story:**

| Submission | ref | Description | Converged score |
|---|---|---|---|
| **heuristic_v2** (bug-fixed hellburner) | 53118635 | our agent | **970.0** |
| hellburner-original (bugs intact, A/B) | 53118897 | diagnostic | **970.1** |
| teammate "hellburner + local tweaks" | 53125217 | NOT ours | 925.4 |
| ver16 | 53110595 | teammate's prior best | 816.0 |

**Conclusions (these supersede the 2026-05-28 "next-step recommendations"):**
1. **Hellburner is the strongest base, not ver16.** v2 = 970 vs ver16 = 816 (+154). Keep
   iterating on the hellburner/v2 base. Do NOT pivot to ver16.
2. **Comet-chasing is NOT the differentiator.** ver16 actively chases comets and scores
   816; v2 ignores comets entirely and scores 970. Adding comet logic (old task #11) is
   speculative and risks diverting ships from the core conquest game. Deprioritized.
3. **Local-overfitting actively HURTS on Kaggle — proven.** The teammate took hellburner,
   made "changes to beat it locally," and the result scored **925 < plain hellburner's 970.**
   Tuning a variant to beat the previous best in local h2h is the wrong gate. Validate
   against a *diverse pool* (win-share vs all 5 public adversaries + simple bots), not h2h
   vs the previous version.
4. **The early-game bug-fix ended up neutral at convergence** (v2 970.0 ≈ hellburner 970.1).
   The +161 mid-convergence gap was an artifact. The fix is harmless and correct, but it is
   NOT the reason v2 is strong — the hellburner architecture itself is.

### Environment facts (verified from kaggle_environments source this session)
- `agents: [2, 4]` — env supports both 2p and 4p. Competition is 4-player FFA, winner = the
  single player with the highest score (planet ships + fleet ships), tie ⇒ no winner.
- **`actTimeout = 1.0s` per turn**, `agentTimeout = 2s`, `episodeSteps = 500`,
  `shipSpeed = 6.0`, `cometSpeed = 4.0`. The 1s/turn cap is tight → a slow turn forfeits.
- Comets: spawn at steps [50,150,250,350,450], `COMET_PRODUCTION = 1` (low), ship count =
  `min` of 4 uniform(1,99) draws (biased low). They follow a fixed `paths` list visible in
  `obs['comets']` and **expire** when path_index runs off the end. Captured comets are a
  temporary, low-production asset — explains why ignoring them (v2) beats chasing them (ver16).

### Strategy going forward (this session)
The ONLY changes safe to ship without risking the live 970 are ones with zero strategic
downside: (a) fixing silent exceptions that waste turns, (b) eliminating >1s turns that
forfeit. Anything that changes targeting/strategy must clear a diverse-pool win-share gate
AND show no regression — and even then, treat local results with suspicion (see point 3).
Submission quota is SHARED with the teammate (they submitted 53125217) — do not burn slots.

### Robustness audit of v2 (eval/diag_v2.py) — CLEAN
Ran v2 instrumented (exceptions surfaced, every turn wall-timed) over real games:
- **2p** (12 games, 4262 turns): 0 exceptions, turn time mean 55ms / p99 178ms / max 341ms,
  zero turns >0.5s. 21% empty-move turns.
- **4p** (12 games, 3073 turns): 0 exceptions, turn time mean 22ms / max 216ms, zero >0.5s.
  **62.6% empty-move turns.**
Conclusion: no silent crashes, no timeout risk anywhere (cap is 1.0s; we peak at 0.34s). The
two "free win" categories (bugs, timeouts) are both already clean — there is no more free
upside on the v2 architecture. The 4p passivity (63%) is the interesting bit (see v4 below).

### v2 4p FFA baseline (eval/ffa4.py, new reproducible evaluator)
4-player free-for-all, v2 + 3 opponents sampled from the 5 strongest public agents, 200
games, hero seat rotated, lineups fixed by game index (so any candidate plays IDENTICAL
games): **v2 = 76/200 = 38.0%, CI95 [31.6%, 44.9%]**. Well above the 25% fair share — v2
wins ~1.5x its share against the toughest possible field. v2 is genuinely strong at 4p.

### v4 experiment — multiplayer threat re-calibration — REJECTED (worse)
Hypothesis: v2's evaluate_frontline_strategy assumes ALL connected enemies focus-fire a
neighbor simultaneously at 50% strength before it will commit that neighbor's ships. In 4p
that over-counts threat (the 3 opponents fight each other), causing the 63% passivity. v4
scales the per-attacker fraction by 2/n_sides (0.50 in 2p → 0.25 in full 4p). Provably a
**no-op in 2p**: verified byte-for-byte identical to v2 over 1881 2p turns, 0 mismatches.

Result on the identical 200 4p games: **v4 = 61/200 = 30.5%, CI95 [24.5%, 37.2%]** — i.e.
**7.5 points WORSE than v2.** Loosening the worst-case threat model makes v2 over-extend and
get punished. v2's passivity is disciplined, not broken: in a 4-way knife fight, conserving
ships and not overcommitting is what wins. **Rejected. submission.py stays = v2 (970).**

This was the third failed "improvement" at the time (teammate's 925<970, v3 neutral, v4
worse). It pointed at a local optimum — but the failures were all *strategy* tweaks. A
*parameter* sweep had never been run. That's what finally worked (v5 below).
`agents/heuristic_v4.py` kept in repo as a documented negative result (like v3); NOT shipped.

### v5 — mode-aware MAX_DISTANCE=30 — CONFIRMED IMPROVEMENT (shipped to submission.py)
The user asked to push past the local optimum. Rather than guess again, ran a disciplined
one-at-a-time parameter sweep (`eval/sweep.py`) on a tunable v2 clone (`agents/heuristic_tune.py`,
constants from HB_* env vars; byte-identical to v2 when unset), gated on 4p FFA win-share.

**Screen (seeds 0–199):** MAX_DISTANCE down helps monotonically (44→30.0%, 38→38.0%,
32→43.0%); EARLY_ROUNDS=5 looked +8; GARRISON/REINFORCEMENT neutral.
**Held-out confirmation (seeds 100000+, 250 games each):** MAX_DISTANCE 30 = **57.2% vs
baseline 44.4% (net +32 paired)**, peak of the 28–34 range (all +16..+32). Crucially,
**EARLY_ROUNDS=5 flipped to net −6 on held-out → seed-luck, REJECTED.** (This is exactly why
held-out confirmation matters; it caught a false positive.)

So the one real, robust lever is reach: 38 → ~30. Mechanism: a shorter MAX_DISTANCE makes
the proximity graph local, so the agent concentrates force near home instead of flinging
fleets across the board to be picked off, and counts fewer distant enemies as threats.

`agents/heuristic_v5.py` = v2 with **MAX_DISTANCE = 30 when n_sides > 2 (3p/4p), 38 in 1v1.**
Gated on active sides so it's **byte-identical to v2 in 2p** (verified: 2051 turns, 0
mismatches) — preserving the 2p strength that drives most of the Kaggle score — and reverts
to 38 automatically when a 4p game collapses to a 1v1 endgame. Cheaper than v2 (fewer edges)
so zero timeout risk.

**Validation (every test on seeds/pools the tuning never saw):**

| Test | seeds | opponent pool | v2 | v5 | paired net |
|---|---|---|---|---|---|
| Confirmation | offset 100k | 5 strongest | 44.4% | 57.2% | +32 |
| Final A/B | offset 200k | 5 strongest | 42.8% | 51.2% | +21 |
| Generalization | offset 300k | 8 mixed (incl. older) | 50.0% | **67.5%** | +35 |

Consistent +21..+35 net across three independent seed ranges AND a different opponent mix →
not seed-luck, not pool-overfit. **submission.py updated to heuristic_v5 (md5 5240492b...).**
make_submission.sh now emits v5.

### Session status / hand-off
- **submission.py = agents/heuristic_v5.py (md5 5240492b...).** v5 = v2 + 4p-only reach=30.
  v2 (the live 970 agent) is unchanged in the repo and v5 is identical to it in 2p.
- **Not yet uploaded to Kaggle** — quota is shared with the teammate; awaiting go-ahead.
- New tooling: `eval/diag_v2.py` (exception+timing audit), `eval/ffa4.py` (reproducible 4p
  FFA win-share, supports `--offset` for held-out), `eval/sweep.py` (OAT param sweep),
  `agents/heuristic_tune.py` (env-var-tunable v2 clone).
- Reproduce v5's edge: `python -m eval.ffa4 --hero heuristic_v2 --games 250 --offset 200000`
  then `--hero heuristic_v5 ...` (identical lineups).

---

## 2026-05-28 — Heuristic pivot: ship heuristic_v2 (bug-fixed hellburner)

### Context
After the RL/imitation track was abandoned (see previous session entry), the user delegated heuristics to teammates and asked us to push the project forward. Teammate's best entry was `other_adversaries/ver16-800score.py` — public Kaggle score 825.8. New public adversaries added to `other_adversaries/`: `hellburner.py`, `LB958.py`, `Proto-V15.py`, `inProgress.py`, plus `ver16-800score.py`.

### Pool calibration
First baseline: `heuristic_v1` (which previously went 50-0 vs the in-house pool of adv_distance / adv_lbmax / adv_structured / adv_rf_v0..v2) gets crushed by every new adversary, 40 games each:

| heuristic_v1 vs | win rate |
|---|---|
| adv_ver16 | 12.5% |
| adv_lb958 | 5.0% |
| adv_hellburner | 2.5% |
| adv_proto_v15 | 2.5% |
| adv_in_progress | 17.5% |

Lesson: the old in-house pool was a soft calibration target. `heuristic_v1` is overfit to it.

### Public-pool internal ordering (30 games each pair)
Compiled head-to-heads to find the strongest available base:

| Agent | overall win share | notes |
|---|---|---|
| **adv_hellburner** | **77.5%** | strongest |
| adv_proto_v15 | 63.3% | |
| adv_in_progress | ~45% | |
| adv_ver16 | ~47.5% | teammate's 825-pt agent |
| adv_lb958 | 24% | name is misleading — weakest of the 5 |

`hellburner` it is, as our base.

### hellburner bugs (silent)
Reading `hellburner.py` end-to-end, two latent crashes in the early-game (steps 0–2):
1. Line 687: `viz.add_text(self.scene_step, ...)` — `viz` is never imported. NameError.
2. Line 777: `elapsed_ms = (time.perf_counter() - _t0) * 1000` — `_t0` is never defined. NameError.

Both fire on every early-game turn. The agent wrapper has `try/except Exception: return []`, so the agent silently no-ops for its first 3 turns. Confirmed by tracing: the original hellburner never executes its DFS-based early-game capture optimizer in any game it plays.

### heuristic_v2 — bug-fixed hellburner
Created `agents/heuristic_v2.py` as a verbatim copy of `hellburner.py` with both bug lines removed (replaced by `# BUGFIX` comments documenting what was there). No other behaviour changes.

### Eval results (40 games per pair, both seatings, 2-player)

| heuristic_v2 vs | W-L-D | win rate |
|---|---|---|
| adv_hellburner | 19-15-6 | 47.5% (essentially tied — same agent, plus 3 extra early turns) |
| adv_proto_v15 | 31-9 | **77.5%** |
| adv_ver16 | 35-5 | **87.5%** |
| adv_lb958 | 34-6 | 85.0% |
| adv_in_progress | 37-3 | 92.5% |
| heuristic_v1 | 39-1 | 97.5% |

4-player mixed-pool spot check (16 games, 3 strong opponents): **37.5%** wins (random floor = 25%).

### Submission wiring
- `main.py` now prefers `agents.heuristic_v2` (`heuristic_v1` is the fallback if the env-package import inside v2 ever fails).
- `scripts/make_submission.sh` adds `agents/heuristic_v2.py` to the tarball.
- Built `submission.tar.gz` (24 KB, 9 files).

### Submission #1 — tarball ERRORED
First submission was the tarball (`submission.tar.gz`, 24 KB, 9 files: `main.py` + `agents/` + `rl/`). Kaggle returned `SubmissionStatus.ERROR` (no public score, no log surfaced via the CLI). Hypothesis: Kaggle's Arena harness expects a single `.py` file for this competition, not a multi-file tarball. Every successful submission in our submission history is a flat `submission.py` — including the strong public ones (ver16 was uploaded from the Kaggle notebook's `%%writefile submission.py` cell). So tarballs may simply not be the supported shape for this competition.

### Submission #2 — single-file
Resubmitted as `submission.py` (just `cp agents/heuristic_v2.py submission.py`). `heuristic_v2` is self-contained: stdlib + `kaggle_environments.envs.orbit_wars.orbit_wars` only, no cross-module imports inside the repo. Single-file is the canonical shape for this competition.

### Kaggle scores (the actual signal — scores update as games accumulate)

| Submission | Kaggle Score (latest sighting) | Notes |
|---|---|---|
| `ver16-800score.py` (teammate, Version 16) | 816-826 (drifting) | the bar to clear |
| `heuristic_v2` single-file (53118635) | **761.1** (rising: 600.0 → 628.4 → 701.4 → 761.1) | bug-fixed hellburner |
| `hellburner.py` original, bugs intact (53118897) | **600.0** (appears plateaued) | A/B baseline |

**Critical finding: local h2h is a poor predictor of Kaggle score.** Locally, `heuristic_v2` beats `adv_ver16` 87.5% (35-5). On Kaggle, ver16 currently outscores it by ~115 points (and the gap is closing as v2 plays more games). The Kaggle pool contains agents not in our local set.

The +101 points so far (701.4 - 600) from the bug-fix is a real win — the early-game DFS does help. But hellburner's broad architecture still looks capped below ver16. **Tentative read: hellburner is a dead end at the top of the leaderboard**, but the bug-fix itself is genuinely valuable (+100ish points).

### Pivot: ver16 is the right base
The teammate's progression 407 (Version 5) → 825 (Version 16) shows ver16-style code can scale. Hellburner-style probably can't (currently bracketed in the 600-700 range, may end higher but unlikely to surpass ver16). Next iteration should be ver16-based, not hellburner-based — **unless v2's score keeps climbing into the 800s**, in which case the comparison flips.

### v3 experiment
While waiting on Kaggle, built `agents/heuristic_v3.py` = v2 + 4p-aware target boost: in n_players >= 3, planets owned by the strongest enemy (production*8 + ships) get target value × 1.4. No change in 2p (no-op when no third owner).

Results (h2h vs v2 / key adversaries):
- v3 vs v2 in 2p: 8-6-26 (26 draws → 52.5% draw-adjusted, essentially identical, as expected since 2p code path is unchanged)
- v3 vs ver16 in 2p: 26-4 = 86.7% (matches v2's 87.5%)
- v3 vs hellburner in 2p: 15-11-4 = 56.7% (slightly above v2's 47.5%, within noise)
- v3 vs proto_v15 in 2p: 20-10 = 66.7% (slightly below v2's 77.5%, also within noise)
- 4p mixed pool (20 games each): v2 = 9/20 = 45.0%, v3 = 9/20 = 45.0% — **identical**

Conclusion: LEADER_TARGET_BOOST = 1.4 didn't move the 4p needle. Hellburner's "iterate target evaluation" loop already picks the highest-production reachable enemy planet, and that often *is* the leader's. Saved v3 in the repo for later iteration but **did not submit** — the v2 submission already in queue is the right entry.

### Submission slots used today (5 max per 24h)
1. tarball submission — ERROR (Kaggle Arena rejects multi-file tarballs for this competition)
2. heuristic_v2 single-file — **628.4**
3. hellburner-original (diagnostic A/B) — **600.0**

**2 slots remaining today.** Holding the rest unless we have a concrete theory of improvement (don't burn slots speculating; the +28 from bug-fixing the early-game already used the only "free" win available on the hellburner architecture).

### What changed in the repo for next iterators
- `submission.py` reset to byte-identical copy of `agents/heuristic_v2.py` (the 628.4 agent). Ready to upload if needed.
- `main.py` still prefers `agents.heuristic_v2`; harmless because checkpoint isn't present.
- `scripts/make_submission.sh` now emits BOTH a tarball and a single-file `submission.py`. Always upload the single-file.
- `agents/heuristic_v3.py` exists in the repo but is **not submitted**; v3 was a wash vs v2 (45/45 in 4p, mostly draws in 2p). Keep as reference, don't ship.
- 5 new public adversaries are in `opponents/REGISTRY` so the eval harness can use them: `adv_ver16`, `adv_hellburner`, `adv_proto_v15`, `adv_lb958`, `adv_in_progress`.

### Concrete next-step recommendations
1. **Pivot to ver16 as the base.** Hellburner is empirically capped at ~630 on Kaggle. ver16 hits 816-825. Trying to incrementally improve hellburner past ver16 is fighting the wrong battle.
2. **Stop trusting local h2h as a proxy for Kaggle score.** v2 wins 87.5% vs ver16 locally but loses ~200 points to it on Kaggle. The Kaggle pool contains agents we don't have. Use local h2h only to *gate regression* (no regression vs previous best), not to predict improvement.
3. **Coordinate with the teammate on ver16 iteration.** They've been climbing 407→825 over 16 versions. We shouldn't blow that runway by submitting our own speculation. Talk first.
4. **If we DO want to fork ver16:** the lowest-risk additions are probably
   - hellburner's `simulate_planet_timeline` to replace ver16's simpler `planet_under_threat` (more accurate defense decisions)
   - hellburner's "trim excess" logic (don't overcommit ships) — but only for the attack pass, leave ver16's defense untouched
   Both are surgical and reviewable.
5. **Comet handling (task #11)** — ver16 *already* handles comets well; hellburner doesn't. No work needed here from a ver16 base.

### Hand-off to teammates
- All five new adversaries are in `opponents/REGISTRY` under `adv_hellburner`, `adv_proto_v15`, `adv_ver16`, `adv_lb958`, `adv_in_progress`. Just `import opponents` and pull from the registry.
- Run `python -m eval.arena_cli h2h --a heuristic_v2 --b adv_hellburner --games 40` to repro the eval.
- For v3 ideas: the only place where v2 didn't dominate was vs hellburner itself (47.5%). That's expected (same agent), but if a teammate wants to push further, the gaps to look at are:
  - 4p mode (only 37.5% in spot check)
  - Comet capture (hellburner explicitly filters comets out of `self.planets`; ver16 actively chases them)
  - Multi-leg sun routing (hellburner's `first_planet_hit` simply bails when sun blocks; ver16 has `multi_leg_path`)
  - proto_v15's 22.5% wins probably correlate with specific board layouts — worth replaying losses to diagnose.

---

## 2026-05-28 — Session end: artifact inventory & next steps

### What we stopped (at user's request)
- All `rl.train`, `rl.collect_imitation`, `rl.imitation_train`, `rl.evaluate` processes on the pod killed via `pkill -9`. Pod confirmed clean (only `jupyter-lab` left, RunPod's built-in service — ignore).
- All my local background tasks (Bash monitors, Monitor watchers) ended or timed out.

### ⚠️ Pod billing still active — user action required
- **Killing processes does NOT stop pod charges.** The 3090 pod itself is still allocated at $0.46/hr.
- User must go to **RunPod UI → My Pods → click pod → "Stop"** (preserves volume + container at ~$0.01/hr) or **"Terminate"** (only network volume persists).
- Network volume `orbit-data` retains all training artifacts even after termination — can be remounted to a new pod later.

### What's on the pod (`/workspace/orbitwars/`) — preserved on network volume
- `checkpoints/`:
  - `imitation.pt` (1.1 MB) — supervised behavior-cloned policy. **Loses to random** (verified single-game eval). Probably not useful as-is but kept for reference.
  - `step_00000176.pt`, `step_00000336.pt`, `step_00000496.pt`, `step_00000656.pt`, `step_00000816.pt`, `latest.pt` — RL checkpoints at iters 11, 21, 31, 41, 51. None evaluated against pool; none expected to be competitive (training never converged).
- `state/imitation_data/` — 381 pickled game files (~50 MB total) from adversaries playing each other. Reusable if someone tries imitation again with better label inference.
- `state/league.json` — final league stats (wins/losses against each opponent) from the killed RL run.
- `logs/train.log`, `logs/collect.log`, `logs/imitation.log` — full stdout from each job.
- All source code, identical to local repo.

### What's in the local repo (`C:\Users\Admin\Downloads\OrbitWars\`)
- **All source code** committed locally — `agents/`, `rl/`, `opponents/`, `eval/`, `main.py`, etc.
- **`agents/heuristic_v1.py`** — the strong handcrafted heuristic. Beat the in-house pool 50-0 on Day 1. Still our submission floor.
- **`main.py`** — auto-selects: looks for `checkpoints/best.pt`, falls back to `agents.heuristic_v1.agent`. Since no `best.pt` exists locally, currently ships heuristic.
- **`scripts/make_submission.sh`** — builds `submission.tar.gz` ready for `kaggle competitions submit orbit-wars -f submission.tar.gz -m "..."`. Includes only the files needed for inference (no opponents/, no rl/train, no data).
- **`other_adversaries/`** — 6 public Kaggle submissions (3,000+ lines each in some cases). Useful as eval opponents if anyone resumes the RL track.
- **`ratings.json`** — local Glicko/win-rate data from Day 1 round-robin (sniper 72%, heuristic_v1 100% etc.).
- **`diary.md`** — this file.

### What you can do right now
1. **Stop the pod** in RunPod UI (saves ~$0.46/hr).
2. **Submit the heuristic to Kaggle** if you want a baseline leaderboard entry:
   ```powershell
   bash scripts\make_submission.sh
   kaggle competitions submit orbit-wars -f submission.tar.gz -m "heuristic_v1 baseline"
   ```
3. **Coordinate with teammates** — they're extending the heuristic; merge their version into `agents/heuristic_v1.py` (or new file) before next submission.
4. **Regenerate Kaggle API token** at https://www.kaggle.com/settings/api (since it was shared in this chat).

### Final cost accounting
- Total RunPod spend (this session): **~$1.40** of $20 budget.
- Remaining: **~$18.60** — usable as standby (~1850 hours of stopped-pod retention) or for a future training pod.

---

## 2026-05-28 — Day 1 (continued): RL + Imitation post-mortem

### Result: both approaches failed in the deadline window. Honest accounting below.

### RL from scratch (PPO + self-play + opponent league)

**Setup**: entity-transformer policy (d_model=96, 4 heads, 3 layers, ~1.2M params), pointer-style target selection, PPO with GAE (γ=0.997, λ=0.95), 12 multiprocessing rollout workers on RunPod 3090. Trained against "easy" league: random, do_nothing, sniper, rusher, defender, heuristic_v1, adv_rf_v1, adv_rf_v2. 16 episodes per iter, 15% 4p mix.

**Trajectory over 50 iters (~2.5 hrs)**:
- Average wins: 1–2 / 16 (~10-20% true win rate), draws variable
- T (active steps before elimination): bounced 4–40, no upward trend
- Entropy oscillated wildly: 0.08 → 27.7 across iters, no convergence
- KL bouncy (−0.005 to 0.28) but mostly tiny (<0.05) — policy barely updating most iters
- Big improvement signals appeared on iters with massive entropy spikes (18, 26, 29, 32, 39) — T temporarily jumped to 36-39, wins hit 3/16 once, but never sustained
- After iter 32, no further improvement

**What went wrong with RL**:
1. **Sparse terminal reward + 500-turn horizon** = too long for credit assignment. GAE with γ=0.997 helps but not enough for sparse +1/−1 signal.
2. **Random init dies in 2 turns** (mitigated by launch_head bias=-1.5 init, but learning still slow).
3. **Variance dominated**: 16 episodes/iter gave noisy gradient estimates; policy oscillated rather than converged.
4. **Entropy bonus (0.01) too small** to maintain exploration but too big to allow exploitation — policy got stuck in local optima where rare big-entropy spikes were the only progress mechanism.
5. **Adversaries too strong**: even the "easy" pool included adv_rf_v1/v2 (~600-700 line heuristics) that beat us reliably.
6. **Bottleneck never moved to GPU**: rollouts dominated wall time (~165s/iter) while GPU sat at 0%.

### Imitation pretraining (supervised behavior cloning)

**Setup**:
- Collected 381 game files of adversaries playing each other (6 targets: adv_rf_v1/v2/structured, heuristic_v1, sniper, rusher; 25 games per pair × 15 pairs).
  - First batch (15 heavy-adversary pairs at 20 games) abandoned after 8 minutes per game realized = 5+ hours total. Switched to lighter pool.
- Supervised cross-entropy on (launch / target / fraction) labels inferred from winner trajectories.
- 3 epochs × ~530 batches × 128 batch size on 3090.

**Result**: **policy net was worse than random**.
- Single eval game: imitation.pt vs random → **lost (reward = −1)** in full 500 steps.
- CPU inference: **3.5 minutes per game** (Kaggle's 1s/turn limit makes the transformer arch unshippable even if it learned).

**Why imitation failed**:
1. **Label noise**: my `_label_move` heuristic infers target slot from raycast direction. Adversaries use orbital intercept prediction — they aim at *future* positions, so the target with smallest angular delta to current positions is often wrong. Mass-mislabeled.
2. **Mixture of experts**: training on data from 6 different heuristics with different strategies → model averaged across them, learning none coherently.
3. **Loss never converged**: oscillated in 3.0–6.5 range across 41 minutes / 1611 steps. L_target best ~1.6 (vs random 4.56), worst >4.5 (worse than random in some batches).
4. **Imbalanced batches**: launch base rate is ~20% per planet per turn; many batches had zero launches → target/fraction losses computed on tiny samples.
5. **Skipped non-winner trajectories**: only ~50% of games had a clear winner; we dropped the loser data, halving effective dataset.

### Architectural verdict
The transformer policy was **doubly wrong**:
- Too slow for Kaggle inference (3.5 min/game on CPU — env runs 500 turns; we have 1s per turn).
- Too underdetermined by data we could collect in our budget.

A **much smaller, much faster** policy (small MLP, ~10K params, runs in <10ms per turn) might both train faster (less variance) and ship within Kaggle limits. But we didn't have time to redo the architecture.

### Total RL track cost
- Pod wall time: ~3 hours
- Pod spend: ~$1.40 of $20 budget
- **No deployable RL artifact produced.**

### Recommendation going forward
- **Heuristic track is the only realistic submission path.** `heuristic_v1` beat the in-house pool 50-0 on Day 1. Teammates are extending it.
- Stop the pod to save remaining ~$18 of credit.
- If we revisit RL post-deadline, the architectural lessons here matter:
  - Submission inference must run in <1s/turn → MLP or distilled compact transformer
  - Cleaner imitation targets (per-adversary specialization, accurate target inference)
  - Larger batch sizes per iter to tame variance
  - Curriculum that starts with much weaker opponents (random only) for many iters before introducing heuristics

---

## 2026-05-28 — Day 1 (continued): pivot to RL

### Pivot decision
- User delegated heuristic dev to teammates. Our track is now **RL only**.
- 6 public-leaderboard adversaries added to `other_adversaries/` (paths registered in `opponents.REGISTRY` as `adv_distance`, `adv_lbmax`, `adv_structured`, `adv_rf_v0`, `adv_rf_v1`, `adv_rf_v2`). They're 500–3500 lines each — much more sophisticated than our `nearest_sniper` baseline. These are real opponents.
- New pod address (third deploy): `213.192.2.68:40013`. SSH alias `runpod-orbit` updated.
- Kaggle API token written to `~/.kaggle/access_token` (user shared in plain chat — recommended regeneration after project ends).

### RL stack built
- `rl/features.py` — obs → entity-list tensor (MAX_ENTITIES=96, ENTITY_DIM=32, GLOBAL_DIM=12). Includes type onehots, owner onehots, position, ships (log-scaled), production, orbital flags, comet flag, distance to nearest owned planet, orbital phase sin/cos, fleet velocity. Carries slot→planet_id and slot→fleet_id maps for action decoding.
- `rl/policy.py` — entity transformer (d_model=96, 4 heads, 3 layers, GELU, prenorm). Three action heads per slot: launch_logit (Bernoulli), target_logits (bilinear pointer attention over all other entities, masked to planets+fleets only), fraction_logits (5 bins: 0.10/0.25/0.50/0.75/0.95). Value head from masked-mean-pooled entity embeddings.
- `rl/action_space.py` — samples per-owned-planet launch/target/fraction; computes angle deterministically from src→target with sun-avoidance tangent routing. Each launch produces one `[from_id, angle, ships]` move.
- `rl/reward.py` — terminal (engine-aligned: +1 win, +0.5 tied-for-first, −1 loss) + shaping (planet captures, ship advantage delta, production delta). Shaping coef annealed during training.
- `rl/rollout_worker.py` — one-episode worker. Builds `agent_fn` closure that logs (obs, action, value) inside `env.run`. Reusable across episodes via lazy `_POLICY` global.
- `rl/ppo.py` — PPO with GAE (γ=0.997, λ=0.95). Joint log-prob per step = sum across all owned-planet decisions. `_step_logp_entropy` handles per-step variable K (number of owned planets) with masked log-softmax. Linter pass cleaned up the entropy computation with `torch.where` to avoid `0 * -inf` NaNs at masked target slots.
- `rl/league.py` — opponent pool with inverse-win-rate sampling (hard opponents get more attention). Default league includes all 6 adversaries + in-house heuristics.
- `rl/train.py` — main loop with multiprocessing Pool (fork on Linux, spawn on Windows). 2p / 4p mix configurable. Saves checkpoints to `checkpoints/step_XXX.pt` + `latest.pt`.
- `agents/rl_inference.py` — load checkpoint, expose `agent(obs)` for submission.
- `main.py` — auto-selects RL (`checkpoints/best.pt`) or falls back to `heuristic_v1`.

### Local smoke test results
- 1 iter, 2 episodes, CPU: pipeline runs end-to-end. KL=1.09 high (expected on first update from random init). Entropy=9.93 (high — random policy). 20s/episode locally.
- Pod sync via `tar | ssh ... tar -xzf` (rsync not available on Windows bash). `--ignore-installed` needed to bypass blinker debian package conflict. `--break-system-packages` for PEP 668.
- Pod: torch 2.8.0+cu128 pre-installed (CUDA confirmed). kaggle_environments 1.30.1 installed.

### Smoke + first training launch
- First pod smoke test: 16 workers / 16 episodes / 1 iter. **Two issues:**
  1. T=2 average — agent dies in ~2 turns. Random policy was suicidally aggressive (launch_logit≈0.5, fraction bias uniform, often sent 95% of garrison each turn).
  2. `OSError: [Errno 24] Too many open files` in `multiprocessing.resource_sharer` mid-run. Default ulimit -n was too low for 16 workers.
- **Fixes applied:**
  - Bias initial policy: `launch_head.bias = -1.5` (sigmoid ≈ 0.18 launches/turn/planet), `fraction_head.bias = [1.0, 0.5, 0, -0.5, -1.0]` (favor 0.10 and 0.25 fractions early). Random init no longer suicides.
  - Reduced workers 16 → 12; `ulimit -n 65536` before launching.
  - Cleaner launch via `/tmp/launch_train.sh` with `nohup ... & disown` (raw nohup-via-ssh had exit code 255 issues).
- **First iter wall time: ~90s for 16 episodes, 12 workers** (~5.6s effective per game). PPO update <1s. GPU at 0% during rollouts (CPU-bound env), spikes during update.

### Training command
```
ulimit -n 65536; PYTHONUNBUFFERED=1 nohup python -u -m rl.train \
  --workers 12 --episodes-per-iter 16 --total-iters 500 --save-every 10 \
  --device cuda --league easy --mix-4p-prob 0.15 --shape-anneal-iters 200
```
- League "easy": skips the heaviest adversaries (adv_distance, adv_lbmax, adv_structured, adv_rf_v0). Includes adv_rf_v1, adv_rf_v2, heuristic_v1, and the in-house baselines.
- 16 episodes/iter × 500 iters = 8000 episodes total = ~12-13 hours.
- Checkpoints: every 10 iters → `/workspace/orbitwars/checkpoints/step_XXXXXXXX.pt` and `latest.pt`.
- Logs: `/workspace/orbitwars/logs/train.log` (PYTHONUNBUFFERED so we can `tail -f`).

## 2026-05-28 — Day 1: Project setup, cloud provisioned, eval harness scaffolding

### Cloud (RunPod)

- Provisioned RunPod **Secure Cloud** pod (first attempt: Community Cloud key injection failed; redeployed Secure).
- GPU: **RTX 3090 24 GB**, 32 vCPU (cgroup-limited; host shows 256 cores / 1 TiB), 125 GB RAM, $0.46/hr.
- Container disk 30 GB, network volume `orbit-data` 50 GB mounted at `/workspace` (persists across pod deletion).
- SSH: `runpod-orbit` alias in `~/.ssh/config` → `root@213.192.2.110:40185` via `~/.ssh/id_ed25519`.
- Verified: `nvidia-smi` shows the 3090; `python3 --version` = 3.12.3.

### Local environment

- Windows 10, Python 3.12.7, torch 2.9.1+cu126 already present, numpy/scipy installed.
- Installing `kaggle-environments>=1.28.0` and `kaggle` CLI now (background pip job).
- Project directory structure: `eval/`, `opponents/`, `agents/`, `rl/`, `replays/`, `checkpoints/`, `state/`, `logs/`.
- `requirements.txt` pinned with current deps.

### Strategic decisions (carried over from planning phase)

- **Heuristic-first, RL-on-top**: Day 1–2 builds eval harness + `heuristic_v1` (clean, ignores existing `START.ipynb` code which scores poorly). Heuristic ships as submission floor by end of Day 2.
- **RL architecture**: Entity transformer encoder + pointer-based action head. Angles computed deterministically from src→target (bakes geometric prior into the architecture). Small net (~1–2M params).
- **Training**: PPO + GAE, imitation warmup from heuristic, league self-play with Glicko matchmaking.
- **Submission decision rule**: best local Glicko wins; `main.py` auto-selects between heuristic and RL checkpoint.
- **Format**: 2p and 4p both targeted equally.

### Notes for future me / final report

- The existing notebook agent (`START.ipynb` cell 11) is **not used** — user reports it scores horribly in real competition. Kept only as a structural reference for env API usage.
- The original `main.py` (60-line "nearest sniper") is being relocated to `opponents/nearest_sniper.py` as the dumbest reference opponent in our pool. It is replaced at the repo root by the heuristic/RL submission entrypoint.
- Deadline: **2026-06-08**. 10 days from project start.

### Engine inspection (`site-packages/.../orbit_wars/orbit_wars.py`) — gotchas to remember

- **x/y are swapped on planet creation** (line 103: stored as `[id, -1, y, x, ...]`). The engine is internally consistent — `planet[2]` is treated as "x" throughout — but the human-readable variable names in `generate_planets` are misleading. Don't try to overlay positions on the README's diagram literally; trust the engine's `planet[2], planet[3]` ordering.
- **Continuous swept-pair collision** (`swept_pair_hit`, line 46): both fleet motion and planet rotation are linearized to chords over a tick. A fleet hits a planet iff the swept-pair distance falls below `planet.radius` for some `t ∈ [0, 1]`. The heuristic intercept solver must produce angles where this holds, not just "fleet endpoint lands inside the planet's new position."
- **Combat resolution** (lines 635–674): per-owner sum first; top vs second difference survives with top's owner; if top ties second, all attackers destroyed (`survivor_ships = 0`); then survivor either reinforces if owner matches planet or fights garrison (ownership flips if garrison goes negative, with surplus = `abs(negative)`).
- **Tie-for-first all get reward +1** (line 712): the condition `scores[i] == max_score and max_score > 0` rewards every player tied at the top. So we want to score strictly higher than #2 to be safe; ties are wins.
- **Comet schedule deterministic from seed**: per-spawn RNG is seeded with `f"orbit_wars-comet-{episode_seed}-{step+1}"`. Same seed → same comets. Replay reproducibility guaranteed.
- **Built-in agents in source**: `random_agent` (line 765) and `starter_agent` (line 778, static-only sniper with ≥20-ship threshold). Both accessible as `"random"` and `"starter"` in `env.run([...])`. Could use both as additional opponents.
- **OpenSpiel import noise**: kaggle_environments dumps several large lists to stdout when imported. Happens once per process. Tolerable; can suppress later via stdout redirect during import if it harms log readability.

