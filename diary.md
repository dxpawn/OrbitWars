# Orbit Wars — Progress Diary

Reverse-chronological log of decisions, setup, training runs, and results. Newest on top.

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

