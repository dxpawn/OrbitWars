# Orbit Wars — Progress Diary

Reverse-chronological log of decisions, setup, training runs, and results. Newest on top.

---

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

