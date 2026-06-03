# Orbit Wars — Progress Diary

Reverse-chronological log of decisions, setup, training runs, and results. Newest on top.

---

## 2026-06-02 (cont. 2) — DISTILLATION PIPELINE BUILT + TRAINED + INTEGRATED + SUBMITTED (53292001).

> **✅ RESULT: 53292001 jumped 878 → 885 → 1110.6 (still converging) — DISTILLATION TRANSFERRED.**
> New TEAM BEST (prev 1016.5), +113 over our v6 (997), only ~30 below the friend's 1140. Status COMPLETE
> (the base64 single-file bundle loaded + ran on Kaggle — packaging validated). Our own model, his
> transformer never shipped, 14× faster. The local-4p anti-prediction held one more time: ours scored
> 43.8% locally yet ~1110 on the ladder. Distillation alone ≈ his level; Stage-2 RL-finetune still needed
> to EXCEED 1140 (winner-take-all class). Monitoring for further climb.
>
> **SUBMITTED `submission_distilled.py` → ref 53292001.** Single-file base64 bundle (660 KB):
> his `orbit_base` hull + `main` feature extractor + OUR `student_weights_{2p,4p}` (renamed to the
> `feature46_weights_*` names his main imports). His 7.6 MB transformer NOT included. Built by
> `scripts/build_distilled_submission.py` (loads each source as a real sys.modules module → no
> agent/_nearest_targets name clash; decode utf-8-sig to strip his BOM). Selftest: won a 4p game
> [4301,0,0,0]. Quota 2/5 used (3 free). **This is the only valid test — see the timing finding below.**
>
> **KEY: local 4p eval is INVALID for these agents (the friend's LB-1140 agent scores 25% = RANDOM
> locally).** Per-turn timing probe (`turns>1s`): friend max=1.703s, **21/140 turns >1s** → his pure-
> Python transformer TIMES OUT locally (moves dropped → random). OURS: max=0.123s, mean=0.042s, **0
> timeouts, 0 exc** — ~14× faster (we replaced his transformer with our MLP). So ours is a faithful
> (76%/95% top1/3), MUCH faster, timing-safe copy of the 1140 policy. Local 4p (v6 45.8% / ours 43.8% /
> friend 25%) can't rank them — ladder is ground truth. Monitoring 53292001.

Executed the pivot. Full distill stack built and validated in one session. **Decision (user): reuse
his feature extractor as a library; the MODEL must be ours.** So no feature reimpl — we reuse his
`orbit_base` hull + `_candidate_features`, train OUR scorer to replace his transformer.

### Pipeline (all new files, all validated)
- **`rl/distill_collect.py`** — monkey-patches his `feature46_weights_{2p,4p}.score_many` to log
  every `(46-feature row -> his raw logit)` as his agent plays; parallel 2p+4p games vs a fast pool;
  per-worker `.npz` shards. **Dataset: `rl/distill_data/dataset.npz` = 227k (2p) + 284k (4p) = 511k rows.**
- **`rl/distill_train.py`** — trains a small MLP (46->64->64->1, ReLU, MSE on his logit) per mode on
  GPU; **exports pure-Python `rl/student/student_weights_{2p,4p}.py`** whose `score_many` is a math-only
  drop-in for his. Results: **2p R²=0.862, 4p R²=0.852, pearson ~0.92.**
- **`rl/validate_student.py`** — the RIGHT fidelity metric: per candidate-set, argmax(our)==argmax(his)?
  **2p top1=75.1% / top3=94.7% / Spearman=0.80; 4p top1=77.5% / top3=94.8% / Spearman=0.85.**
- **`opponents/distilled.py`** (registered **`ours_distilled`**) — loads his `main.py` but injects our
  student under the name `feature46_weights_{2p,4p}` (then restores sys.modules), so his hull + features
  run with OUR scorer. **His 7.6 MB transformer NEVER loads.** Verified: plays, beat v6 in a sample game.

### Key findings
- **The pointwise distillation ceiling ≈ 76% top1 / 95% top3 is the teacher's CROSS-CANDIDATE ATTENTION,
  not capacity.** A 256-wide student raised R² (0.88) but did NOT improve top1 (73.6%). His score for a
  target depends on the other candidates; our per-row MLP can't see that. **Locked the 64-wide student**
  (equal behavioral fidelity, faster inference). 95% top-3 + his SOFT re-rank (idx − bonus·sigmoid, bonus
  1.25-1.45) ⇒ actual chosen actions stay close to his. Good enough — beating 1140 is an RL job, not
  perfect mimicry.
- **Packaging is easy:** the friend submitted a multi-file FOLDER (he's on the LB at 1140), so Kaggle
  supports folder submissions here. Our submission = his `orbit_base.py` + `main.py` + OUR
  `student_weights` renamed to `feature46_weights_{2p,4p}.py`. (Confirm single-file-vs-folder mechanics
  before submitting — diary's old "tarball rejected" was early-v2; the friend proves multi-file works.)

### Operational pitfalls (recorded — cost real time)
- **His agent is computationally HEAVY locally** (~2.5× game time; pure-Python transformer scoring every
  candidate every turn). Benchmarks where he PLAYS are straggler-bound. **Data-gen avoids this only partly
  (he still plays to generate on-policy states); the deployed agent inherits his feature-extraction cost
  but swaps the transformer for a fast MLP → ours is ≤ his per-turn time → within Kaggle's 1s budget.**
- **Killing his game processes leaves ORPHANS** that survive `TaskStop` and keep burning cores (saw 64
  procs 100+ min old). Cleanup: `Get-CimInstance Win32_Process -Filter "Name='bash.exe'"` → kill the
  ones whose CommandLine matches `eval\.ffa4|distill_collect|...`, THEN `Stop-Process python`. **Prefer
  letting bounded runs COMPLETE over killing.**

### Status / next
- Phase 0-3 done (tasks #3-6). **Phase 4 (validate+submit) in progress (#7):** a clean 4p win-share
  (v6 vs ours_distilled vs friend) is running to quantify how much of his 1140 distillation captured.
  Then: package + submit `ours_distilled` (ladder = ground truth; local 4p is slow/unreliable here),
  then **Stage 2 — RL-finetune the re-ranker on outcomes (`rl/ppo.py`) to EXCEED 1140** (class is
  winner-take-all; distillation only ties). GPU available (torch 2.9 cu126). 36c/72t, parallel ~48-64.

---

## 2026-06-02 (cont.) — STRATEGIC PIVOT: distill a friend's LB-1140 imitation-transformer into OUR OWN learned target re-ranker

### Why we pivoted (local heuristic tuning is exhausted — proven on the ladder)
- **`VAL_PROD_W=16` (53264512) settled ~910 on the ladder — does NOT transfer.** It won +14 net broad
  in local 2p across 2 seedbases and was 4p-neutral, yet landed ~90 BELOW our v6 base (997.5). The
  Nth confirmation that **local eval (even paired 2p h2h) over-predicts; only the ladder is truth.**
- **exp30 (LB~1072) = HEURISTIC1000 + ONE line** (`SEARCH_MAX_ACTIONS_TO_PICK_2P` 7→9). Ported as our
  2p-gated `SEARCH_MAX_ACTIONS_2P`; held-out screen 9/10/12 all FLAT (+1/+1/+0) — **saturated, we
  already sit at 8.** The sibling-agent vein (H1000/LB1050/exp30, all share our brain) is fully mined.
- ⇒ Knob-hunting on the heuristic hull is over. The whole field is stuck ~1000-1072; the ONLY thing
  ~140 pts above is a different paradigm.

### THE CONTEXT (changes everything)
This is fundamentally a **university RL-class competition graded RELATIVE / winner-take-all** (top team
takes the points; others scaled down by rank), using the Kaggle LB as the scoreboard. A friend (NOT in
the class, impersonating a team to sabotage the curve) shared his **LB 1140.9** agent and gave **full
permission to do whatever** with it. We will NOT submit his file ("his shit") — we build OUR OWN, using
his agent as teacher/oracle/benchmark. Goal: **beat ~1140 to win the class.**

### WHAT HIS AGENT IS (the winning recipe, decoded)
Folder: `other_adversaries/submission_feature46_transformer_v2_late_recapture_2p_v1/`
(`main.py` + `orbit_base.py` heuristic hull + `feature46_weights_2p.py`/`_4p.py` = pure-Python `math`-only
transformer weights, ~3.8 MB each + `feature46_manifest.json`).
- **NOT a full policy.** It's a heuristic hull (hellburner-family: aiming, combat sim, fleet sizing,
  reinforcement) with a **learned TARGET RE-RANKER bolted on.** `main.py` monkey-patches
  `base._nearest_targets`: take the heuristic's candidate targets, expand (+10), compute **46 hand-
  engineered features per (src,target)** (`_candidate_features` / `FEATURE_NAMES`), score them through
  the transformer (`score_many`), re-sort by `adjusted = idx − bonus·sigmoid(score)` (bonus 1.45 2p /
  1.25 4p), execute on the re-ordered targets.
- **Trained by IMITATION**: he crawled top players' game steps and trained the transformer to predict
  which target a top player picks. Separate 2p/4p models. Pure-Python (no torch) → runs in Kaggle sandbox.
- **Why it's ~140 above the heuristic family:** it learned the single highest-leverage decision (target
  selection) from the people already winning, incl. **temporal/trend features** (momentum, enemy rhythm,
  approach-rate, convergence-threat over a rolling history) that our stateless v6 brain cannot see.

### THE PLAN — distill his oracle into our own re-ranker, then RL past him
Two-stage. We have his trained model as a perfect ORACLE (we own permission), so we **skip replay-crawling**.
- **Stage 1 (floor ≈1140): knowledge distillation.** Generate states cheaply (our hull / pool play),
  label each candidate with HIS `score_many`, train OUR student to match. Gets us ~his level with OUR weights.
- **Stage 2 (exceed >1140): RL-finetune** the re-ranker on actual game OUTCOMES (we have `rl/ppo.py`) —
  optimize for winning, not for imitating his picks. He's static; we improve → we pass him.

5 phases (tasks #3-7):
- **Ph0 De-risk/benchmark** (in progress): see results below.
- **Ph1** Reimplement the 46-feature encoder on OUR hull (heuristic_v6 data structures), faithful to his
  spec. Consistency-check: feed OUR features into HIS scorer → must reproduce his target picks.
- **Ph2** Generate distillation dataset (our features → his scores), 2p + 4p, 64-way parallel.
- **Ph3** Train + export pure-Python student (start MLP 46→hidden→1, ranking loss); validate rank-corr vs teacher.
- **Ph4-5** Bolt onto our hull, held-out eval vs pool INCLUDING the friend, submit; then Stage-2 RL.

### DESIGN DECISIONS (locked)
- **Our hull + our model + distillation labels = defensibly OURS** (it's an RL class; deliverable must be
  ours if audited). Knowledge distillation is a standard legit technique; resulting weights are ours.
- **Friend registered as `adv_friend_tf`** (lazy loader `opponents/friend_transformer.py`: loads his `main.py`
  under a unique module name with his folder on sys.path; ~7.6 MB import deferred to first call). Usable as
  benchmark opponent AND in the eval/data-gen pool.
- **Compute: 36 physical / 72 logical cores. Everything is CPU-bound & embarrassingly parallel** (game
  sim/rollouts/data-gen). Run pools at ~64. GPU only helps Ph3 training of a small net (minutes on CPU
  anyway); RL rollouts are CPU. Deployed re-ranker MUST stay pure-Python <1s/turn (his runs ~0.2-0.3s/turn,
  ~2.5× slower games — a model-size constraint).

### PHASE 0 RESULTS (de-risk)
- **ORACLE USABLE ✓** — `feature46_weights_2p/4p.score_many(rows)` returns finite scores standalone (46-dim
  input). Distillation is viable; the whole plan is de-risked.
- **v6 vs friend, 2p: 3–3 (of 6).** Our heuristic v6 is ALREADY competitive head-to-head in 2p! ⇒ his
  ~140-pt ladder edge is almost certainly in **4p FFA** (the dominant ladder format), NOT 2p. (A larger
  64-way parallel benchmark — 2p h2h + 4p win-shares for v6 vs friend vs strong pool — is running to confirm;
  it will FOCUS the re-ranker, possibly 4p-only.)
- We already have a full RL stack from earlier: `rl/policy.py` (entity-transformer), `rl/collect_imitation.py`,
  `rl/imitation_train.py`, `rl/ppo.py`, `rl/features.py`, `checkpoints/final.pt`. That effort stalled
  (full-policy RL is hard; we shipped heuristics). The re-ranker reframe is the tractable reuse.

### Fallbacks preserved
- `heuristic_v6_1017.py` (frozen 1017) + `submission.py` (now PROD_W=16, but that's ~910 — revert to 8 if we
  want the cleaner v6). Team best on the board remains the teammate's 1016.5 (53244319). v6 brain 997.5.

---

## 2026-06-02 — First transferable lever since the brain: VAL_PROD_W 8→16 (broad +14 net 2p, 4p-neutral). Submitted 53264512.

> **LADDER VERDICT (later 06-02): 53264512 settled ~910 — DID NOT TRANSFER.** Plateaued 558→923→912→910,
> ~90 below the v6 base (997.5). The broad replicated +14 local-2p gain did not move the real ladder —
> local eval over-predicted again. Lever shelved; see the pivot entry above. submission.py currently still
> carries PROD_W=16 (harmless, best-of protects rank); revert to 8 if we want the clean v6 back.


Task: "keep finding ways to improve the agent." Worked the **structural** knobs of the v6 brain
(the only category that has ever transferred to the ladder), under strict held-out discipline.

### Methodology hardened this session
- **Noise floor ≈ ±5 net @ 120 games/opp** (paired, `eval/confirm_ab.py`). Established empirically:
  self-emit screened +6 then **−4** on a second seedbase. ⇒ **nothing under ~+8 on a single screen
  is trustworthy; require replication on a 2nd seedbase + a monotonicity check** (a real effect
  trends with the parameter; noise is non-monotonic).
- **Signal hierarchy (from prior entries, re-applied):** 2p h2h vs a fixed opponent is the
  *ladder-predictive* signal; 4p FFA win-share has been *anti*-predictive (v5); the ladder is
  mixed FFA+2p. So the bar for a real candidate = **broad 2p gain (replicated) AND no 4p regression.**

### DEAD lever: phantom self-emit rate (`FWD_SELF_EMIT`, new env `V6_SELF_EMIT`, default 0.5)
The brain models itself launching phantom fleets at 0.5× the opponents' rate (asymmetric pessimism,
hypothesised cause of the midgame freeze). Exposed it (default 0.5 = byte-identical to 1017, identity
check net +0). Screen@200k: 1.0 = +6 (looked good, but **all from one opponent**). Confirm@350k:
**−4** — did not replicate; 0.7=+3 / 0.8=−1 non-monotonic. **Noise. Not shipped.**

### WIN: value weight `VAL_PROD_W` 8 → 16 (env `V6_PROD_W`)
Compute-neutral, diagnosis-motivated (value a production lead more → the existing search contests
high-prod planets natively, attacking the midgame bleed without a bolt-on pass). Held-out 2p
(`confirm_ab`, diverse pool: v2/hellburner/lb958/proto/H1000):

| PROD_W | screen@200k | confirm@350k | combined (240g/opp) | shape |
|---|---|---|---|---|
| 10 | +0 | — | — | flat (threshold) |
| **12** | **+5** | **+6** | **+11** | broad, replicated, 4p-neutral |
| **16** | **+9** | **+5** | **+14** | broad, replicated, **shipped** |

Trend is monotonic-increasing (12<16) = a real, mechanistic effect, not a lucky point. 4 of 5
opponents net-positive combined; only H1000 −1 (n.s.). **4p FFA gate (`eval/ffa4.py`, the scored
format), 500 games over 2 offsets:** baseline 50.2% vs PROD_W=16 49.4% = −4/500 games, **statistically
neutral (overlapping CIs, no detectable 4p effect).** ⇒ clears the bar: improves the predictive
signal, no v5-style 4p regression. **This is the first lever since the forward-sim brain to survive
the held-out test that killed the hammer, self-emit, and the LB1050 refinements.**

### Submitted: ref 53264512 (PROD_W=16) — 2026-06-01 18:29 UTC, PENDING (monitoring)
- **The ONLY diff from the 1017/997.5 v6 agent is `VAL_PROD_W: 8.0 → 16.0`** (one line, `submission.py:127`).
  Verified `forward_score` (the only fn PROD_W touches) is byte-identical between `submission.py` and
  the benchmarked `agents/heuristic_v6.py`, and `_score_projection` is behaviorally identical at default
  knobs — so the shipped file reproduces exactly what the A/B measured. Smoke-tested: crushed `random`
  9733-0, seat status DONE, **0 anomalies**; timing inherited from 1017 (compute unchanged by a constant).
- **Quota:** shared 5/UTC-day. 06-01 UTC had 2 teammate submissions; this is the 3rd → **2 slots left**
  for the team today. One slot spent, as authorized.
- **Ladder context (NEW this session):** the team's best is now **1023.5** (teammate's ref 53244971,
  06-01) and 1016.5 (53244319) — *above* our v6 brain, which **drifted 1017→1007→997.5** (53186031).
  So PROD_W=16 (v6 base + a few pts) likely lands **below 1023.5** → it will **not** take #1, but
  best-of-N means it **can't hurt rank**. Its value is the **signal**: does the +14 2p gain transfer
  to the ladder vs our v6 base (997.5)? If 53264512 converges clearly above 997.5, the value-weight
  lever transfers and is worth carrying onto the teammate's stronger 1023.5 line.

### Files / state
- `submission.py`: VAL_PROD_W 8→16 (shipped). **Uncommitted** (per standing rule, commit only when asked).
- `agents/heuristic_v6.py`: added `FWD_SELF_EMIT` knob (default 0.5 = off) — dev-only, default PROD_W
  kept at **8** so the registered eval baseline stays the stable 1017 brain (override with `V6_PROD_W`).
- Eval used: `eval/confirm_ab.py` (2p held-out paired) + `eval/ffa4.py` (4p scored format, env-injected).

---

## 2026-05-31 — v6 BRAIN WINS THE LADDER: 1017.2 vs v2's 911.7 (same-day, +105). Brain transfers. Starting 2p tactical port.

> ### Ladder state update (later 05-31, for the record)
> - **Our v6 brain (ref 53186031) drifted 1017.2 → 1007.0.** Expected cross-day TrueSkill drift as the
>   ladder population shifts (same pattern as v2: 970.0 → 910.4). Still our best, still the team's #1.
> - **A separate submission exists on the shared quota: ref 53209940 = 960.0** ("a minor update of 1025
>   with more brain of h1000", submitted 05-31 08:10, still converging). **NOT from this session** — I
>   ran zero `kaggle submit` (every experiment here was a held-out regression; explicitly chose not to
>   ship). This is a teammate's agent. Leaderboard is best-of, so 960 can't drag us below the 1007 brain;
>   if it converges higher it only helps. (Recorded so the lineage is unambiguous.)

The two 05-30 submissions converged. This is the cleanest, strongest result we've had:

| ref | agent | submitted | converged score |
|---|---|---|---|
| 53186031 | **heuristic_v6 (forward-sim brain)** | 05-30 13:41 | **1017.2** ← team best |
| 53185991 | heuristic_v2 (re-submit of proven 970) | 05-30 13:39 | **911.7** |

**v6 beats v2 by +105.5 in a controlled same-day paired test** — both submitted within 2 min,
both converged ~24h against the *same* opponent population. This is the gold-standard comparison
we never had before (no cross-day scale drift to confound it).

**Two big lessons, both confirmed:**
1. **The brain transfers.** The open question from 05-30 ("does the v6 forward-sim brain beat
   v2's 970?") is answered emphatically: yes, +105 over a *concurrently measured* baseline. The
   forward-projection + leader-relative value + 1-ply search core is a real upgrade, not a
   local-metric mirage. v6 is now our agent.
2. **Cross-day absolute scores are untrustworthy — proven by a control.** The *identical* v2
   agent scored 970.0 on 05-28 but 911.7 on 05-30 (−58 for the same code). The ladder scale
   drifted (competitors strengthened). ⇒ Only same-day pairings are reliable; this is why we
   re-submitted v2 alongside v6 instead of comparing v6 to the old 970. Worth the slot.

Note: local 2p h2h (v6 beats v2-2p ~60%) WAS directionally predictive here, even though local 4p
FFA was anti-predictive for v5's reach. The distinction: 2p h2h vs a fixed opponent is a cleaner
signal than 4p FFA win-share. Still trust the ladder over local for any close call.

### Tactical port attempt — cheap knob levers are DEAD (attack AND defense). Key reframe below.
Added 4 env-gated, default-off 2p levers to `heuristic_v6.py` (shipped agent byte-identical with
all off) and swept them paired vs H1000 (`eval/sweep_v6_2p.py`):
  - `OVERSEND_2P` (skip capture-fleet trim → land full force): net +0 @N60 (neutral).
  - `PRESS_2P` (post-search pass to press hold-able high-prod captures): net +0 (INERT — by the
    time it runs, plan_midgame has already spent the available ships; nothing left to press).
  - `DEF_PRESSURE_FRAC` 0.75/1.0/1.5 (source holds back bigger garrison vs counterattack): net −3
    each (more defense slightly HURTS).
⇒ Three independent lever families (attack-oversend, attack-press, defense-reserve), all
neutral-to-negative. This RE-CONFIRMS the 05-30 finding with new evidence: **the 2p gap to H1000
is structural, not closable by tuning v6's knobs.** baseline v6 vs H1000 in 2p ≈ 13%.

### THE REFRAME (important): v6 ALREADY MATCHES H1000 ON THE LADDER (1017 vs 1000-1100) — despite
losing their direct 2p h2h ~87-13. The ladder is mostly multi-opponent FFA + mixed 2p, not the
H1000 mirror. So "close the 2p gap to H1000" is very likely the WRONG objective: v6 is broadly
strong (beats v5 ~60% 2p, beat v2 by +105 on the ladder) and only loses to this ONE heavily-tuned
opponent head-to-head. Optimizing the H1000 matchup ≠ optimizing ladder score — and per our
hardest-won lesson, local single-opponent metrics mislead. The only knob that ever transferred
(the brain) was a *structural* change, not a tuned constant.

The one remaining tactical lever with real upside is H1000's PERSISTENT multi-turn staggered
hammer (launch from staggered sources so a big combined fleet lands on ONE turn, beating a
reinforcing defender). v6 can't do this today: `agent()` re-instantiates `Hellburner()` every turn,
so v6 has ZERO cross-turn memory. Porting it = giving v6 module-level persistent plan state — a
substantial, higher-risk build whose ladder payoff local eval can't reliably predict. Decision on
whether to invest in it (vs holding our strong 1017) surfaced to the user.

### Built the persistent staggered hammer (user chose the big swing) — then KILLED it. False positive.
Implemented it fully (env-gated `V6_HAMMER`, default OFF, 2p-only so 4p == 1017 byte-for-byte):
module-global plan state keyed by player + reset on game restart (`_HAMMER_PLANS`/`_HAMMER_LAST_STEP`),
`_predict_defender` (per-target forecast at arrival), `_build_hammer` (pick high-prod target + stagger
source launch turns to land on one turn, sized to defender×overkill), `plan_hammer` (validate/build/fire
+ reserve pending stockpiles), and a `_reserved_ids` skip in evaluate_frontline_strategy/send_reinforcements.
Timing-safe (0 exc, max 202ms/turn over 1135 turns); genuinely active (11 plans/27 launches over 6 games).

The trap: paired vs our OWN frozen 1017 (`heuristic_v6_1017`), the default hammer looked GREAT —
net +12 (+18/-6), 25% vs 10%. **But that was tie-breaking in a near-mirror match, not strength.**
The held-out cross-opponent confirm (`eval/confirm_hammer.py`, seedbase 200k, hammer-on vs -off paired
per opponent) demolished it — the hammer is WORSE vs EVERY opponent:
  - heuristic_v2 **net -33** (60.8% → 33.3%)   - adv_hellburner **net -25** (69.2% → 48.3%)
  - adv_proto_v15 **net -13** (82.5% → 71.7%)  - adv_heuristic1000 -7   - adv_lb958 -1
Cause: the telegraphed multi-turn buildup + ships locked idle in reserve make v6 far less responsive;
any competent opponent punishes the staged commitment. **NOT SUBMITTED.** Textbook re-proof of THE
lesson: a single-opponent (mirror) local metric misled us; the held-out cross-opponent eval is the
reliable one. Hammer code stays in-tree but default-OFF (shipped agent byte-identical to 1017).

### VERDICT: the entire tactical-port direction is exhausted and negative.
Three rounds, all dead: (1) constant micro-tuning [prior sessions], (2) cheap 2p knobs
(oversend/press/def_frac, all ~0/neg), (3) the structural persistent hammer (broadly -13..-33).
v6's forward-sim BRAIN (the structural change) was the only thing that ever transferred (+105 ladder).
**v6=1017 (hammer off) stands as our best.** Further local tactical work is low-EV; trust the ladder.

### Studied a NEW public agent (LB1050) for a structural idea — found none that transfers.
User dropped in `other_adversaries/Heuristic Simulation Agent Test 3 LB1050.py` (1050 LB, 3799 lines).
It is a SIBLING of HEURISTIC1000 (same auto-tuned hellburner family) and — key finding — its decision
core (`search_step_action` + `melis_evaluate` + `forward_score`) is the SAME brain + the SAME
leader-relative value function as v6 (ships−leader + 5·planets + 8·prod, byte-for-byte). So our brain
reimplementation was on target. Its "COUNCIL" header lists 3 portable structural deltas; ported all,
env-gated default-OFF (shipped == 1017), tested held-out cross-opponent (`eval/confirm_ab.py`, 120
games/opp, seedbase 200k, the discipline that caught the hammer):
  - **SNAP_WEIGHT** (1/t snapshot weighting, vs our equal weight): net ~0 (heuristic_v2 -1, proto +1).
  - **ARR_DECAY=0.97** (2p: discount gain by decay^arrival): folded into the above ~0 combo.
  - **DEPTH2** (counter-response penalty: re-rank top-K captures by whether a nearby strong enemy
    retakes the target): **SUM net -10** across 5 opps (v2 -3, hellburner -7, proto -5; lb958 +2,
    H1000 +3). Broadly slightly negative.
None transfers. Makes sense: same brain + same value fn ⇒ these are marginal selection tweaks that
don't net positive in our codebase. (Harness note: confirm_ab first gave a fake +0/-0 — treatment
env was in a module global that spawn-workers don't inherit; fixed by passing it through the job
tuple. The `base_wr` in the depth-2 run exactly matched prior baselines, proving the plan_midgame
restructure preserved the 1017 off-path.) **Nothing submitted. v6=1017 remains our best.**

---

## 2026-05-30 — v5 REGRESSED on the ladder (918<970); ported a forward-sim "brain" (v6); submitted v2+v6.

### CRITICAL: v5 (yesterday's "best") scored 918, BELOW v2's 970. Local 4p FFA is anti-predictive.
The converged public scores are now in, and they overturn yesterday's conclusion:

| ref | agent | converged score |
|---|---|---|
| 53118635 | **heuristic_v2** (reach 38 everywhere) | **970.0** |
| 53118897 | hellburner (orig, bugs) | 966.3 |
| 53125217 | hellburner + local tweaks | 945.3 |
| 53154166 | **heuristic_v5** (reach 30 in 4p) | **~915** (918.5 → 914.7, drifting) |
| 53110595 | ver16 | 816.0 |

v5's ONLY difference from v2 is `MAX_DISTANCE` 38→30 in 4p. That change won local 4p FFA
decisively (+21..+35 net across 3 held-out seed ranges + a mixed pool) but **LOST 52 points
on the real ladder.** This is the SECOND time a local-4p-FFA-validated change regressed the
ladder (first: teammate's "hellburner + local tweaks" 945 < 970). **Conclusion: our local 4p
FFA win-share metric is not merely unreliable — for reach tuning it was anti-correlated with
ladder score. Treat ALL local-eval-only conclusions as unproven until a ladder submission says
otherwise.** v2 (970) is our true best. (HANDOFF/diary entries below that call v5 "best" are wrong.)

### A public agent (other_adversaries/HEURISTIC1000.py) scores 1000-1100 — studied it.
4868-line, auto-tuned agent (constants like SO1_STATIC_BONUS_4P=2.95474 ⇒ CMA-ES). It is a
different CLASS of agent. Its decision core ("brain"):
  - **forward_project**: projects EVERY planet's (owner,ships) forward ~12-20 turns at once,
    including "phantom" opponent launches (each live planet flings a fraction of surplus at its
    nearest non-friendly target), resolved with engine simultaneous-combat math.
  - **forward_score**: LEADER-RELATIVE value — (my_ships − leader_ships) + 5·(planets lead) +
    8·(prod lead), leader = strongest opponent. Matches Kaggle's single-highest-wins rule.
  - **search_step_action**: 1-ply search — score each candidate capture by its projected score
    delta vs doing nothing, pick best. Plus a tactical pipeline (defense reserve/coalitions,
    cheap-pickup, expand, accumulator, mega-hammer/hammer, multiprong), all time-budgeted.
Verified in our harness: **2p h2h it beats v5 94%**; 4p FFA ~tied vs our strong pool; timing-safe
(2p max 35ms, 4p max 170ms, 0 over 1.0s, 0 exc); self-contained single file. The gap is a
*brain* gap (lookahead+leader-relative value), not a tuning gap — which is exactly why our
constant micro-tuning kept producing nulls.

### Built heuristic_v6 = v2's machinery + our own reimplementation of that brain.
`agents/heuristic_v6.py`: keeps ALL of v5/v2's geometry, combat sim (simulate_planet_timeline),
candidate gen (evaluate_frontline_strategy), early-game DFS, reinforcement. Replaces ONLY the
greedy `value=production` mid-game loop with:
  - `forward_project()` (global board projection w/ phantom launches),
  - `forward_score()` (leader-relative),
  - `plan_midgame()` (1-ply search committing best-projected-gain actions until none help or the
    SEARCH_SOFT_BUDGET=0.85s deadline). ~330 lines added; reuses everything else.
Brain knobs are env-tunable (V6_*) like heuristic_tune. Timing-safe (2p max 218ms, 4p max 143ms,
0 over 1.0s, 0 exc).

Validation (local — now known to be an unreliable predictor, but reported):
  - 2p h2h: **v6 beats v5 ~60%** (58.5% @off300k, 60.9% @off900k) — robust, two seed ranges.
  - 4p FFA: v6 vs v5 offset-dependent (+13 @off300k, −2 @off900k) ⇒ ~tied/noisy.
  - vs strong pool v6 ≥ v5; vs weak mixed pool v6 ≈ v5 (both ~70%).
  - v6 vs HEURISTIC1000 in 2p: only 13% (H1000 still dominates).

### Tuning: EMIT_FRAC 0.20→0.10 is the one real brain knob (then a dead end on 2p).
Sweep of brain constants (eval/sweep_v6.py), held-out confirmed:
  - **FWD_EMIT_FRAC 0.20→0.10**: +7/+11/+8 net paired across 3 seed ranges (0,500k,700k); 0.10 is
    the peak (0.05/0.08/0.12 lower). Baked into v6. (Less opponent-pessimism ⇒ v6 stops skipping
    good captures it wrongly feared would be sniped.)
  - HORIZON, PLANET_W, PROD_W: dead/noisy. Not changed.
2p gap to H1000 diagnosed (eval/diag_2p.py): v6 is EVEN through step ~50, then STALLS and bleeds
planets in the midgame (steps 75-150) while H1000 keeps expanding + cracking v6's planets — the
mirror image of how v6 beats v5. A 2p-aggression sweep (eval/sweep_v6_2p.py vs H1000) found NO
knob helps; longer 2p reach is sharply HARMFUL (−11/−15, reconfirming local concentration even in
2p). The 2p gap is structural (H1000's whole tuned tactical layer), not tunable.

### Submissions (today, 2 of 5 daily slots; quota confirmed free):
  - **53185991 — heuristic_v2** (reach 38): re-establish the proven 970 as the ACTIVE agent
    (since v5/915 was the latest = our worse agent was live).
  - **53186031 — heuristic_v6**: the brain experiment. REBASED to reach-38 (MAX_DISTANCE_MP 30→38)
    so the ONLY difference from v2(970) is the forward-sim brain — cleanest possible test. PENDING;
    needs hours to converge. The whole point is ground truth: local eval can't tell us if the brain
    beats 970.

### New tooling
  - eval/gap_h1000.py (v6/v5/H1000 FFA + 2p h2h), eval/h2h.py (general 2p h2h),
    eval/time_h1000.py + inline timing (per-turn ms audits), eval/sweep_v6.py (brain sweep),
    eval/sweep_v6_2p.py (2p-aggression sweep vs an opponent), eval/diag_2p.py (2p trajectory diag),
    eval/v6_eval.py (decision-grade v6 vs v5 vs H1000).
  - opponents/__init__.py: added adv_heuristic1000 (HEURISTIC1000.py) and heuristic_v6.
  - submission.py currently = heuristic_v6.py (reach-38 brain). make_submission.sh now defaults
    to v6 with `SUBMIT_AGENT=agents/heuristic_v2.py` override for the proven 970 baseline.

### Repo cleanup (end of session)
Removed accumulated scratch so the tree is clean: 22 transient sweep/eval logs (eval/*.out,
eval/*.err — key numbers already transcribed above), the dead submission.tar.gz (Kaggle rejects
tarballs), and all __pycache__/*.pyc (the prior session had committed these into git). Consolidated
.gitignore to keep them out (__pycache__/, *.pyc, *.log, eval/*.out, eval/*.err, submission.tar.gz).
KEPT: all eval/*.py tooling (reusable, referenced above for reproducibility) and the documented
agent lineage (v1-v6, heuristic_tune) incl. the rejected/regressed ones as negative-result records.
NOT touched: the abandoned-RL files (rl/, checkpoints/ 3.3MB, agents/rl_inference.py, main.py) —
dead weight from the dropped RL path, but left in place pending an explicit call to purge them.

### What to watch / next levers
  - GROUND TRUTH: when 53185991 + 53186031 converge — does the brain (v6) beat v2's 970? v2 should
    re-confirm ~970. If v6 > 970, the brain transfers and porting H1000's 2p tactical layer (hammer,
    multiprong, anti-snipe, defensive reserve) becomes worth the (large) effort. If v6 ≤ 970, the
    brain doesn't transfer (local eval misled again) and v2 stays our agent.
  - DO NOT trust local 4p FFA for reach/aggression tuning — it has now mispredicted twice.

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

The "5 strongest" pool = adv_hellburner, adv_ver16, adv_proto_v15, adv_lb958, adv_in_progress.
The "8 mixed" generalization pool = adv_ver16, adv_proto_v15, adv_lb958, adv_in_progress,
adv_distance, adv_lbmax, adv_structured, adv_rf_v1 (drops the near-identical adv_hellburner,
adds 4 older/weaker public agents the tuning never saw). ffa4 samples 3 opponents per game.

### Session status / hand-off
- **submission.py = agents/heuristic_v5.py (md5 5240492b...).** v5 = v2 + 4p-only reach=30.
  v2 (the live 970 agent) is unchanged in the repo and v5 is identical to it in 2p.
- **SUBMITTED 2026-05-29 13:34 UTC — ref 53154166 (status PENDING).** Score will converge over
  hours (do not read the early number). Expectation: ≥ v2's 970 (better in 4p, identical in 2p).
  v2's own submission (53118635, 970.0) remains live in parallel as a safety net — each
  submission is rated independently. Quota at submit time: ~1 of 5 free.
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

