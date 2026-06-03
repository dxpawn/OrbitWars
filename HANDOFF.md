# Orbit Wars — Hand-off (2026-06-02)

Short version for the team. Full detail in `diary.md` (top entry).

## ⚡ PIVOT (2026-06-02 cont.) — building a learned target re-ranker (distill the LB-1140 transformer)
- **Heuristic knob-tuning is exhausted/proven dead on the ladder.** `VAL_PROD_W=16` (53264512) settled
  **~910** — did NOT transfer despite +14 local 2p. exp30 (LB1072 = HEURISTIC1000 + one 2p-cap line) is a
  knob we already have (saturated). The whole heuristic field caps ~1000-1072.
- **New paradigm (the real competition is an RL-class relative/winner-take-all grade):** a friend gave us
  his **LB 1140.9** agent with full permission. It's a **heuristic hull + a learned 46-feature TARGET
  RE-RANKER** (imitation-trained on top players, pure-Python transformer, separate 2p/4p). We will NOT ship
  his file — we **distill his model (as an oracle) into OUR OWN re-ranker on OUR hull**, then RL-finetune to
  **beat 1140**. Folder: `other_adversaries/submission_feature46_transformer_v2_late_recapture_2p_v1/`.
- **Phase 0 de-risk DONE:** his `score_many` works as a standalone oracle (distillation viable ✓); v6 is
  already **3-3 vs him in 2p** → his edge is in **4p**, which focuses the re-ranker. Registered as
  `adv_friend_tf`. We already have an `rl/` stack (policy/ppo/imitation/features) to reuse.
- **Plan = 5 phases (tasks #3-7):** feature encoder on our hull → distill dataset (our features→his scores,
  64-way parallel) → train+export pure-Python student → integrate+validate+submit → RL-finetune past 1140.
- **Compute:** 36c/72t, CPU-bound, parallel at ~64. No GPU needed (small net; rollouts dominate).
- Full detail: `diary.md` top entry. Fallbacks intact (`heuristic_v6_1017`, v6 brain 997.5, team best 1016.5).

## TL;DR (2026-06-02 — superseded by the PIVOT note above; PROD_W=16 settled ~910, did NOT transfer)
- **Ladder state:** team best is now **1023.5** (ref 53244971) + **1016.5** (53244319), both 06-01.
  Our v6 brain (53186031) **drifted 1017→997.5**. Best-of-N ranking, so the team sits at 1023.5.
- **NEW transferable lever found: `VAL_PROD_W` 8 → 16.** Held-out 2p A/B (`eval/confirm_ab.py`,
  diverse pool, 2 seedbases): **+14 net broad, replicated** (12=+11, 16=+14, monotonic). **4p FFA
  neutral** (`eval/ffa4.py`, 500 games: 50.2%→49.4%, n.s.). First lever since the brain to pass the
  held-out test that killed the hammer/self-emit/LB1050 tweaks. By our signal hierarchy (2p h2h is
  ladder-predictive; 4p FFA anti-predictive) this should be net-positive on the mixed ladder.
- **SUBMITTED for validation: ref 53264512** (v6 brain + `VAL_PROD_W=16`, the ONLY diff), 06-01 18:29
  UTC, PENDING. It's a marginal change on the v6 base (997.5) so it likely lands **below 1023.5** —
  it won't take #1, but best-of can't hurt rank. **Purpose: confirm the +14 2p gain transfers to the
  ladder vs v6's 997.5.** If it does, carry `VAL_PROD_W=16` onto the stronger 1023.5 line.
- **DEAD this session:** phantom self-emit rate (`V6_SELF_EMIT`) — noise (+6 then −4 on 2nd seedbase).
- `submission.py` = the shipped file (one-line PROD_W change). **Uncommitted** — commit when ready.
  Quota: 5/UTC-day shared; 3 used on 06-01 (2 teammate + this), **2 left today**.

## TL;DR (2026-05-31 — superseded by the 06-02 note above)
- **Our best is now `agents/heuristic_v6.py` = Kaggle 1017.2** (forward-sim brain on v2's reach-38).
  It is the team's active/top submission.
- **v6 beat v2 by +105 in a controlled same-day paired test:** v6 = 1017.2, a fresh re-submit of
  the *identical* v2 = 911.7, both submitted within 2 min on 05-30 and converged ~24h against the
  same opponent pool. The forward-projection + leader-relative + 1-ply-search BRAIN transfers —
  this was the open question, now answered yes.
- **Cross-day absolute scores are untrustworthy (proven):** the same v2 code scored 970.0 on 05-28
  but 911.7 on 05-30. The ladder scale drifts as competitors strengthen. **Only same-day pairings
  are reliable.** (This is why we re-submitted v2 next to v6 — to get a clean control.)
- **`heuristic_v5` REGRESSED (914.7) — do NOT ship.** Its MAX_DISTANCE 38→30 in 4p won local 4p FFA
  by +21..+35 but lost ladder points. 2nd local-4p-FFA win that regressed the ladder. **⇒ local 4p
  FFA win-share is anti-predictive for reach/aggression tuning. Trust ONLY ladder for those.**
- **IN PROGRESS:** porting HEURISTIC1000's 2p tactical layer (hammer, multiprong, anti-snipe,
  defensive reserve) onto v6 — the plan's triggered contingency now that the brain is proven.

## Submission history (key rows)
- **53186031 — heuristic_v6 (brain, reach 38): 1017.2** ← current best / active.
- **53185991 — heuristic_v2 (re-submit, reach 38): 911.7** (control; same code scored 970 on 05-28).
- **53154166 — heuristic_v5 (reach 30 in 4p): 914.7** ← regressed, do not ship.
- **53118635 — heuristic_v2 (first submit): 970.0** (05-28 scale — not comparable to 05-30 numbers).

## The scores converged (this is the whole lesson)
Kaggle Arena scores drift for *hours*. Final converged public scores:

| Agent | ref | Score |
|---|---|---|
| heuristic_v2 (bug-fixed hellburner, reach 38) | 53118635 | **970.0** ← best |
| hellburner original (bugs intact) | 53118897 | 966.3 |
| hellburner + "local tweaks to beat it" | 53125217 | 945.3 ← regressed |
| **heuristic_v5 (reach 30 in 4p)** | 53154166 | **~915** ← REGRESSED |
| ver16 | 53110595 | 816.0 |

Takeaways:
1. **hellburner/v2 base > everything.** Build on `heuristic_v2`, NOT v5, NOT ver16.
2. **Local 4p FFA win-share does NOT predict the ladder — twice now it was anti-correlated**
   (v5's reach-30 +32 local but −52 ladder; teammate's tweaks similar). Do not tune reach or
   aggression to local 4p FFA. The only trustworthy signal is a ladder submission.

## What's verified about v2
- **Robust:** 0 exceptions, max turn 341ms (cap 1.0s) over ~7,300 real turns. No timeout risk.
- **Strong, disciplined:** plays cautiously (skips ~63% of 4p turns) and that discipline wins.

## Experiments that FAILED / REGRESSED (don't redo)
- **v5** (`heuristic_v5.py`): MAX_DISTANCE 38→30 in 4p. Local +21..+35; **ladder −52 (918<970)**.
- **v3** (`heuristic_v3.py`): boost strongest-enemy target value in 4p. Local-neutral. Not shipped.
- **v4** (`heuristic_v4.py`): loosen 4p threat model. Local 30.5% vs v2 38% (7.5pt worse). Not shipped.
- **v6 2p-aggression knobs** (`eval/sweep_v6_2p.py`): NO knob closes the 2p gap to HEURISTIC1000;
  longer 2p reach is sharply harmful (−11/−15). The 2p gap is structural, not tunable.
- **v6 cheap 2p tactical knobs** (oversend / press / def_frac, `V6_*`, all default-OFF): paired N=60
  vs H1000 all neutral-to-negative (oversend +0, press +0 inert, def_frac −3). Dead.
- **v6 PERSISTENT STAGGERED HAMMER** (`V6_HAMMER`, default-OFF, the big 05-31 build): looked +12 vs
  the v6 *mirror* but the held-out cross-opponent confirm (`eval/confirm_hammer.py`) showed it is
  WORSE vs every opponent — heuristic_v2 −33, adv_hellburner −25, proto_v15 −13, H1000 −7, lb958 −1.
  Telegraphed buildup + reserved-idle ships kills responsiveness. **NOT shipped.** ⇒ The whole
  tactical-port direction is exhausted; only the structural BRAIN ever transferred. Hold v6=1017.
- **LB1050 "council" refinements** (`V6_SNAP_WEIGHT`, `V6_ARR_DECAY`, `V6_DEPTH2`, all default-OFF):
  LB1050 (new in-repo, 1050 LB) is an H1000 sibling with the SAME brain+value-fn as v6. Ported its 3
  structural deltas; held-out A/B (`eval/confirm_ab.py`): snap-weight ~0, arr-decay ~0, depth-2 SUM
  net −10 (v2 −3/hellburner −7/proto −5, lb958 +2/H1000 +3). None transfers. **NOT shipped.**
  ⇒ Studying stronger same-family agents is tapped out — they share our brain. Hold v6=1017.

## v6 brain (the current experiment) — how it works & what's left
- `forward_project()` projects all planets forward ~18 turns w/ phantom opponent launches;
  `forward_score()` = leader-relative (ours − strongest-opponent in ships + 5·planets + 8·prod);
  `plan_midgame()` 1-ply search commits best projected-gain captures under a 0.85s budget.
- Tuned: `FWD_EMIT_FRAC 0.20→0.10` (held-out +7/+11/+8). Other knobs dead.
- 2p-gap diagnosis (`eval/diag_2p.py`): v6 is even with H1000 to ~step 50, then stalls/bleeds
  planets in the midgame. Closing it needs H1000's tuned TACTICAL layer (hammer, multiprong,
  anti-snipe, defensive reserve/coalitions) — a large port, only worth it **if v6 beats 970 on
  the ladder** (i.e. if the brain transfers at all).

## How to evaluate a candidate (use these, not 1v1-vs-previous-best)
- 4p FFA win-share (the format Kaggle scores):
  `python -m eval.ffa4 --hero <name> --games 250 --offset 200000`
  Lineups/seats/seeds are fixed by index, so two heroes with the same `--games`/`--offset`
  play IDENTICAL games — a fair paired A/B. **Always confirm a winner on a held-out `--offset`
  the tuning never used** (this session that flipped a false positive negative).
- Parameter search: `python -m eval.sweep --games 200` (screen) then `--offset 100000` (confirm).
- Robustness/timing audit: `python -m eval.diag_v2` (adapt the import for a new agent).
- 2p sanity: `python -m eval.arena_cli h2h --a <name> --b heuristic_v2 --games 100`

## Operational notes
- **Submission quota is shared across the team (5 / 24h).** Coordinate before submitting.
- Submit the SINGLE FILE: `kaggle competitions submit orbit-wars -f submission.py -m "..."`.
  Multi-file tarballs are rejected (SubmissionStatus.ERROR).
- Comets are NOT worth chasing: ver16 chases them and scores *lower* (816 < 970). Comet
  production is 1/turn and they expire — low value. v2 ignores them and wins.
- **Security:** the Kaggle API token was shared in plaintext earlier. Regenerate it at
  https://www.kaggle.com/settings/api before the project ends.
