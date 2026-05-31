# Orbit Wars — Hand-off (2026-05-31)

Short version for the team. Full detail in `diary.md` (top entry).

## TL;DR (2026-05-31 — supersedes the 05-30 note)
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
