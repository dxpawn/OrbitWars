# Orbit Wars — Hand-off (2026-05-29)

Short version for the team. Full detail in `diary.md` (top entry).

## TL;DR
- **Our best agent is `agents/heuristic_v2.py` — Kaggle score 970.** It's a bug-fixed
  clone of the public `hellburner` agent. `submission.py` is a byte-identical copy of it.
- **Keep submitting v2-based work, NOT ver16.** Converged scores: v2 **970** vs ver16 **816**.
- **Do not "tune to beat the current agent locally" — it backfires on Kaggle.** Proven 3x.

## The scores actually converged (this matters)
Kaggle Arena scores drift for *hours* after submission as games accumulate. An early
reading is meaningless. Final, converged public scores:

| Agent | ref | Score |
|---|---|---|
| heuristic_v2 (bug-fixed hellburner) | 53118635 | **970.0** |
| hellburner original (bugs intact) | 53118897 | 970.1 |
| hellburner + "local tweaks to beat it" | 53125217 | **925.4** ← regressed |
| ver16 | 53110595 | 816.0 |

Takeaways:
1. **hellburner base > ver16 base.** Build on `heuristic_v2`.
2. The "hellburner + local tweaks" entry (925) scored **lower than plain hellburner (970)** —
   the local tuning *hurt*. This is the central lesson: beating the previous version in a
   local 1v1 does NOT predict Kaggle gains. It usually means you overfit.

## What's verified about v2
- **Robust:** 0 exceptions, max turn 341ms (the per-turn cap is 1.0s) over ~7,300 real
  turns in 2p+4p. No silent crashes, no timeout risk.
- **Strong in 4p FFA:** 38.0% win-share (CI [31.6, 44.9]) vs the 3 strongest public agents;
  fair share is 25%. v2 plays *cautiously* (skips ~63% of 4p turns) and that discipline wins.

## Experiments that FAILED (don't redo these)
- **v3** (`agents/heuristic_v3.py`): boost target value for the strongest enemy in 4p.
  Neutral (45% = 45%). Not shipped.
- **v4** (`agents/heuristic_v4.py`): loosen the 4p worst-case threat model so v2 is less
  passive. Provably identical to v2 in 2p (1881 turns, 0 diffs); in 4p it scored **30.5% vs
  v2's 38.0% on identical games — 7.5 pts worse.** Over-aggression gets you ganged up. Not shipped.

Both kept in the repo as documented negative results.

## How to evaluate a candidate (use these, not 1v1-vs-previous-best)
- 4p FFA win-share (the format Kaggle scores):
  `python -m eval.ffa4 --hero <name> --games 200`
  Lineups/seats/seeds are fixed by game index, so two heroes evaluated with the same
  `--games` play IDENTICAL games — a fair paired A/B. Beat v2's 38.0% with margin before
  even *considering* a submission, and remember local results may still not transfer.
- Robustness/timing audit: `python -m eval.diag_v2.py` (adapt the import for a new agent).
- 2p sanity: `python -m eval.arena_cli h2h --a <name> --b heuristic_v2 --games 100`

## Operational notes
- **Submission quota is shared across the team (5 / 24h).** Coordinate before submitting.
- Submit the SINGLE FILE: `kaggle competitions submit orbit-wars -f submission.py -m "..."`.
  Multi-file tarballs are rejected (SubmissionStatus.ERROR).
- Comets are NOT worth chasing: ver16 chases them and scores *lower* (816 < 970). Comet
  production is 1/turn and they expire — low value. v2 ignores them and wins.
- **Security:** the Kaggle API token was shared in plaintext earlier. Regenerate it at
  https://www.kaggle.com/settings/api before the project ends.
