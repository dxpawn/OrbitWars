# Orbit Wars — Hand-off (2026-05-30)

Short version for the team. Full detail in `diary.md` (top entry).

## TL;DR (2026-05-30 — supersedes the 05-29 note)
- **Proven best is `agents/heuristic_v2.py` = Kaggle 970** (bug-fixed `hellburner`, reach 38).
- **`heuristic_v5` REGRESSED to ~915 on the ladder — BELOW v2's 970.** Its only change
  (MAX_DISTANCE 38→30 in 4p) won local 4p FFA by +21..+35 but LOST 52 ladder points. The
  05-29 "v5 is best" claim is **WRONG**. This is the 2nd local-4p-FFA win that regressed the
  ladder (teammate's 945 too). **⇒ Our local 4p FFA metric is anti-predictive for reach/
  aggression tuning. Trust ONLY ladder submissions for these.**
- **`agents/heuristic_v6.py` — new EXPERIMENT (submitted, pending):** our own reimplementation
  of the public ~1000-1100 agent's decision core (global forward-projection + LEADER-RELATIVE
  value + 1-ply search) on v2's machinery & reach-38, so the ONLY diff from v2 is the brain.
  Local: beats v2-2p ~60% in 2p, ~tied 4p, timing-safe, 0 exc. **Whether it beats 970 is
  unknown until the ladder converges** — that's why it's submitted.

## Submissions today (2026-05-30)
- **53185991 — heuristic_v2 (reach 38):** re-submitted to make the proven 970 the ACTIVE agent
  (v5/915 had been the latest = our *worse* agent was live).
- **53186031 — heuristic_v6:** the brain experiment (rebased to reach-38). PENDING.
- Read converged scores in HOURS, not early. The KEY question: does v6 (brain) beat v2's 970?

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
