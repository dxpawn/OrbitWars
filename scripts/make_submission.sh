#!/bin/bash
# Build a Kaggle submission for orbit-wars.
#
# IMPORTANT: As of 2026-05-28, Kaggle Arena for this competition rejects
# multi-file tarballs (returns SubmissionStatus.ERROR with no log). It accepts
# only a single Python file. So by default this script writes
# `submission.py`.
#
# AGENT CHOICE (2026-05-30): PROVEN best is agents/heuristic_v2.py (Kaggle 970).
# heuristic_v5 REGRESSED on the ladder (~915 < 970) — do NOT ship it. The current
# experiment is agents/heuristic_v6.py (forward-sim brain on v2's reach-38);
# pending ladder result. Set AGENT below to v2 (safe) or v6 (experiment).
#
# It still emits submission.tar.gz too, in case tarballs ever start working
# again, but the single-file `submission.py` is what to upload.
#
# Usage:
#   ./scripts/make_submission.sh                    # uses checkpoints/best.pt if present
#   ./scripts/make_submission.sh path/to/policy.pt  # explicit checkpoint
#
# Output:
#   - submission.py        (single-file, USE THIS for Kaggle)
#   - submission.tar.gz    (multi-file, currently rejected by Kaggle)
#
# Submit with:
#   kaggle competitions submit orbit-wars -f submission.py -m "msg"

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="${1:-checkpoints/best.pt}"

if [ -f "$CKPT" ] && [ "$CKPT" != "checkpoints/best.pt" ]; then
    # Copy specified checkpoint to expected location
    mkdir -p checkpoints
    cp "$CKPT" checkpoints/best.pt
    echo "Copied $CKPT -> checkpoints/best.pt"
fi

if [ ! -f checkpoints/best.pt ]; then
    echo "WARNING: no checkpoints/best.pt — will fall back to heuristic_v1"
fi

# What goes in the tarball
FILES=(
    main.py
    agents/__init__.py
    agents/heuristic_v1.py
    agents/heuristic_v2.py
    agents/heuristic_v5.py
    agents/heuristic_v6.py
    agents/rl_inference.py
    rl/__init__.py
    rl/features.py
    rl/policy.py
    rl/action_space.py
)
# Add checkpoint only if present
if [ -f checkpoints/best.pt ]; then
    FILES+=(checkpoints/best.pt)
fi

# opponents/__init__.py imports adversaries; in submission we don't need that
# so we replace with a minimal stub.
echo "Building submission.tar.gz with ${#FILES[@]} files (currently rejected by Kaggle)..."
tar -czf submission.tar.gz "${FILES[@]}"
du -sh submission.tar.gz

# Preferred path: single-file submission.py. Default to the v6 experiment;
# override with SUBMIT_AGENT=agents/heuristic_v2.py for the proven 970 baseline.
AGENT="${SUBMIT_AGENT:-agents/heuristic_v6.py}"
cp "$AGENT" submission.py
echo "submission.py <- $AGENT"
echo "Wrote single-file submission.py ($(wc -l < submission.py) lines, $(du -sh submission.py | cut -f1))"
echo
echo "Done."
echo
echo "To submit (use the single file, NOT the tarball):"
echo "  kaggle competitions submit orbit-wars -f submission.py -m \"your message\""
