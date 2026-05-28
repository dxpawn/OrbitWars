#!/bin/bash
# Build a Kaggle submission tarball with main.py + dependencies + checkpoint.
#
# Usage:
#   ./scripts/make_submission.sh                    # uses checkpoints/best.pt if present
#   ./scripts/make_submission.sh path/to/policy.pt  # explicit checkpoint
#
# Output: submission.tar.gz at repo root, ready for:
#   kaggle competitions submit orbit-wars -f submission.tar.gz -m "msg"

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
echo "Building submission.tar.gz with ${#FILES[@]} files..."
tar -czf submission.tar.gz "${FILES[@]}"

du -sh submission.tar.gz
echo "Done."
echo
echo "To submit:"
echo "  kaggle competitions submit orbit-wars -f submission.tar.gz -m \"your message\""
