#!/usr/bin/env bash
# Run the full analysis layer over every model that has emotion_vectors.pt:
#   geometry (cosine/cluster/circumplex) + logit-lens, then cross-family
#   scaling plots. Safe to re-run; geometry/logitlens overwrite per model.
#
# Usage: ./scripts/analyze_all.sh

set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/sanyam-0605/Desktop/sanyam_projects/.venv/bin/python"
export HF_TOKEN="${HF_TOKEN:-}"

# Geometry is offline (no model load) -> do all at once.
"$PY" "$PROJECT/src/analyze_geometry.py" --all

# Logit-lens needs the model (final norm + unembed). One repo per out/ dir.
for d in "$PROJECT"/out/*/emotion_vectors.pt; do
  tag="$(basename "$(dirname "$d")")"
  repo="${tag//__//}"
  echo ""
  echo ">>> logit-lens: $repo"
  "$PY" "$PROJECT/src/analyze_logitlens.py" --repo "$repo" || echo "  (skipped: $repo)"
done

# Cross-family scaling plots + summary.csv
"$PY" "$PROJECT/src/scaling_plots.py"
echo ""
echo "Analysis done. See $PROJECT/out/scaling/"
