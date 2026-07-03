#!/usr/bin/env bash
# Run extract -> compute_vectors for every downloaded model family/size.
# Skips models whose emotion_vectors.pt already exists (safe to re-run).
#
# Usage:
#   ./scripts/run_all.sh                 # full run, all stories
#   LIMIT=16 ./scripts/run_all.sh        # quick smoke run (16 stories/emotion)
#   BATCH=64 ./scripts/run_all.sh

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/sanyam-0605/Desktop/sanyam_projects/.venv/bin/python"
BATCH="${BATCH:-64}"
LIMIT="${LIMIT:-0}"     # 0 = all 1200/emotion

# Smallest-first so the cheap families validate the pipeline before big ones.
MODELS=(
  EleutherAI/pythia-70m-deduped
  cerebras/Cerebras-GPT-111M
  cerebras/Cerebras-GPT-256M
  EleutherAI/pythia-160m-deduped
  cerebras/Cerebras-GPT-590M
  EleutherAI/pythia-410m-deduped
  Qwen/Qwen3.5-0.8B-Base
  EleutherAI/pythia-1b-deduped
  cerebras/Cerebras-GPT-1.3B
  EleutherAI/pythia-1.4b-deduped
  Qwen/Qwen3.5-2B-Base
  cerebras/Cerebras-GPT-2.7B
  EleutherAI/pythia-2.8b-deduped
  Qwen/Qwen3.5-4B-Base
  cerebras/Cerebras-GPT-6.7B
  EleutherAI/pythia-6.9b-deduped
  Qwen/Qwen3.5-9B-Base
  cerebras/Cerebras-GPT-13B
  EleutherAI/pythia-12b-deduped
)

# N_STORIES: stories/emotion used for class means, FIXED across all models for a
# fair cross-family comparison (300 is cosine-0.9998 identical to full 1200).
# We extract this many for not-yet-done models (faster), and compute_vectors caps
# to it. Already-extracted models (full 1200) just get recomputed at this N.
N_STORIES="${N_STORIES:-300}"

EXTRA=()
# Extract only N_STORIES (4x faster on slow models) unless LIMIT overrides.
# NOTE: LIMIT defaults to 0 above; 0 means "no explicit override" -> use N_STORIES.
LIM="$LIMIT"
[[ "$LIM" == "0" ]] && LIM="$N_STORIES"
[[ "$LIM" != "0" ]] && EXTRA+=(--limit-per-emotion "$LIM")
# COMPILE=1 -> torch.compile every model. DISABLED by default: our batches have
# VARYING sequence lengths (each emotion pads to a different max), so dynamo
# recompiles every batch, blows the recompile_limit, and falls back to eager —
# paying all the compile cost for no gain (confirmed on GB10). Plain fla is faster.
COMPILE="${COMPILE:-0}"

for repo in "${MODELS[@]}"; do
  tag="${repo//\//__}"
  if [[ -f "$PROJECT/out/$tag/emotion_vectors.pt" ]]; then
    echo ">>> SKIP $repo (already computed)"
    continue
  fi
  RUN_EXTRA=("${EXTRA[@]}")
  [[ "$COMPILE" != "0" ]] && RUN_EXTRA+=(--compile)
  # Smaller batch for the biggest models (>=~6B): on GB10's shared 128GB pool,
  # batch 64 pushes the 12B/13B working set to ~80GB. Batch 32 cuts that to ~55GB
  # for safe headroom, at minor speed cost. BIG_BATCH overrides.
  RUN_BATCH="$BATCH"
  case "$repo" in
    *13B|*12b*|*6.7B|*6.9b*) RUN_BATCH="${BIG_BATCH:-32}" ;;
  esac
  echo ""
  echo "############################################################"
  echo "### $repo  (batch=$RUN_BATCH)  ${RUN_EXTRA[*]}"
  echo "############################################################"
  "$PY" "$PROJECT/src/extract.py" --repo "$repo" --batch-size "$RUN_BATCH" "${RUN_EXTRA[@]}"
  "$PY" "$PROJECT/src/compute_vectors.py" --repo "$repo" --n-stories "$N_STORIES"
done

echo ""
echo "All done. Vectors in $PROJECT/out/*/emotion_vectors.pt"
