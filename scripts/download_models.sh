#!/usr/bin/env bash
# Download all model families + dataset for the emotion-concepts scaling study.
# Uses the project venv's `hf` CLI. Token is read from HF_TOKEN env var.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/download_models.sh            # everything
#   HF_TOKEN=hf_xxx ./scripts/download_models.sh qwen       # one family
#   HF_TOKEN=hf_xxx ./scripts/download_models.sh dataset    # just the dataset
#
# Models land in the shared HF cache (~/.cache/huggingface); load them later
# by repo id with AutoModelForCausalLM.from_pretrained(...).

set -euo pipefail

HF="/home/sanyam-0605/Desktop/sanyam_projects/.venv/bin/hf"
export HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-info}"
# Faster transfers if hf_transfer is installed; harmless otherwise.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: set HF_TOKEN env var first." >&2
  exit 1
fi

# NOTE: do NOT pass `--exclude` here. The hf 1.20 CLI mis-parses trailing
# `--exclude PAT1 PAT2 ...` as the positional `filenames` list, which silently
# downloads NOTHING ("Fetching 0 files"). These repos are safetensors-native
# and small, so we just pull the whole repo.

dl_model () {
  local repo="$1"
  echo ""
  echo "=================================================================="
  echo ">>> $repo"
  echo "=================================================================="
  "$HF" download "$repo"
}

dl_dataset () {
  local repo="$1"
  echo ""
  echo ">>> dataset: $repo"
  "$HF" download "$repo" --repo-type dataset
}

# =========================================================================
# BASE / PRETRAINED-ONLY models across all three families.
# Rationale: the paper argues emotion concepts are inherited from PRETRAINING.
# Base models test that directly and keep all families comparable
# (Pythia and Cerebras-GPT are base-only suites anyway).
# =========================================================================

# ---- Qwen3.5 dense BASE ladder (LATEST Qwen series, released Feb 2026).
#      "-Base" = pretrained-only (no chat post-training).
#      Dense base ladder: 0.8B / 2B / 4B / 9B. (Verified against HF API.)
#      Larger Qwen3.5 are MoE (35B-A3B / 122B-A10B / 397B-A17B) — excluded to
#      keep a clean DENSE scaling study. Qwen also ships official SAEs for these
#      (SAE-Res-Qwen3.5-*-Base) — useful later for SAE-based feature comparison.
QWEN=(
  Qwen/Qwen3.5-0.8B-Base
  Qwen/Qwen3.5-2B-Base
  Qwen/Qwen3.5-4B-Base
  Qwen/Qwen3.5-9B-Base
)

# ---- Pythia deduped ladder (base LMs; tests pretraining-inherited concepts).
#      Full ladder 70m..12b — 12B is the LARGEST Pythia ever released.
PYTHIA=(
  EleutherAI/pythia-70m-deduped
  EleutherAI/pythia-160m-deduped
  EleutherAI/pythia-410m-deduped
  EleutherAI/pythia-1b-deduped
  EleutherAI/pythia-1.4b-deduped
  EleutherAI/pythia-2.8b-deduped
  EleutherAI/pythia-6.9b-deduped
  EleutherAI/pythia-12b-deduped
)

# ---- Cerebras-GPT ladder (Chinchilla-optimal base LMs).
#      Full ladder 111M..13B — 13B is the LARGEST Cerebras-GPT ever released.
CEREBRAS=(
  cerebras/Cerebras-GPT-111M
  cerebras/Cerebras-GPT-256M
  cerebras/Cerebras-GPT-590M
  cerebras/Cerebras-GPT-1.3B
  cerebras/Cerebras-GPT-2.7B
  cerebras/Cerebras-GPT-6.7B
  cerebras/Cerebras-GPT-13B
)

TARGET="${1:-all}"

case "$TARGET" in
  qwen)     for m in "${QWEN[@]}";     do dl_model "$m"; done ;;
  pythia)   for m in "${PYTHIA[@]}";   do dl_model "$m"; done ;;
  cerebras) for m in "${CEREBRAS[@]}"; do dl_model "$m"; done ;;
  dataset)  dl_dataset ryancodrai/emotion-probes ;;
  all)
    dl_dataset ryancodrai/emotion-probes
    for m in "${PYTHIA[@]}";   do dl_model "$m"; done   # smallest families first
    for m in "${CEREBRAS[@]}"; do dl_model "$m"; done
    for m in "${QWEN[@]}";     do dl_model "$m"; done   # up to 14B-Base last
    ;;
  *) echo "Unknown target: $TARGET  (use: all|qwen|pythia|cerebras|dataset)"; exit 1 ;;
esac

echo ""
echo "Done. Cache size:"
du -sh ~/.cache/huggingface
