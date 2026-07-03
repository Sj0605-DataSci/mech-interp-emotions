#!/bin/bash
# Run all three missing analyses (Elo prefs, User vs Asst, Steering) across all 19 models.
# Usage: bash run_parts123.sh [--skip-done]

set -e
PY="/home/sanyam-0605/Desktop/sanyam_projects/.venv/bin/python"
PROJ="/home/sanyam-0605/Desktop/sanyam_projects/mech interp emotions"
SKIP_DONE=1  # always skip if output already exists

MODELS=(
  "EleutherAI/pythia-70m-deduped"
  "EleutherAI/pythia-160m-deduped"
  "EleutherAI/pythia-410m-deduped"
  "EleutherAI/pythia-1b-deduped"
  "EleutherAI/pythia-1.4b-deduped"
  "EleutherAI/pythia-2.8b-deduped"
  "EleutherAI/pythia-6.9b-deduped"
  "EleutherAI/pythia-12b-deduped"
  "cerebras/Cerebras-GPT-111M"
  "cerebras/Cerebras-GPT-256M"
  "cerebras/Cerebras-GPT-590M"
  "cerebras/Cerebras-GPT-1.3B"
  "cerebras/Cerebras-GPT-2.7B"
  "cerebras/Cerebras-GPT-6.7B"
  "cerebras/Cerebras-GPT-13B"
  "Qwen/Qwen3.5-0.8B-Base"
  "Qwen/Qwen3.5-2B-Base"
  "Qwen/Qwen3.5-4B-Base"
  "Qwen/Qwen3.5-9B-Base"
)

TOTAL=${#MODELS[@]}
DONE=0

for REPO in "${MODELS[@]}"; do
  TAG="${REPO//\//__}"
  DONE=$((DONE + 1))
  echo ""
  echo "━━━ [$DONE/$TOTAL] $REPO ━━━"

  # --- Elo / activity preferences (Part 1 completion) ---
  if [ -f "$PROJ/out/$TAG/elo_prefs.json" ]; then
    echo "  [✓] elo_prefs already done, skipping"
  else
    echo "  [→] elo_prefs: running ($((30 * 29 / 2)) pairwise comparisons)..."
    "$PY" "$PROJ/src/elo_prefs.py" --repo "$REPO" 2>&1 \
      | grep -vE "^(Some|The |Loading |Special |Setting )" \
      | sed 's/^/    /'
  fi

  # --- User vs Assistant probes (Part 2 completion) ---
  if [ -f "$PROJ/out/$TAG/user_vs_asst.json" ]; then
    echo "  [✓] user_vs_asst already done, skipping"
  else
    echo "  [→] user_vs_asst: 8 scenarios × 2 positions..."
    "$PY" "$PROJ/src/user_vs_asst.py" --repo "$REPO" 2>&1 \
      | grep -vE "^(Some|The |Loading |Special |Setting )" \
      | sed 's/^/    /'
  fi

  # --- Causal steering (Part 3) ---
  if [ -f "$PROJ/out/$TAG/steer_eval.json" ]; then
    echo "  [✓] steer_eval already done, skipping"
  else
    echo "  [→] steer_eval: 10 emotions × 5 prompts..."
    "$PY" "$PROJ/src/steer_eval.py" --repo "$REPO" 2>&1 \
      | grep -vE "^(Some|The |Loading |Special |Setting )" \
      | sed 's/^/    /'
  fi

  echo "  ✓ $REPO complete"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ALL DONE — outputs written to out/<model>/{elo_prefs,user_vs_asst,steer_eval}.json"
echo "Run: python src/scaling_plots.py   to regenerate scaling CSVs with new metrics"
echo "Run: python src/generate_report.py  to rebuild the PDF"
