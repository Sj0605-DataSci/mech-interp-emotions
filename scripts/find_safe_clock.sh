#!/usr/bin/env bash
# Find the highest SUSTAINED-safe GPU clock on this GB10 by stepping the locked
# clock up, running a continuous matmul load at each step, and watching temp.
# Backs off immediately if temp crosses TEMP_HARD, and always restores SAFE_CLOCK
# on exit. You picked 1794 as your known-safe floor; 3003 shut the box down at 96C.
#
# Usage:  sudo ./scripts/find_safe_clock.sh
#   TEMP_TARGET=80 TEMP_HARD=86 sudo ./scripts/find_safe_clock.sh
#
# Output: the highest clock whose sustained temp stayed <= TEMP_TARGET.

set -uo pipefail
PY="/home/sanyam-0605/Desktop/sanyam_projects/.venv/bin/python"
SAFE_CLOCK="${SAFE_CLOCK:-1794}"   # your known-safe floor; restored on exit
TEMP_TARGET="${TEMP_TARGET:-80}"   # want sustained temp at/under this
TEMP_HARD="${TEMP_HARD:-86}"       # abort a step immediately above this
STEPS=(1980 2100 2220 2340 2460)   # candidate clocks between 1794 and 3003
LOAD_SECS="${LOAD_SECS:-35}"       # sustained load per step
BEST="$SAFE_CLOCK"

restore() { echo ">>> restoring safe clock ${SAFE_CLOCK}MHz"; nvidia-smi -lgc "$SAFE_CLOCK" >/dev/null 2>&1; nvidia-smi -rgc >/dev/null 2>&1 || true; }
trap restore EXIT INT TERM

temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits; }

# Background continuous matmul load (bf16, GB10-sized) — stops when flag file removed.
cat > /tmp/_gb10_load.py <<'PYEOF'
import torch, time, sys, os
torch.cuda.init()
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
while os.path.exists("/tmp/_gb10_load.flag"):
    for _ in range(50): a = (a @ b)
    torch.cuda.synchronize()
PYEOF

echo "Safe floor=${SAFE_CLOCK}  target<=${TEMP_TARGET}C  hard-abort>${TEMP_HARD}C"
for clk in "${STEPS[@]}"; do
  echo ""
  echo "=== testing ${clk} MHz for ${LOAD_SECS}s ==="
  if ! nvidia-smi -lgc "$clk" >/dev/null 2>&1; then echo "  can't set ${clk} (need sudo?), stopping"; break; fi
  touch /tmp/_gb10_load.flag
  "$PY" /tmp/_gb10_load.py &
  LOAD_PID=$!
  peak=0; aborted=0
  for ((s=0; s<LOAD_SECS; s++)); do
    sleep 1
    t=$(temp); [[ "$t" -gt "$peak" ]] && peak=$t
    printf "\r  t=%2ds  temp=%sC  peak=%sC " "$s" "$t" "$peak"
    if [[ "$t" -ge "$TEMP_HARD" ]]; then echo ""; echo "  !! ${t}C >= hard ${TEMP_HARD}C — aborting this step"; aborted=1; break; fi
  done
  rm -f /tmp/_gb10_load.flag; wait "$LOAD_PID" 2>/dev/null
  echo ""
  if [[ "$aborted" -eq 1 || "$peak" -gt "$TEMP_TARGET" ]]; then
    echo "  ${clk}MHz -> peak ${peak}C  (too hot, not safe)"; break
  else
    echo "  ${clk}MHz -> peak ${peak}C  OK"; BEST="$clk"
  fi
  sleep 5  # cool a moment between steps
done

echo ""
echo "================================================="
echo "Highest sustained-safe clock: ${BEST} MHz (<= ${TEMP_TARGET}C)"
echo "To use it for the run:  nvidia-smi -lgc ${BEST}"
echo "================================================="
