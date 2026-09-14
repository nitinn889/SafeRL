#!/usr/bin/env bash
# One-command Unreal demo: trained policy flying the debris field, live.
#
#   ./run_ue_demo.sh                      # 450k checkpoint, full 5-hazard field
#   ./run_ue_demo.sh --checkpoint X.zip   # some other checkpoint
#   ./run_ue_demo.sh --fps 15             # slower, easier to watch
#
# What it does, in order:
#   1. finds the Unreal engine install, remounting its drive if needed
#   2. launches the UE editor, which auto-arms a PIE session and builds the
#      phase 8c scene (NASA ACE satellite, Poly Haven rocks, NASA starmap)
#   3. waits for that session to reach live-mirror mode
#   4. runs the trained policy in this repo's venv and streams per-episode
#      metrics to THIS terminal while UE renders the result
#
# The policy cannot run inside Unreal: UE 5.8 embeds Python 3.11, this venv
# is 3.14, and torch/SB3 ship interpreter-locked compiled extensions. So the
# real SafeNav3DEnv + SafetyShield + policy run here and publish world state;
# UE mirrors it. UE renders, it does not simulate -- the role every phase
# since 2 measured it into.
#
# Ctrl-C stops the policy and shuts the editor down.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="saferl/eval/phase9/saferl_phase9_best.zip"   # 450k, the reported checkpoint
FPS=30
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --fps)        FPS="$2"; shift 2 ;;
    -h|--help)    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' \
                      "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

cd "$REPO"

# ── 1. engine ────────────────────────────────────────────────────────────
# The engine lives on a separate drive that does not auto-mount and is not in
# fstab, so a fresh boot (or a fresh session) usually has it unmounted.
# udisksctl does this without sudo; plain `mount` would prompt for a password.
ENGINE="/run/media/$USER/Windows/unreal_engine/Engine/Binaries/Linux/UnrealEditor"
if [[ ! -x "$ENGINE" ]]; then
  echo "[demo] engine not found at $ENGINE -- attempting to mount its drive"
  udisksctl mount -b /dev/nvme0n1p3 >/dev/null 2>&1 || true
fi
if [[ ! -x "$ENGINE" ]]; then
  echo "[demo] ERROR: Unreal engine binary still not found at:"
  echo "         $ENGINE"
  echo "       Mount the drive holding the engine, or edit ENGINE in this script."
  echo "       For a quick check with no Unreal at all, run instead:"
  echo "         .venv/bin/python -m saferl.demo.run_demo"
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[demo] ERROR: checkpoint not found: $CHECKPOINT"; exit 1
fi

UPROJECT="$REPO/ue_spike/SafeRLUESpike.uproject"
HEARTBEAT="$REPO/ue_spike/pie_heartbeat.log"
STATE="$REPO/ue_spike/live_policy_state.json"
UE_LOG="${TMPDIR:-/tmp}/saferl_ue_demo.log"

cleanup() {
  echo ""
  echo "[demo] shutting down..."
  [[ -n "${UE_PID:-}" ]] && kill "$UE_PID" 2>/dev/null
  sleep 1
  [[ -n "${UE_PID:-}" ]] && kill -9 "$UE_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# ── 2. launch the editor ─────────────────────────────────────────────────
rm -f "$HEARTBEAT" "$STATE"
echo "[demo] launching Unreal editor (log: $UE_LOG)"
SAFERL_RUN_PIE=1 SAFERL_UNCAP=1 SAFERL_LIVE_POLICY=1 \
  "$ENGINE" "$UPROJECT" -nosound -log > "$UE_LOG" 2>&1 &
UE_PID=$!

# ── 3. wait for live-mirror mode ─────────────────────────────────────────
echo "[demo] waiting for the PIE session to come up (typically ~40s)..."
DEADLINE=$((SECONDS + 300))
while ! grep -q "LIVE POLICY MODE" "$HEARTBEAT" 2>/dev/null; do
  if ! kill -0 "$UE_PID" 2>/dev/null; then
    echo "[demo] ERROR: the editor exited before reaching live mode. Tail of $UE_LOG:"
    tail -20 "$UE_LOG"; exit 1
  fi
  if (( SECONDS > DEADLINE )); then
    echo "[demo] ERROR: timed out waiting for live mode. Tail of $UE_LOG:"
    tail -20 "$UE_LOG"; cleanup
  fi
  sleep 2
done
grep -E "scene built|live: found" "$HEARTBEAT" | tail -2
echo "[demo] UE is live and mirroring."

# ── 4. run the policy, streaming metrics here ────────────────────────────
echo "[demo] starting the trained policy: $CHECKPOINT"
echo "[demo] Ctrl-C to stop."
echo ""
.venv/bin/python -u -m saferl.demo.live_policy_bridge \
  --checkpoint "$CHECKPOINT" --fps "$FPS" "${EXTRA[@]}"

cleanup
