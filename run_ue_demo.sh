#!/usr/bin/env bash
# One-command Unreal demo: trained policy flying the debris field, live.
#
#   ./run_ue_demo.sh                                        # 2D: 450k checkpoint, 5 hazards
#   ./run_ue_demo.sh --config saferl/configs/space3d.yaml  # 3D flight through a debris volume
#   ./run_ue_demo.sh --config saferl/configs/space3d.yaml --camera chase
#   ./run_ue_demo.sh --checkpoint X.zip   # some other checkpoint
#   ./run_ue_demo.sh --fps 15             # slower, easier to watch
#
# Cameras: fixed (2D default), wide (3D default), chase. Switch while running:
#   echo chase > ue_spike/camera_mode
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
CHECKPOINT=""        # default depends on the config's dims; see below
CONFIG=""
CAMERA=""
FPS=30
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --config)     CONFIG="$2"; shift 2 ;;
    --camera)     CAMERA="$2"; shift 2 ;;
    --fps)        FPS="$2"; shift 2 ;;
    -h|--help)    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' \
                      "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

cd "$REPO"

# ── 0. what the config describes ─────────────────────────────────────────
# UE's embedded Python can't be assumed to have PyYAML, so the config is read
# here with the venv and the scene gets the values it needs as env vars.
read -r DIMS NUM_DEBRIS ENV_SIZE < <(.venv/bin/python - "$CONFIG" <<'PYEOF'
import sys
from saferl.config import load_config
env = load_config(sys.argv[1] or None)["env"]
print(env.get("dims", 2), env["max_hazards"], env["size"])
PYEOF
)
if [[ -z "${DIMS:-}" ]]; then
  echo "[demo] ERROR: could not read config ${CONFIG:-saferl/configs/default.yaml}"; exit 1
fi
if [[ -z "$CHECKPOINT" ]]; then
  if [[ "$DIMS" == "3" ]]; then
    CHECKPOINT="saferl/eval/space3d/long/saferl_space3d_best.zip"
  else
    CHECKPOINT="saferl/eval/phase9/saferl_phase9_best.zip"   # 450k, the reported 2D checkpoint
  fi
fi

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
  echo "[demo] ERROR: checkpoint not found: $CHECKPOINT"
  [[ "$DIMS" == "3" ]] && echo "       The 3D checkpoint is produced by the 3D training pipeline (see README)."
  exit 1
fi

# Refuse to race a demo that is already up. Two bridges publishing to one
# state file corrupt each other's frames, and two editors fight over the
# same PIE session -- both of which happened during phase 10 when a
# previous run survived its shutdown.
if pgrep -f "saferl.demo.live_policy_bridge" >/dev/null 2>&1; then
  echo "[demo] ERROR: a policy bridge is already running:"
  pgrep -af "saferl.demo.live_policy_bridge" | sed 's/^/         /'
  echo "       Stop it first:  pkill -f saferl.demo.live_policy_bridge"
  exit 1
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
rm -f "$HEARTBEAT" "$STATE" "$REPO/ue_spike/camera_mode"
echo "[demo] launching Unreal editor: ${DIMS}D, ${NUM_DEBRIS} rocks (log: $UE_LOG)"
[[ -n "$CAMERA" ]] && export SAFERL_CAMERA="$CAMERA"
SAFERL_RUN_PIE=1 SAFERL_UNCAP=1 SAFERL_LIVE_POLICY=1 \
  SAFERL_DIMS="$DIMS" SAFERL_NUM_DEBRIS="$NUM_DEBRIS" SAFERL_ENV_SIZE="$ENV_SIZE" \
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
echo "[demo] switch camera: echo chase > ue_spike/camera_mode   (or wide / fixed)"
echo "[demo] Ctrl-C to stop."
echo ""
CFG_ARGS=()
[[ -n "$CONFIG" ]] && CFG_ARGS=(--config "$CONFIG")
.venv/bin/python -u -m saferl.demo.live_policy_bridge \
  --checkpoint "$CHECKPOINT" --fps "$FPS" "${CFG_ARGS[@]}" "${EXTRA[@]}"

cleanup
