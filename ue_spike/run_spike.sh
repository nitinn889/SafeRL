#!/usr/bin/env bash
# Runs the headless UE Python-API viability spike and writes
# ue_spike_results.json with the measured steps/sec.
#
# Requires the engine binary to be reachable -- on this machine that means
# /mnt/bigdata (the NTFS partition holding the UE 5.8 install) must be
# mounted first:
#   sudo mkdir -p /mnt/bigdata
#   sudo mount -t ntfs-3g -o remove_hiberfile,rw,uid=1000,gid=1000 /dev/nvme0n1p3 /mnt/bigdata
set -euo pipefail

UE_EDITOR_CMD="${UE_EDITOR_CMD:-/mnt/bigdata/unreal_engine/Engine/Binaries/Linux/UnrealEditor-Cmd}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPROJECT="${PROJECT_DIR}/SafeRLUESpike.uproject"

if [ ! -f "${UE_EDITOR_CMD}" ]; then
    echo "[ERROR] UnrealEditor-Cmd not found at ${UE_EDITOR_CMD}" >&2
    echo "        Mount the drive holding the engine install first (see comment above)." >&2
    exit 1
fi

echo "[*] Running headless UE Python-API spike (nullrhi, unattended)..."
"${UE_EDITOR_CMD}" "${UPROJECT}" \
    -run=pythonscript \
    -script="${PROJECT_DIR}/Content/Python/ue_bridge.py" \
    -nullrhi \
    -unattended \
    -nosplash \
    -nopause \
    -log

echo "[*] Results:"
cat "${PROJECT_DIR}/ue_spike_results.json" 2>/dev/null || echo "  (no results file written -- check the log above for errors)"
