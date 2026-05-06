#!/usr/bin/env bash
# Install the asr-router as a per-user launchd service that starts at login
# and stays alive (auto-restart on crash, with 10s throttle).
#
# Usage:
#   bash asr/scripts/install_launchd.sh        # install + load
#   bash asr/scripts/install_launchd.sh kick   # bounce (reload + restart)
#
# After install:
#   launchctl list | grep com.user.asr-router         # check status
#   tail -f asr/logs/asr-router.err.log               # follow logs
#   bash asr/scripts/uninstall_launchd.sh             # stop + unload + remove

set -euo pipefail

ASR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.user.asr-router"
SRC_PLIST="$ASR_ROOT/scripts/launchd/$LABEL.plist"
DST_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UV_BIN="$(command -v uv || echo /opt/homebrew/bin/uv)"
GUI_DOMAIN="gui/$(id -u)"

mkdir -p "$ASR_ROOT/logs"

# Render plist with absolute paths
sed -e "s#__ASR_ROOT__#$ASR_ROOT#g" \
    -e "s#/opt/homebrew/bin/uv#$UV_BIN#g" \
    "$SRC_PLIST" > "$DST_PLIST"

# Reload (idempotent: bootout will fail silently if not loaded)
launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$GUI_DOMAIN" "$DST_PLIST"
launchctl enable "$GUI_DOMAIN/$LABEL"
launchctl kickstart -k "$GUI_DOMAIN/$LABEL"

# Wait for service to come up (max 30s)
echo "Waiting for asr-router on :18081 ..."
for i in $(seq 1 30); do
  if curl -sf -m 1 http://127.0.0.1:18081/v1/models \
       -H "Authorization: Bearer sk-mlx" >/dev/null 2>&1; then
    echo "✓ asr-router is up at http://localhost:18081/v1 (key: sk-mlx)"
    echo "  also reachable on the LAN via http://$(hostname -s).local:18081/v1"
    echo
    echo "Logs: $ASR_ROOT/logs/asr-router.{out,err}.log"
    echo "Status: launchctl list | grep $LABEL"
    exit 0
  fi
  sleep 1
done

echo "⚠ Service did not respond within 30s. Check logs:"
echo "  tail $ASR_ROOT/logs/asr-router.err.log"
exit 1
