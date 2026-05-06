#!/usr/bin/env bash
# Stop, unload, and remove the asr-router launchd agent.
#
# This does NOT delete logs or jobs. Logs are at asr/logs/.
# Job artifacts are at ~/.asr-router/jobs/ (and stay even if the service is removed).

set -euo pipefail

LABEL="com.user.asr-router"
DST_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
GUI_DOMAIN="gui/$(id -u)"

launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true

if [ -f "$DST_PLIST" ]; then
  rm "$DST_PLIST"
  echo "✓ removed $DST_PLIST"
else
  echo "(plist already absent)"
fi

if launchctl list | grep -q "$LABEL"; then
  echo "⚠ launchd still lists $LABEL. Try logging out and back in."
  exit 1
fi

echo "✓ asr-router service uninstalled."
echo "  Logs preserved at asr/logs/"
echo "  Job artifacts preserved at ~/.asr-router/jobs/"
