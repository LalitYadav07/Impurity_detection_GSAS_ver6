#!/usr/bin/env bash
set -euo pipefail

ROOT="${RADAR_PD_PROJECT_ROOT:-/home/cloud/radar-pd-ux/source}"
WORKSPACES="$(readlink -f "$ROOT/workspaces" 2>/dev/null || printf '%s' "$ROOT/workspaces")"
TEMP_WORKSPACES="$(readlink -f "$ROOT/workspaces/_temporary" 2>/dev/null || printf '%s' "$ROOT/workspaces/_temporary")"

# Remove generated GSAS backup/checkpoint artifacts after they are safely stale.
find "$WORKSPACES" -type f \
  \( -name '*.bak*.gpx' -o -name '*.temp.gpx' -o -name '*.checkpoint.gpx' -o -name '*.bak*.lst' -o -name '*.temp.lst' \) \
  -mmin +360 -delete 2>/dev/null || true

# Anonymous temporary sessions are not persistent workspaces.
find "$TEMP_WORKSPACES" -mindepth 1 -maxdepth 1 -type d \
  -mmin +1440 -exec rm -rf {} + 2>/dev/null || true

# Installer/catalog zip caches can be recreated by the installer if needed.
rm -f /home/cloud/radar-pd-local/cache/catalog_archives/*.zip 2>/dev/null || true
