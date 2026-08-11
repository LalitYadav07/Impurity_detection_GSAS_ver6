#!/usr/bin/env bash
set -euo pipefail

test -n "${GALAXY_URL:-}"
test -n "${GALAXY_API_KEY:-}"
test -n "${HISTORY_ID:-}"

exec python -m radar_pd_nova --host 0.0.0.0 --port 8080 --timeout 0
