#!/usr/bin/env bash
set -euo pipefail

printf '[radar-pd-nova] launcher starting (uid=%s, ep_path=%s)\n' \
  "$(id -u)" "${EP_PATH:-<unset>}"

# Keep the Galaxy job in the foreground. The service manager restarts either
# web service if it exits and forwards both services' output to the Galaxy log.
exec python /usr/local/bin/supervise_services.py
