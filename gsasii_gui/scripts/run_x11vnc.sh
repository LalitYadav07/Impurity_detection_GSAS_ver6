#!/usr/bin/env bash
set -euo pipefail

until xdpyinfo -display "${DISPLAY:-:1}" >/dev/null 2>&1; do
    sleep 0.2
done

exec x11vnc \
    -display "${DISPLAY:-:1}" \
    -forever \
    -shared \
    -nopw \
    -rfbport 5900 \
    -noxdamage \
    -repeat \
    -xkb
