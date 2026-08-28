#!/usr/bin/env bash
set -euo pipefail

until xdpyinfo -display "${DISPLAY:-:1}" >/dev/null 2>&1; do
    sleep 0.2
done

exec x11vnc \
    -quiet \
    -display "${DISPLAY:-:1}" \
    -forever \
    -shared \
    -nopw \
    -rfbport 5900 \
    -xdamage \
    -defer 5 \
    -wait 10 \
    -cursor most \
    -cursorpos \
    -repeat \
    -xkb
