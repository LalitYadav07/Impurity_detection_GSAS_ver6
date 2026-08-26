#!/usr/bin/env bash
set -euo pipefail

until xdpyinfo -display "${DISPLAY:-:1}" >/dev/null 2>&1; do
    sleep 0.2
done

xsetroot -solid '#edf4ef'
exec openbox --sm-disable
