#!/usr/bin/env bash
set -euo pipefail

project="${GSASII_PROJECT_PATH:?GSASII_PROJECT_PATH is required}"
output_project="${GSASII_OUTPUT_PROJECT:?GSASII_OUTPUT_PROJECT is required}"

until xdpyinfo -display "${DISPLAY:-:1}" >/dev/null 2>&1; do
    sleep 0.2
done

mkdir -p "${HOME}/.GSASII" "${XDG_RUNTIME_DIR}"
chmod 0700 "${XDG_RUNTIME_DIR}"

set +e
dbus-run-session -- /opt/conda/bin/GSAS-II "${project}"
status=$?
set -e

if [[ -s "${project}" ]]; then
    cat "${project}" > "${output_project}"
    chmod a+r "${output_project}"
fi

exit "${status}"
