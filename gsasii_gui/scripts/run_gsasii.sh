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
dbus-run-session -- /opt/conda/bin/python /opt/gsasii-gui/launch_gsasii.py "${project}"
status=$?
set -e

if /opt/conda/bin/python /opt/gsasii-gui/project_tracking.py sync \
    && /opt/conda/bin/python /opt/gsasii-gui/snapshot_exports.py \
        "${GSASII_EXPORT_DIR}" "${GSASII_OUTPUT_ARCHIVE}" "${GSASII_SESSION_DIR}/export-manifest.json"; then
    echo "Saved the active final GPX and named project branches to Galaxy"
else
    echo "Could not preserve the active saved GPX; inspect the session log" >&2
    [[ "${status}" -ne 0 ]] || status=74
fi

exit "${status}"
