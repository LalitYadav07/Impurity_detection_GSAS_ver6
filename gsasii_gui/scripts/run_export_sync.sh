#!/usr/bin/env bash
set -uo pipefail

export_dir="${GSASII_EXPORT_DIR:?GSASII_EXPORT_DIR is required}"
output_archive="${GSASII_OUTPUT_ARCHIVE:?GSASII_OUTPUT_ARCHIVE is required}"
state_file="${GSASII_SESSION_DIR:-/tmp}/export-manifest.json"

sync_once() {
    /opt/conda/bin/python /opt/gsasii-gui/snapshot_exports.py \
        "${export_dir}" "${output_archive}" "${state_file}" || {
        echo "Could not update the Galaxy export archive; retrying" >&2
        return 0
    }
}

trap 'sync_once; exit 0' INT TERM

while true; do
    sync_once
    sleep 1
done
