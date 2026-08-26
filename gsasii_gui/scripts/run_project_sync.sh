#!/usr/bin/env bash
set -euo pipefail

source_project="${GSASII_PROJECT_PATH:?GSASII_PROJECT_PATH is required}"
output_project="${GSASII_OUTPUT_PROJECT:?GSASII_OUTPUT_PROJECT is required}"
last_digest=""
snapshot=/tmp/gsasii-project-snapshot.gpx

while true; do
    if [[ -s "${source_project}" ]]; then
        source_digest="$(sha256sum "${source_project}" | cut -d ' ' -f 1)"
        cp "${source_project}" "${snapshot}"
        snapshot_digest="$(sha256sum "${snapshot}" | cut -d ' ' -f 1)"
        current_digest="$(sha256sum "${source_project}" | cut -d ' ' -f 1)"
        if [[ "${source_digest}" == "${snapshot_digest}" \
              && "${source_digest}" == "${current_digest}" \
              && "${source_digest}" != "${last_digest}" ]]; then
            cat "${snapshot}" > "${output_project}"
            chmod a+r "${output_project}"
            last_digest="${source_digest}"
            echo "Saved GPX snapshot ${source_digest:0:12} to the Galaxy output"
        fi
    fi
    sleep 0.5
done
