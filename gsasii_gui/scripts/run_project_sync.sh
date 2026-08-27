#!/usr/bin/env bash
set -uo pipefail

source_project="${GSASII_PROJECT_PATH:?GSASII_PROJECT_PATH is required}"
output_project="${GSASII_OUTPUT_PROJECT:?GSASII_OUTPUT_PROJECT is required}"
last_digest=""
snapshot="${GSASII_SESSION_DIR:-/tmp}/project-sync-snapshot.gpx"
output_copy="${output_project}.sync.$$"

cleanup() {
    rm -f "${snapshot}" "${output_copy}"
}
trap cleanup EXIT
trap 'exit 0' INT TERM

sync_once() {
    local source_digest snapshot_digest current_digest
    [[ -s "${source_project}" ]] || return 0

    source_digest="$(sha256sum "${source_project}" 2>/dev/null | cut -d ' ' -f 1)" || return 0
    [[ -n "${source_digest}" ]] || return 0
    cp "${source_project}" "${snapshot}" 2>/dev/null || return 0
    snapshot_digest="$(sha256sum "${snapshot}" 2>/dev/null | cut -d ' ' -f 1)" || return 0
    current_digest="$(sha256sum "${source_project}" 2>/dev/null | cut -d ' ' -f 1)" || return 0

    if [[ "${source_digest}" != "${snapshot_digest}" \
          || "${source_digest}" != "${current_digest}" \
          || "${source_digest}" == "${last_digest}" ]]; then
        return 0
    fi

    cp "${snapshot}" "${output_copy}" 2>/dev/null || return 0
    if mv -f "${output_copy}" "${output_project}" && chmod a+r "${output_project}"; then
        last_digest="${source_digest}"
        echo "Saved GPX snapshot ${source_digest:0:12} to the Galaxy output"
    else
        echo "Could not update the Galaxy GPX output; retrying" >&2
    fi
    rm -f "${output_copy}"
}

while true; do
    sync_once
    sleep 0.5
done
