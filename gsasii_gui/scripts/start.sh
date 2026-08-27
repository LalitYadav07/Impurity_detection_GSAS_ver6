#!/usr/bin/env bash
set -euo pipefail

source_project="${GSASII_SOURCE_PROJECT:-}"
output_project="${GSASII_OUTPUT_PROJECT:-}"
session_dir="${GSASII_SESSION_DIR:-${TMPDIR:-/tmp}/gsasii-session-$(id -u)}"

if [[ -z "${source_project}" || ! -s "${source_project}" ]]; then
    echo "GSASII_SOURCE_PROJECT must name a non-empty GPX file" >&2
    exit 66
fi
if [[ -z "${output_project}" ]]; then
    echo "GSASII_OUTPUT_PROJECT must name the Galaxy output GPX" >&2
    exit 64
fi

mkdir -p "${session_dir}"
if [[ ! -w "${session_dir}" ]]; then
    session_dir="${TMPDIR:-/tmp}/gsasii-session-$(id -u)"
    mkdir -p "${session_dir}"
fi
install -m 0644 "${source_project}" "${session_dir}/radar_pd_project.gpx"
install -m 0666 "${session_dir}/radar_pd_project.gpx" "${output_project}"

export GSASII_PROJECT_PATH="${session_dir}/radar_pd_project.gpx"
export GSASII_OUTPUT_PROJECT="${output_project}"
export HOME="${session_dir}/home"
export USER="$(id -un 2>/dev/null || id -u)"
export LOGNAME="${USER}"
export DISPLAY=:1
export XDG_RUNTIME_DIR="${session_dir}/runtime"
mkdir -p "${HOME}" "${XDG_RUNTIME_DIR}"
chmod 0700 "${HOME}" "${XDG_RUNTIME_DIR}"

echo "Starting browser-hosted GSAS-II"
echo "GSAS-II source revision: ${GSASII_REF:-unknown}"
echo "Input GPX: ${source_project}"
echo "Writable session GPX: ${GSASII_PROJECT_PATH}"
echo "Galaxy output GPX: ${GSASII_OUTPUT_PROJECT}"

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/gsasii.conf
