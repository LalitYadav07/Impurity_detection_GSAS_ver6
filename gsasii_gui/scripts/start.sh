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
export DISPLAY=:1
export XDG_RUNTIME_DIR="${session_dir}/runtime"
mkdir -p "${HOME}" "${XDG_RUNTIME_DIR}"
chmod 0700 "${HOME}" "${XDG_RUNTIME_DIR}"

if ! id -un >/dev/null 2>&1; then
    runtime_uid="$(id -u)"
    runtime_gid="$(id -g)"
    cp /etc/passwd "${session_dir}/passwd"
    cp /etc/group "${session_dir}/group"
    printf 'gsasii-runtime:x:%s:%s:GSAS-II Runtime:%s:/bin/bash\n' \
        "${runtime_uid}" "${runtime_gid}" "${HOME}" >> "${session_dir}/passwd"
    printf 'gsasii-runtime:x:%s:\n' "${runtime_gid}" >> "${session_dir}/group"
    export NSS_WRAPPER_PASSWD="${session_dir}/passwd"
    export NSS_WRAPPER_GROUP="${session_dir}/group"
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnss_wrapper.so${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

export USER="$(id -un)"
export LOGNAME="${USER}"

echo "Starting browser-hosted GSAS-II"
echo "GSAS-II source revision: ${GSASII_REF:-unknown}"
echo "Input GPX: ${source_project}"
echo "Writable session GPX: ${GSASII_PROJECT_PATH}"
echo "Galaxy output GPX: ${GSASII_OUTPUT_PROJECT}"

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/gsasii.conf
