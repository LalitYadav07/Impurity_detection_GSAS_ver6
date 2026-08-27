#!/usr/bin/env bash
set -euo pipefail

ep_path="${EP_PATH:-/gsasii}"
if [[ "${ep_path}" != /* ]]; then
    ep_path="/${ep_path}"
fi
ep_path="${ep_path%/}"
if [[ -z "${ep_path}" || ! "${ep_path}" =~ ^(/[A-Za-z0-9._~-]+)+$ ]]; then
    echo "EP_PATH is not a safe URL path: ${ep_path}" >&2
    exit 64
fi
export EP_PATH="${ep_path}"
export EP_WS_PATH="${EP_PATH#/}"
gui_version="${GSASII_GUI_VERSION:-unknown}"
if [[ ! "${gui_version}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    gui_version="unknown"
fi
export GUI_VERSION="${gui_version}"

runtime_dir=/tmp/gsasii-nginx
mkdir -p \
    "${runtime_dir}/client-body" \
    "${runtime_dir}/proxy" \
    "${runtime_dir}/fastcgi" \
    "${runtime_dir}/uwsgi" \
    "${runtime_dir}/scgi"

wait_for_port() {
    local host="$1"
    local port="$2"
    local service="$3"
    local attempt
    for attempt in $(seq 1 200); do
        if (exec 9<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
            exec 9>&-
            exec 9<&-
            return 0
        fi
        sleep 0.1
    done
    echo "Timed out waiting for ${service} on ${host}:${port}" >&2
    return 1
}

wait_for_port 127.0.0.1 5900 x11vnc
wait_for_port 127.0.0.1 6080 websockify

envsubst '${EP_PATH} ${EP_WS_PATH} ${GUI_VERSION}' \
    < /etc/nginx/gsasii.conf.template \
    > /tmp/gsasii-nginx.conf

exec nginx -e /dev/stderr -c /tmp/gsasii-nginx.conf -g 'daemon off;'
