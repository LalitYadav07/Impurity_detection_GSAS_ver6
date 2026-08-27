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

envsubst '${EP_PATH} ${EP_WS_PATH} ${GUI_VERSION}' \
    < /etc/nginx/gsasii.conf.template \
    > /tmp/gsasii-nginx.conf

exec nginx -e /dev/stderr -c /tmp/gsasii-nginx.conf -g 'daemon off;'
