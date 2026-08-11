#!/usr/bin/env bash
set -euo pipefail

ep_path="${EP_PATH:-}"
if [[ -n "${ep_path}" && "${ep_path}" != /* ]]; then
  ep_path="/${ep_path}"
fi
ep_path="${ep_path%/}"
if [[ "${ep_path}" == "/" ]]; then
  ep_path=""
fi
if [[ -n "${ep_path}" && ! "${ep_path}" =~ ^(/[A-Za-z0-9._~-]+)+$ ]]; then
  exit 64
fi
# Keep the fallback root location unique during local/root-path launches.
# NOVA supplies a non-empty entry-point path in production.
export EP_PATH="${ep_path:-/__radar_pd_root__}"
runtime_dir="/tmp/radar-pd-nginx"
mkdir -p \
  "${runtime_dir}/client-body" \
  "${runtime_dir}/proxy" \
  "${runtime_dir}/fastcgi" \
  "${runtime_dir}/uwsgi" \
  "${runtime_dir}/scgi"

envsubst '${EP_PATH}' \
  < /etc/nginx/templates/radar-pd.conf.template \
  > /tmp/radar-pd-nginx.conf

exec nginx -c /tmp/radar-pd-nginx.conf -g 'daemon off;'
