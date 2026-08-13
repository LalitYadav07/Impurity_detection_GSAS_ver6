#!/usr/bin/env bash
set -eu

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
export EP_PATH="${ep_path:-/__radar_pd_root__}"
runtime_dir="/tmp/radar-pd-nginx"
mkdir -p \
  "${runtime_dir}/client-body" \
  "${runtime_dir}/proxy" \
  "${runtime_dir}/fastcgi" \
  "${runtime_dir}/uwsgi" \
  "${runtime_dir}/scgi"

until wget -q -O /dev/null http://127.0.0.1:8080/; do
  echo '[radar-pd-nova] waiting for Trame on port 8080'
  sleep 0.2
done

envsubst '${EP_PATH}' < /etc/nginx/nginx.conf.template > /tmp/radar-pd-nginx.conf

exec nginx -c /tmp/radar-pd-nginx.conf -g 'daemon off;'
