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
# Keep the fallback root location unique during local/root-path launches.
# NOVA supplies a non-empty entry-point path in production.
export EP_PATH="${ep_path:-/__radar_pd_root__}"

envsubst '${EP_PATH}' \
  < /etc/nginx/templates/radar-pd.conf.template \
  > /etc/nginx/conf.d/radar-pd.conf

exec nginx -g 'daemon off;'
