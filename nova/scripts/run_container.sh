#!/usr/bin/env bash
set -euo pipefail

trame_pid=""
nginx_pid=""

stop_children() {
  trap - EXIT INT TERM
  if [[ -n "${trame_pid}" ]]; then
    kill -TERM "${trame_pid}" 2>/dev/null || true
  fi
  if [[ -n "${nginx_pid}" ]]; then
    kill -TERM "${nginx_pid}" 2>/dev/null || true
  fi
  if [[ -n "${trame_pid}" ]]; then
    wait "${trame_pid}" 2>/dev/null || true
  fi
  if [[ -n "${nginx_pid}" ]]; then
    wait "${nginx_pid}" 2>/dev/null || true
  fi
}

terminate() {
  local status="$1"
  stop_children
  exit "${status}"
}

trap 'terminate 143' TERM
trap 'terminate 130' INT
trap stop_children EXIT

/usr/local/bin/run_trame.sh &
trame_pid=$!
/usr/local/bin/run_nginx.sh &
nginx_pid=$!

if wait -n "${trame_pid}" "${nginx_pid}"; then
  child_status=0
else
  child_status=$?
fi

stop_children
exit "${child_status}"
