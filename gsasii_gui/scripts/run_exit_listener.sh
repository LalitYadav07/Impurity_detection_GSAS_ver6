#!/usr/bin/env bash
set -euo pipefail

supervisor_socket="unix:///tmp/gsasii-supervisor.sock"

while true; do
    printf 'READY\n'
    IFS= read -r header || exit 0
    if [[ ! "${header}" =~ (^|[[:space:]])len:([0-9]+)($|[[:space:]]) ]]; then
        printf 'RESULT 4\nFAIL'
        continue
    fi

    payload_length="${BASH_REMATCH[2]}"
    payload=""
    if (( payload_length > 0 )); then
        IFS= read -r -N "${payload_length}" payload || true
    fi

    event_name=""
    process_name=""
    if [[ "${header}" =~ (^|[[:space:]])eventname:([^[:space:]]+) ]]; then
        event_name="${BASH_REMATCH[2]}"
    fi
    if [[ "${payload}" =~ (^|[[:space:]])processname:([^[:space:]]+) ]]; then
        process_name="${BASH_REMATCH[2]}"
    fi

    printf 'RESULT 2\nOK'
    if [[ "${event_name}" == "PROCESS_STATE_FATAL" \
          || ( "${event_name}" == "PROCESS_STATE_EXITED" \
               && "${process_name}" == "gsasii" ) ]]; then
        echo "Stopping the GSAS-II session after ${event_name} for ${process_name}" >&2
        supervisorctl -s "${supervisor_socket}" shutdown >/dev/null 2>&1 || true
        exit 0
    fi
done
