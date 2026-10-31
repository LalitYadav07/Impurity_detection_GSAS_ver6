#!/usr/bin/env bash
set -euo pipefail
exec /opt/conda/bin/python /opt/gsasii-gui/project_tracking.py sync --watch
