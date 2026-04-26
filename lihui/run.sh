#!/usr/bin/env bash
# Quick-start wrapper. Forwards all args to run_one.py.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python lihui/experiments/run_one.py "$@"
