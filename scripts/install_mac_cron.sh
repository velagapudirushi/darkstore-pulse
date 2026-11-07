#!/usr/bin/env bash
# Installs a silent hourly cron job on macOS (8:00 AM - 11:00 PM local time)
# Zero login required; runs headless in the background while your Mac is awake.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$(command -v python3)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

CRON_CMD="0 8-23 * * * cd \"${REPO_DIR}\" && \"${PYTHON_BIN}\" scripts/run_hourly_collector.py >> \"${LOG_DIR}/collector.log\" 2>&1"

if crontab -l 2>/dev/null | grep -Fq "scripts/run_hourly_collector.py"; then
    echo "Hourly DarkStore Pulse cron job is already installed."
else
    (crontab -l 2>/dev/null || true; echo "${CRON_CMD}") | crontab -
    echo "Installed silent hourly cron job (08:00-23:00):"
    echo "  ${CRON_CMD}"
fi
