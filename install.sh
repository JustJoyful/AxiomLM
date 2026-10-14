#!/usr/bin/env bash
# Both entry points use the same dependency/model/device contract.
set -euo pipefail
exec bash "$(dirname "$0")/scripts/setup_env.sh" "$@"
