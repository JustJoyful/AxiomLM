#!/usr/bin/env bash
# Shared, non-destructive setup; inspect --dry-run before installing.
set -euo pipefail
cd "$(dirname "$0")/.."
SETUP_DEVICE=cuda
SETUP_VENV=.venv
SETUP_PULL=1
SETUP_DRY=0
while (($#)); do
  case "$1" in
    --cpu) SETUP_DEVICE=cpu; shift ;;
    --venv) SETUP_VENV="${2:?--venv needs a path}"; shift 2 ;;
    --no-model) SETUP_PULL=0; shift ;;
    --dry-run) SETUP_DRY=1; shift ;;
    --help) echo 'Usage: bash scripts/setup_env.sh [--cpu] [--venv PATH] [--no-model] [--dry-run]'; exit 0 ;;
    *) echo "Unknown setup option: $1" >&2; exit 2 ;;
  esac
done
command -v uv >/dev/null 2>&1 || {
  echo 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 127
}
SETUP_PROFILE=cuda
if [[ "$SETUP_DEVICE" == cpu ]]; then SETUP_PROFILE=cpu; fi
SETUP_MODEL="$(python3 -c 'from src.config import OLLAMA_MODEL; print(OLLAMA_MODEL)')"
SETUP_EMBED_MODEL="$(python3 -c 'from src.config import EMBED_MODEL; print(EMBED_MODEL)')"
SETUP_EMBED_REVISION="$(python3 -c 'from src.config import EMBED_REVISION; print(EMBED_REVISION)')"
run_setup() {
  printf '%q ' "$@"; printf '\n'
  if (( ! SETUP_DRY )); then "$@"; fi
}
run_setup env UV_PROJECT_ENVIRONMENT="$SETUP_VENV" uv sync --python 3.11 --extra "$SETUP_PROFILE" --extra dev --locked --inexact
if (( SETUP_PULL )); then
  run_setup env UV_PROJECT_ENVIRONMENT="$SETUP_VENV" uv run --no-sync --extra "$SETUP_PROFILE" python -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))' "$SETUP_EMBED_MODEL" "$SETUP_EMBED_REVISION"
  run_setup ollama pull "$SETUP_MODEL"
fi
echo "Setup device: $SETUP_DEVICE. Activate with: source $SETUP_VENV/bin/activate"
