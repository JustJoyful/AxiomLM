#!/usr/bin/env bash
# AxiomLM -- Environment Setup
set -euo pipefail

echo "[1/4] Installing 'uv' (Fast Python Manager) to handle ML versions..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

VENV_DIR=".venv"
echo "[2/4] Wiping broken venv and creating a pure Python 3.11 venv..."
rm -rf "$VENV_DIR"
uv venv --python 3.11 "$VENV_DIR"

echo "[3/4] Installing dependencies very fast using uv pip..."
# 1. Install torch with CUDA explicitly first!
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 2. Install everything else
uv pip install -r requirements.txt
# 3. Optional: install MinerU (uncomment if you want to test warped scans later)
# uv pip install mineru

echo "[4/4] Pulling Gemma via Ollama ..."
ollama pull gemma

echo "Setup complete! Activate with: source $VENV_DIR/bin/activate"
