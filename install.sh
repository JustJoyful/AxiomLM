#!/usr/bin/env bash
set -e

# AxiomLM Installer
# Robust installer for the AxiomLM study assistant

COLOR_CYAN='\033[0;36m'
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_RESET='\033[0m'

echo -e "${COLOR_CYAN}Installing AxiomLM...${COLOR_RESET}"

# 1. Dependency Checks
echo -e "\n${COLOR_CYAN}Checking dependencies...${COLOR_RESET}"

# Python check
if ! command -v python3 &>/dev/null; then
    echo -e "${COLOR_RED}Error: Python 3 is not installed.${COLOR_RESET}"
    exit 1
fi

# Python version check (3.11+)
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)' &>/dev/null; then
    echo -e "${COLOR_RED}Error: Python 3.11+ is required.${COLOR_RESET}"
    exit 1
fi
echo -e "${COLOR_GREEN}Python 3.11+ found.${COLOR_RESET}"

# Ollama check
if ! command -v ollama &>/dev/null; then
    echo -e "${COLOR_RED}Warning: Ollama not found.${COLOR_RESET}"
    echo "Please install Ollama from https://ollama.ai to use AxiomLM."
else
    echo -e "${COLOR_GREEN}Ollama found.${COLOR_RESET}"
fi

# 2. Environment Setup
echo -e "\n${COLOR_CYAN}Setting up isolated environment...${COLOR_RESET}"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv || (echo "Error: venv creation failed. Please install python3-venv" && exit 1)
fi
VENV_PYTHON="./.venv/bin/python"

# Ensure pip is available and updated
if ! $VENV_PYTHON -m pip --version &>/dev/null; then
    echo "Installing pip into venv..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | $VENV_PYTHON
fi
$VENV_PYTHON -m pip install --upgrade pip

# 3. Install Requirements
echo -e "\n${COLOR_CYAN}Installing core dependencies from requirements.txt...${COLOR_RESET}"

# CRITICAL: We force-purge any existing broken torch/cuda libs first
echo "Cleaning environment for stable install..."
$VENV_PYTHON -m pip uninstall -y torch torchvision torchaudio nvidia-nccl-cu12 nvidia-nccl-cu13 &>/dev/null || true

# Defaulting to CPU for maximum compatibility as a 'fail-safe'.
# CUDA versions often have complex library conflicts (e.g. NCCL symbol errors).
echo "Installing stable CPU-optimized Torch..."
$VENV_PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing remaining requirements..."
$VENV_PYTHON -m pip install -r requirements.txt

# 4. Install Package
echo -e "\n${COLOR_CYAN}Installing AxiomLM in editable mode...${COLOR_RESET}"
$VENV_PYTHON -m pip install -e ".[dev]"

# 5. Path Integration
echo -e "\n${COLOR_CYAN}Setting up CLI access...${COLOR_RESET}"

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

INSTALL_PATH=$(pwd)
WRAPPER_PATH="$BIN_DIR/axiomlm"

echo "Creating wrapper script at $WRAPPER_PATH..."
cat << EOF > "$WRAPPER_PATH"
#!/bin/bash
cd "$INSTALL_PATH"
source .venv/bin/activate
axiomlm "\$@"
EOF

chmod +x "$WRAPPER_PATH"

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo -e "${COLOR_RED}Warning: $BIN_DIR is not in your PATH.${COLOR_RESET}"
    echo "Please add this to your .bashrc or .zshrc:"
    echo -e "  export PATH=\"\$PATH:$BIN_DIR\""
else
    echo -e "${COLOR_GREEN}CLI ready at $WRAPPER_PATH${COLOR_RESET}"
fi

echo -e "\n${COLOR_GREEN}AxiomLM installed successfully!${COLOR_RESET}"
echo -e "Run it with: ${COLOR_CYAN}axiomlm${COLOR_RESET}"
