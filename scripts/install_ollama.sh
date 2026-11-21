#!/usr/bin/env bash
set -euo pipefail

if command -v ollama >/dev/null 2>&1; then
  ollama --version
  exit 0
fi

echo "Installing Ollama from the official install script."
echo "This requires sudo; review https://ollama.com/install.sh before continuing."
curl -fsSL https://ollama.com/install.sh | sh
