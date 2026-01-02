#!/bin/bash
set -e

echo "=== Setting up development environment ==="

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check for GPU availability
echo "=== Checking GPU availability ==="
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (GPU info unavailable)"
    GPU_AVAILABLE=true
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    echo "No NVIDIA GPU detected - PyTorch will use CPU"
    GPU_AVAILABLE=false
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# Create virtual environment with uv
echo "=== Creating virtual environment with uv ==="
cd /workspace
uv venv --python 3.12

# Install base project dependencies
echo "=== Installing project dependencies ==="
uv sync

# Install PyTorch with appropriate backend
echo "=== Installing PyTorch from $PYTORCH_INDEX ==="
uv pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

# Install dev dependencies
echo "=== Installing dev dependencies ==="
uv sync --group dev

# Verify PyTorch installation
echo "=== Verifying PyTorch installation ==="
.venv/bin/python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Running in CPU mode')
"

echo "=== Development environment ready! ==="
