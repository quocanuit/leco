#!/bin/bash

echo "Setting up environment for the project..."

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing specific versions of urllib3 and chardet..."
pip install urllib3==1.26.15 chardet==4.0.0

echo "Reinstalling PyTorch with CUDA support..."
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Setting environment variables..."
export PYTHONPATH="$PYTHONPATH:$PWD"
export USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"

echo "Environment setup complete!"