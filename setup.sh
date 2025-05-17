#!/bin/bash

echo "Setting up environment for the project..."

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing specific versions of urllib3 and chardet..."
pip install urllib3==1.26.15 chardet==4.0.0

echo "Reinstalling PyTorch with CUDA support..."
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Environment setup complete!"