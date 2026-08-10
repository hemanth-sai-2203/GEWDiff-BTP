#!/bin/bash

set -e

echo "=== GEWDiff BTP Kaggle Setup ==="

cd /kaggle/working

# Remove old copy if present
rm -rf GEWDiff-BTP

# Clone our baseline branch
git clone -b baseline https://github.com/hemanth-sai-2203/GEWDiff-BTP.git

cd GEWDiff-BTP/src/GEWDiff

echo "=== Repository ready ==="
pwd

echo "=== Installing required packages ==="
pip install -q accelerate diffusers einops tifffile scipy scikit-learn scikit-image

echo "=== Setup complete ==="