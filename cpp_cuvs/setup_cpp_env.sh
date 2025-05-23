#!/bin/bash

# Script to set up the C++ development environment for cuVS.
#
# IMPORTANT:
# This script installs C++ development tools (compilers, CUDA toolkit components,
# cuVS/RAFT development libraries, and build tools) directly into your
# CURRENT ACTIVE Conda environment.
#
# In some workspaces (like certain "Studio" environments), creating new, separate
# Conda environments is restricted. This script assumes you are in your primary
# Conda environment (e.g., 'cloudspace') and want to add these C++ tools to it.
#
# Make sure your primary Conda environment is active before running this script.
#
# Mamba is used for faster package installation. If Mamba is not available,
# you can try replacing 'mamba' with 'conda' in the command below,
# though it might be significantly slower.

echo "Attempting to install C++ development tools for cuVS (nightly 25.06)..."
echo "This will modify your current active Conda environment."
echo "Targeting: libcuvs-dev=25.06, libraft-dev=25.06, cuda-nvcc=12.4"
echo ""

conda install -c rapidsai -c conda-forge -c nvidia libcuvs cuda-version=12.8

echo ""
echo "Installation command finished."
echo "If successful, the C++ tools are now part of your active Conda environment."
echo "You can try verifying with commands like 'which nvcc', 'which g++', 'cmake --version'."
echo "Ensure your system meets prerequisites for CUDA and RAPIDS (NVIDIA driver, GPU)." 