#!/bin/bash

# setup conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/etc/profile.d/conda.sh

# create conda env
read -rp "Enter environment name: " ENV_NAME
read -rp "Enter python version (default 3.9): " PYTHON_VERSION
if [ -z "$PYTHON_VERSION" ]; then
  PYTHON_VERSION="3.9"
fi
conda create -yn "$ENV_NAME" python="$PYTHON_VERSION"
conda activate "$ENV_NAME"

# replace placeholder env with $ENV_NAME in scripts/train.sh
NEW_CONDA_LINE="source \$CONDA_BASE/bin/activate $ENV_NAME"
sed -i '' "s,.*bin/activate.*,$NEW_CONDA_LINE,g" scripts/train.sh

# install torch
read -rp "Enter cuda version (e.g. '11.3', default no cuda support): " CUDA_VERSION
if [ -z "$CUDA_VERSION" ]; then
    conda install -y pytorch cpuonly -c pytorch
else
    conda install -y pytorch cudatoolkit="$CUDA_VERSION" -c pytorch
fi

# install python requirements
pip install -r requirements.txt
