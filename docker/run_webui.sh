#!/bin/bash

# Ensure the script fails if any command fails, this will prevent silent issues with conda activation
set -e

source /miniconda/etc/profile.d/conda.sh

conda activate llava

cd /code
pip install loguru

python3 -m webui.server --controller http://controller:10000 --model-list-mode reload --port 11000