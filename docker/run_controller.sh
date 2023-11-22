#!/bin/bash

# Ensure the script fails if any command fails, this will prevent silent issues with conda activation
set -e

source /miniconda/etc/profile.d/conda.sh

conda activate llava

cd /code

python3 -m llava.serve.controller --host 0.0.0.0 --port 10000
