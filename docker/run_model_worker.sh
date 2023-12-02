#!/bin/bash

# Ensure the script fails if any command fails, this will prevent silent issues with conda activation
set -e

source /miniconda/etc/profile.d/conda.sh

conda activate llava

cd /code


python3 -m llava.serve.model_worker --host 0.0.0.0 --controller http://controller:10000 --port 40000 --worker http://model_worker:40000 --model-path SkunkworksAI/BakLLaVA-1
# python3 -m llava.serve.model_worker --host 0.0.0.0 --controller http://controller:10000 --port 40000 --worker http://model_worker:40000 --model-path liuhaotian/llava-v1.5-13b