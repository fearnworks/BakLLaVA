#!/bin/bash

cd /code/Grounded-Segment-Anything

if [ ! -f "/models/sam_vit_h_4b8939.pth" ]; then
    echo "/models/sam_vit_h_4b8939.pth does not exist."
    exit 1
fi

if [ ! -f "/models/groundingdino_swint_ogc.pth" ]; then
    echo "/models/groundingdino_swint_ogc.pth does not exist."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

cd /code
python serve/grounded_sam.py \
  --host 0.0.0.0 \
  --port 41000 \
  --worker-address "http://grounded_sam:41000" \
  --controller-address "http://controller:10000" \
  --model-path "/models/groundingdino_swint_ogc.pth" \
  --model-config "/code/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --sam-path "/models/sam_vit_h_4b8939.pth" \
  --device cuda \
  --limit-model-concurrency 5 \
  --stream-interval 2 \
  --grounding-dino-server grounding_dino \
  --sam-server sam
