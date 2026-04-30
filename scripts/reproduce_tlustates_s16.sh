#!/bin/bash

# Script to reproduce TLU States 16-shot results (SOTA: 76.5%)
# Agri-VLG: GAT + CutMix + RAL

PYTHON=python # Change to your python path

echo "🚀 Reproducing TLU States 16-shot Results..."

$PYTHON main.py \
    --dataset tlu_states \
    --dataset_root data/tlu_states \
    --data_source fewshot+retrieved \
    --method cutmix \
    --use_gat True \
    --alpha 0.5 \
    --attentive_threshold 0.85 \
    --attentive_name "c-name" \
    --shots 16 \
    --seed 1 \
    --epochs 50 \
    --prompt_name most_common_name \
    --prefix REPRODUCE_S16

echo "✅ Done! Check results in output/ folder."
