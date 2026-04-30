#!/bin/bash

# Configuration
PYTHON=/opt/anaconda3/envs/swatai/bin/python
DATASET="tlu_states"
DATASET_ROOT="data/tlu_states"
PROMPT="most_common_name"
EPOCHS=20
SEED=1
OUTPUT_ROOT="output/TLU_STATES_BASELINES_FINAL"

mkdir -p $OUTPUT_ROOT

SHOTS=(4 8 16)

for S in "${SHOTS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 CHẠY BASELINES TLU STATES: $S SHOTS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. Fine-tune (Few-shot only)
    echo "🔹 Chạy Fine-tune (Few-shot) - $S shots..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --folder $OUTPUT_ROOT/SHOT_$S/FineTune \
        --data_source fewshot --method finetune --use_gat False \
        --shots $S --epochs $EPOCHS --prompt_name $PROMPT --log_mode file

    # 2. Linear Probe (Few-shot only)
    echo "🔹 Chạy Linear Probe (Few-shot) - $S shots..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --folder $OUTPUT_ROOT/SHOT_$S/LinearProbe \
        --data_source fewshot --method finetune --use_gat False \
        --shots $S --epochs 0 --prompt_name $PROMPT --log_mode file

    # 3. FLYP
    echo "🔹 Chạy FLYP - $S shots..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --folder $OUTPUT_ROOT/SHOT_$S/FLYP \
        --data_source fewshot --method FLYP --use_gat False \
        --shots $S --epochs $EPOCHS --prompt_name $PROMPT --log_mode file

    # 4. CMLP (CrossModal Linear Probe)
    echo "🔹 Chạy CMLP - $S shots..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --folder $OUTPUT_ROOT/SHOT_$S/CMLP \
        --data_source fewshot --method CMLP --use_gat False \
        --shots $S --epochs $EPOCHS --prompt_name $PROMPT --log_mode file
done

echo "🎉 TẤT CẢ BASELINES ĐÃ HOÀN TẤT!"
