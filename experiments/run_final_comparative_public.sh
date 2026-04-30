#!/bin/bash

PYTHON="/opt/anaconda3/envs/swatai/bin/python"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset"
OUTPUT_DIR="output/FINAL_COMPARISON"

# Cấu hình các tập dữ liệu
# Dataset Name | Prefix | GAT LR
configs=(
    "eurosat_ms|Grid_T0.85_K10|0.001"
    "flowers102|Grid_T0.85_K5|0.001"
    "semi-aves|Grid_T0.85_K3|0.001"
)

for config in "${configs[@]}"; do
    IFS="|" read -r DATASET PREFIX GAT_LR <<< "$config"
    
    echo "=========================================================="
    echo "🚀 RUNNING COMPARISON FOR: $DATASET ($PREFIX)"
    echo "=========================================================="
    
    # 1. Chạy SWAT Baseline
    echo "🥊 Running SWAT Baseline..."
    $PYTHON main.py \
        --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
        --folder "$OUTPUT_DIR/${DATASET}_SWAT" \
        --data_source fewshot+retrieved --method cutmix --use_gat False \
        --shots 16 --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
    
    # 2. Chạy Agri-VLG Turbo
    echo "🔥 Running Agri-VLG Turbo..."
    $PYTHON main.py \
        --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
        --folder "$OUTPUT_DIR/${DATASET}_AgriVLG_Turbo" \
        --data_source fewshot+retrieved --method cutmix --use_gat True \
        --shots 16 --epochs 100 --lr_gat $GAT_LR --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

done

echo "🎉 ALL COMPARATIVE EXPERIMENTS COMPLETED!"
