#!/bin/bash

PYTHON="/opt/anaconda3/envs/swatai/bin/python"
DATASET="eurosat_ms"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/EuroSAT"
PREFIX="Grid_T0.85_K10"
OUTPUT_DIR="output/EUROSAT_FULL_TABLE"
SHOTS=(4 8 16)

echo "🚀 KHỞI ĐỘNG TỔNG TIẾN CÔNG EUROSAT (4-8-16 SHOTS)..."

for SHOT in "${SHOTS[@]}"; do
    echo "----------------------------------------------------------"
    echo "📍 ĐANG CHẠY CHO MỨC: $SHOT SHOTS"
    echo "----------------------------------------------------------"
    
    # 1. FT-FewShot (Org)
    echo "🔹 1. Chạy FT-FewShot (Org)..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --folder "$OUTPUT_DIR/SHOT_${SHOT}_FT_FewShot" \
        --data_source fewshot --method finetune --use_gat False \
        --shots $SHOT --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
        
    # 2. FLYP (Org)
    echo "🔹 2. Chạy FLYP (Org)..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --folder "$OUTPUT_DIR/SHOT_${SHOT}_FLYP" \
        --data_source fewshot --method FLYP --use_gat False \
        --shots $SHOT --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

    # 3. SWAT (Baseline)
    echo "🔹 3. Chạy SWAT (Baseline)..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
        --folder "$OUTPUT_DIR/SHOT_${SHOT}_SWAT" \
        --data_source fewshot+retrieved --method cutmix --use_gat False \
        --shots $SHOT --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
        
    # 4. Agri-VLG (Turbo)
    echo "🔹 4. Chạy Agri-VLG Turbo..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
        --folder "$OUTPUT_DIR/SHOT_${SHOT}_AgriVLG" \
        --data_source fewshot+retrieved --method cutmix --use_gat True \
        --shots $SHOT --epochs 100 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
done

echo "🎉 HOÀN THÀNH TỔNG BIỂU EUROSAT!"
