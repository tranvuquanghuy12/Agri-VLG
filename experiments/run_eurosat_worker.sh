#!/bin/bash
# Worker Script: Chạy 1 mốc Shot duy nhất cho 5 phương pháp
# Cách dùng: bash run_eurosat_worker.sh <SHOT> <GPU_ID>
# Ví dụ: bash run_eurosat_worker.sh 4 0  (Chạy mốc 4 shots trên GPU 0)

SHOT=$1
GPU_ID=${2:-0} # Mặc định là GPU 0 nếu không truyền

if [ -z "$SHOT" ]; then
    echo "❌ Vui lòng truyền vào số SHOT (ví dụ: 4, 8, 16)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
PYTHON="/opt/anaconda3/envs/swatai/bin/python"

DATASET="eurosat_ms"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/EuroSAT"
PREFIX="Grid_T0.85_K10"
OUTPUT_ROOT="output/EUROSAT_FULL_TABLE/SHOT_$SHOT"

mkdir -p $OUTPUT_ROOT

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 ĐANG XỬ LÝ EUROSAT - MỐC $SHOT SHOTS TRÊN GPU $GPU_ID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. FLYP (Org)
echo "🔹 1. Chạy FLYP (Org)..."
$PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
    --retrieved_path ./data/ --retrieval_split "none" \
    --folder "$OUTPUT_ROOT/FLYP" \
    --data_source fewshot --method FLYP --use_gat False \
    --shots $SHOT --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

# 2. SWAT (Baseline)
echo "🔹 2. Chạy SWAT (Baseline)..."
$PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
    --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
    --folder "$OUTPUT_ROOT/SWAT_Baseline" \
    --data_source fewshot+retrieved --method cutmix --use_gat False \
    --shots $SHOT --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

# 3. Argi - FT on Few-shot
echo "🔹 3. Chạy Argi (FT on Few-shot)..."
$PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
    --retrieved_path ./data/ --retrieval_split "none" \
    --folder "$OUTPUT_ROOT/Argi_FT_FewShot" \
    --data_source fewshot --method finetune --use_gat True \
    --shots $SHOT --epochs 50 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

# 4. Argi - FT on Retrieved
echo "🔹 4. Chạy Argi (FT on Retrieved)..."
$PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
    --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
    --folder "$OUTPUT_ROOT/Argi_FT_Retrieved" \
    --data_source retrieved --method finetune --use_gat True \
    --shots $SHOT --epochs 50 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

# 5. Agri-VLG (Full)
echo "🔹 5. Chạy Agri-VLG (Full)..."
$PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
    --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
    --folder "$OUTPUT_ROOT/AgriVLG" \
    --data_source fewshot+retrieved --method cutmix --use_gat True \
    --shots $SHOT --epochs 50 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

echo "🎉 HOÀN THÀNH MỐC $SHOT SHOTS CỦA EUROSAT!"
