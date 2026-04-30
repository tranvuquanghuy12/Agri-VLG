#!/bin/bash

# Configuration
PYTHON=/opt/anaconda3/envs/swatai/bin/python
DATASET="semi_aves"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/semi-aves"
PROMPT="most_common_name"
RETRIEVAL_FILE="Grid_T0.9_K10.txt"
ALPHA=0.5
TAU=0.9
K=10
EPOCHS=20
SEED=1
OUTPUT_ROOT="output/SEMI_AVES_FULL_TABLE_T0.9_K10"

mkdir -p $OUTPUT_ROOT

SHOTS=(4 8 16)

for S in "${SHOTS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 BẮT ĐẦU CHẠY SEMI-AVES: $S SHOTS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. SWAT (Cutmix + RAL)
    echo "🔹 Chạy SWAT (Baseline) - $S shots..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split $RETRIEVAL_FILE \
        --folder $OUTPUT_ROOT/SHOT_$S/SWAT \
        --data_source fewshot+retrieved --method cutmix --use_gat False \
        --shots $S --epochs $EPOCHS --prompt_name $PROMPT --log_mode file

    # 2. Agri-VLG Full (Cutmix + RAL + GAT)
    echo "🔹 Chạy Agri-VLG (Full) - $S shots..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split $RETRIEVAL_FILE \
        --folder $OUTPUT_ROOT/SHOT_$S/AgriVLG_Full \
        --data_source fewshot+retrieved --method cutmix --use_gat True \
        --alpha $ALPHA --attentive_threshold $TAU --attentive_name "c-name" \
        --shots $S --epochs $EPOCHS --prompt_name $PROMPT --log_mode file

    # 3. Agri-VLG No Cutmix (RAL + GAT)
    echo "🔹 Chạy Agri-VLG (No Cutmix) - $S shots..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split $RETRIEVAL_FILE \
        --folder $OUTPUT_ROOT/SHOT_$S/AgriVLG_NoCutmix \
        --data_source fewshot+retrieved --method finetune --use_gat True \
        --alpha $ALPHA --attentive_threshold $TAU --attentive_name "c-name" \
        --shots $S --epochs $EPOCHS --prompt_name $PROMPT --log_mode file
done

echo "🎉 TẤT CẢ ĐÃ HOÀN TẤT!"
