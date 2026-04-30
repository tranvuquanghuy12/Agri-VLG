#!/bin/bash
export TMPDIR="/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT/scratch/tmp_dir"
export WANDB_DIR="/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT/scratch/tmp_dir"
export WANDB_MODE="offline" # Tắt luôn wandb online để tiết kiệm tài nguyên và né lỗi

PYTHON="/opt/anaconda3/envs/swatai/bin/python"
DATASET="eurosat_ms"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/EuroSAT/eurosat_ms"
PREFIX="Grid_T0.85_K10"
SHOT=4
OUTPUT_ROOT="output/FIXED_BENCHMARK_v2/${DATASET}/SHOT_${SHOT}"

methods=("SWAT_Baseline" "AgriVLG_Cutmix" "Ours_Final")

for method in "${methods[@]}"
do
    echo "🚀 Chạy LẠI SIÊU TỐC: ${method} - 4 shots (Redirected TMP)..."
    
    USE_GAT="False"
    METHOD_TYPE="cutmix"
    SOURCE="fewshot+retrieved"
    
    if [ "$method" == "AgriVLG_Cutmix" ]; then
        USE_GAT="True"
        METHOD_TYPE="cutmix"
    elif [ "$method" == "Ours_Final" ]; then
        USE_GAT="True"
        METHOD_TYPE="finetune"
    fi

    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/$method" \
        --data_source $SOURCE --method $METHOD_TYPE --use_gat $USE_GAT \
        --shots $SHOT --epochs 20 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
done
