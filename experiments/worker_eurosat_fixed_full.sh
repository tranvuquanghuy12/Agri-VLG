#!/bin/bash
export TMPDIR="/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT/scratch/tmp_dir"
export WANDB_DIR="/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT/scratch/tmp_dir"
export WANDB_MODE="offline"

PYTHON="/opt/anaconda3/envs/swatai/bin/python"
DATASET="eurosat_ms"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/EuroSAT/eurosat_ms"
PREFIX="Grid_T0.85_K10"
OUTPUT_ROOT="output/FIXED_BENCHMARK_v2/${DATASET}"

# --- GIAI ĐOẠN 1: Chạy nốt 4-SHOT (2 phương pháp còn thiếu) ---
echo "🚀 Chạy nốt 4-Shot: FT Few-shot & Retrieved..."
for method in "Argi_FT_FewShot" "Argi_FT_Retrieved"
do
    METHOD_TYPE="finetune"
    USE_GAT="True"
    if [ "$method" == "Argi_FT_FewShot" ]; then SOURCE="fewshot"; else SOURCE="retrieved"; fi

    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/SHOT_4/$method" \
        --data_source $SOURCE --method $METHOD_TYPE --use_gat $USE_GAT \
        --shots 4 --epochs 20 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
done

# --- GIAI ĐOẠN 2: Chạy FULL BỘ 5 cho 8-SHOT và 16-SHOT ---
SHOTS=(8 16)
methods=("SWAT_Baseline" "AgriVLG_Cutmix" "Ours_Final" "Argi_FT_FewShot" "Argi_FT_Retrieved")

for SHOT in "${SHOTS[@]}"
do
    for method in "${methods[@]}"
    do
        echo "🚀 Chạy FIXED FULL: ${method} - ${SHOT} shots..."
        
        USE_GAT="True"
        if [ "$method" == "SWAT_Baseline" ]; then USE_GAT="False"; fi

        if [ "$method" == "AgriVLG_Cutmix" ] || [ "$method" == "SWAT_Baseline" ]; then
            METHOD_TYPE="cutmix"
            SOURCE="fewshot+retrieved"
        elif [ "$method" == "Ours_Final" ]; then
            METHOD_TYPE="finetune"
            SOURCE="fewshot+retrieved"
        elif [ "$method" == "Argi_FT_FewShot" ]; then
            METHOD_TYPE="finetune"
            SOURCE="fewshot"
        elif [ "$method" == "Argi_FT_Retrieved" ]; then
            METHOD_TYPE="finetune"
            SOURCE="retrieved"
        fi

        $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
            --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
            --folder "$OUTPUT_ROOT/SHOT_${SHOT}/$method" \
            --data_source $SOURCE --method $METHOD_TYPE --use_gat $USE_GAT \
            --shots $SHOT --epochs 20 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
    done
done
