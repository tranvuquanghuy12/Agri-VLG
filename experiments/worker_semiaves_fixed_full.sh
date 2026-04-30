#!/bin/bash
export TMPDIR="/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT/scratch/tmp_dir"
export WANDB_DIR="/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT/scratch/tmp_dir"
export WANDB_MODE="offline"
PYTHON="/opt/anaconda3/envs/swatai/bin/python"
DATASET="semi_aves"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/Semi-Aves/semi_aves"
PREFIX="Grid_T0.7_K5"
SHOTS=(4 8 16)
OUTPUT_ROOT="output/FIXED_BENCHMARK_v2/${DATASET}"

methods=("SWAT_Baseline" "AgriVLG_Cutmix" "Ours_Final" "Argi_FT_FewShot" "Argi_FT_Retrieved")

for SHOT in "${SHOTS[@]}"
do
    for method in "${methods[@]}"
    do
        echo "🚀 Semi-Aves FIXED: ${method} - ${SHOT} shots..."
        USE_GAT="True"
        if [ "$method" == "SWAT_Baseline" ]; then USE_GAT="False"; fi
        if [ "$method" == "AgriVLG_Cutmix" ] || [ "$method" == "SWAT_Baseline" ]; then METHOD_TYPE="cutmix"; SOURCE="fewshot+retrieved"
        elif [ "$method" == "Ours_Final" ]; then METHOD_TYPE="finetune"; SOURCE="fewshot+retrieved"
        elif [ "$method" == "Argi_FT_FewShot" ]; then METHOD_TYPE="finetune"; SOURCE="fewshot"
        elif [ "$method" == "Argi_FT_Retrieved" ]; then METHOD_TYPE="finetune"; SOURCE="retrieved"; fi

        $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" --folder "$OUTPUT_ROOT/SHOT_${SHOT}/$method" --data_source $SOURCE --method $METHOD_TYPE --use_gat $USE_GAT --shots $SHOT --epochs 20 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
    done
done
