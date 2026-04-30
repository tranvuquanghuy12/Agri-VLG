#!/bin/bash
# ---------------------------------------------------------
# MASTER SCRIPT: Public Datasets Comparison (SWAT vs Agri-VLG)
# ---------------------------------------------------------

MODEL_CFG="vitb32_openclip_laion400m"
EPOCHS=20 # Public datasets usually converge faster

# Array of datasets and their retrieval splits
# format: "dataset_name|split_name"
DATASETS=("flowers102|T2T_Filtered_Flowers102.txt" "eurosat_ms|T2T_Filtered.txt" "semi-aves|T2T_Filtered_SemiAves.txt")

for DS_INFO in "${DATASETS[@]}"
do
    IFS='|' read -r DS SPLIT <<< "$DS_INFO"
    echo "========================================"
    echo "📂 DATASET: $DS"
    echo "========================================"
    
    for SHOTS in 4 8 16
    do
        echo "--- RUNNING $SHOTS-SHOT ---"
        
        # 1. SWAT (Original)
        echo "🚀 [SWAT] $DS - $SHOTS shots"
        conda run -n swatai python main.py --dataset $DS --shots $SHOTS --method cutmix --data_source fewshot+retrieved \
                       --use_gat False --retrieval_split $SPLIT --epochs $EPOCHS \
                       --model_cfg $MODEL_CFG --folder output/300_PUBLIC_${DS}_SWAT_S${SHOTS}
        
        # 2. Agri-VLG (Proposed)
        echo "🚀 [Agri-VLG] $DS - $SHOTS shots"
        conda run -n swatai python main.py --dataset $DS --shots $SHOTS --method cutmix --data_source fewshot+retrieved \
                       --use_gat True --retrieval_split $SPLIT --epochs $EPOCHS \
                       --model_cfg $MODEL_CFG --folder output/301_PUBLIC_${DS}_AgriVLG_S${SHOTS}
    done
done

echo "✅ ALL PUBLIC DATASET EXPERIMENTS COMPLETED."
