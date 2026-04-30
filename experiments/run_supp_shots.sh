#!/bin/bash
# ---------------------------------------------------------
# Supplementary Run: 4-shot and 8-shot for TLU-States
# To complete the comparison table for the user
# ---------------------------------------------------------

DATASET="tlu_states"
EPOCHS=50
MODEL_CFG="vitb32_openclip_laion400m"
RETR_SPLIT="T2T_Filtered"

for SHOTS in 4 8
do
    echo "--- RUNNING $SHOTS-SHOT EXPERIMENTS ---"
    
    # 1. SWAT (Original)
    echo "🚀 Running SWAT (Original) - $SHOTS shots"
    conda run -n swatai python main.py --dataset $DATASET --shots $SHOTS --method cutmix --data_source fewshot+retrieved \
                   --use_gat False --retrieval_split ${RETR_SPLIT}.txt --epochs $EPOCHS \
                   --model_cfg $MODEL_CFG --folder output/150_SUPP_SWAT_S${SHOTS}
    
    # 2. Agri-VLG (Proposed)
    echo "🚀 Running Agri-VLG (Proposed) - $SHOTS shots"
    conda run -n swatai python main.py --dataset $DATASET --shots $SHOTS --method cutmix --data_source fewshot+retrieved \
                   --use_gat True --retrieval_split ${RETR_SPLIT}.txt --epochs $EPOCHS \
                   --model_cfg $MODEL_CFG --folder output/151_SUPP_AgriVLG_S${SHOTS}
done

echo "✅ SUPPLEMENTARY EXPERIMENTS COMPLETED."
