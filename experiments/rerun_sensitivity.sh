#!/bin/bash
# ---------------------------------------------------------
# Rerunning Sensitivity Experiments after fixing FileNotFoundError
# ---------------------------------------------------------

DATASET="tlu_states"
SHOTS=16
EPOCHS=50
MODEL_CFG="vitb32_openclip_laion400m"

# 1. Sensitivity to Tau
for TAU in 0.5 0.7 0.9; do
    PREFIX="T2T_tau${TAU}"
    echo "🔍 Sampling for τ = $TAU"
    conda run -n swatai python retrieval/sample_retrieval.py --dataset $DATASET --sampling_threshold $TAU \
                                        --num_samples 10 --sampling_method T2T-rank \
                                        --prefix $PREFIX --prompt_name alternates
    
    echo "🚀 Training for τ = $TAU"
    conda run -n swatai python main.py --dataset $DATASET --shots $SHOTS --method cutmix --data_source fewshot+retrieved \
                   --use_gat True --retrieval_split ${PREFIX}.txt --epochs $EPOCHS \
                   --model_cfg $MODEL_CFG --folder output/210_REFIX_Sens_Tau_${TAU}
done

# 2. Sensitivity to K
for K in 1 3 5; do
    PREFIX="T2T_K${K}"
    echo "🔍 Sampling for K = $K"
    conda run -n swatai python retrieval/sample_retrieval.py --dataset $DATASET --sampling_threshold 0.85 \
                                        --num_samples $K --sampling_method T2T-rank \
                                        --prefix $PREFIX --prompt_name alternates
    
    echo "🚀 Training for K = $K"
    conda run -n swatai python main.py --dataset $DATASET --shots $SHOTS --method cutmix --data_source fewshot+retrieved \
                   --use_gat True --retrieval_split ${PREFIX}.txt --epochs $EPOCHS \
                   --model_cfg $MODEL_CFG --folder output/220_REFIX_Sens_K_${K}
done

echo "✅ RERUN COMPLETED."
