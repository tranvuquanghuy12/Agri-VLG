#!/bin/bash

# Configuration
PYTHON=/opt/anaconda3/envs/swatai/bin/python
DATASET="tlu_states"
METHOD="finetune"
DATA_SOURCE="fewshot+retrieved"
PROMPT="most_common_name"
RETRIEVAL_SPLIT="T2T_Filtered.txt"
ALPHA=0.5
TAU=0.85
K=10
EPOCHS=50
SEED=1

echo "🚀 Bắt đầu chạy TLU States No Cutmix 4-shot..."
$PYTHON main.py \
    --dataset $DATASET \
    --method $METHOD \
    --data_source $DATA_SOURCE \
    --prompt_name $PROMPT \
    --retrieval_split $RETRIEVAL_SPLIT \
    --alpha $ALPHA \
    --attentive_threshold $TAU \
    --attentive_name "c-name" \
    --shots 4 \
    --seed $SEED \
    --epochs $EPOCHS \
    --prefix "NoCutmix_FINAL_S4"

echo "🚀 Bắt đầu chạy TLU States No Cutmix 8-shot..."
$PYTHON main.py \
    --dataset $DATASET \
    --method $METHOD \
    --data_source $DATA_SOURCE \
    --prompt_name $PROMPT \
    --retrieval_split $RETRIEVAL_SPLIT \
    --alpha $ALPHA \
    --attentive_threshold $TAU \
    --attentive_name "c-name" \
    --shots 8 \
    --seed $SEED \
    --epochs $EPOCHS \
    --prefix "NoCutmix_FINAL_S8"

echo "✅ Hoàn thành toàn bộ mẻ chạy!"
