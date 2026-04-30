#!/bin/bash

# Dataset list
DATASETS=("eurosat_ms" "flowers102" "semi-aves")

# Tau and K grids
TAUS=(0.5 0.7 0.85 0.9)
KS=(1 3 5 10)

# Root directories for images
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset"

# Output base directory
OUTPUT_BASE="./output_2"
mkdir -p $OUTPUT_BASE

echo "🚀 Starting Master Grid Search (FINAL CORRECTED FLOW)..."

for DATASET in "${DATASETS[@]}"; do
    echo "📂 Processing Dataset: $DATASET"
    
    # Path logic: Agri-VLG appends /dataset_name to dataset_path
    # So we provide the PARENT directory of the dataset folder
    if [ "$DATASET" == "eurosat_ms" ]; then
        PARENT_IMG_PATH="$DATASET_ROOT/EuroSAT"
    else
        PARENT_IMG_PATH="$DATASET_ROOT"
    fi

    for TAU in "${TAUS[@]}"; do
        for K in "${KS[@]}"; do
            EXP_NAME="${DATASET}_Tau${TAU}_K${K}"
            PREFIX="Grid_T${TAU}_K${K}"
            SAMPLING_LOG="$OUTPUT_BASE/${EXP_NAME}_sampling.log"
            TRAINING_LOG="$OUTPUT_BASE/${EXP_NAME}_run.log"
            
            echo "   🔍 Step 1: Sampling (Tau=$TAU, K=$K)"
            conda run -n swatai python retrieval/sample_retrieval.py \
                --dataset $DATASET \
                --sampling_threshold $TAU \
                --num_samples $K \
                --sampling_method T2T-rank \
                --prefix $PREFIX \
                --prompt_name alternates \
                > "$SAMPLING_LOG" 2>&1
            
            # Paths
            MODEL_CFG="vitb32_openclip_laion400m"
            GEN_LIST="output/${DATASET}_${MODEL_CFG}_${PREFIX}/${PREFIX}.txt"
            TARGET_LIST="data/${DATASET}/${PREFIX}.txt"
            
            if [ -f "$GEN_LIST" ]; then
                mkdir -p "data/${DATASET}"
                cp "$GEN_LIST" "$TARGET_LIST"
                echo "   🚀 Step 2: Training $EXP_NAME"
                
                # DATA SOURCE is fewshot+retrieved to use the sampled images
                conda run -n swatai python main.py \
                    --dataset $DATASET \
                    --dataset_path $PARENT_IMG_PATH \
                    --retrieved_path ./data/ \
                    --retrieval_split "$PREFIX.txt" \
                    --folder "$OUTPUT_BASE/$EXP_NAME" \
                    --data_source fewshot+retrieved \
                    --method cutmix \
                    --use_gat True \
                    --shots 16 \
                    --epochs 15 \
                    --lr_classifier 0.001 \
                    --lr_backbone 0.00001 \
                    --bsz 32 \
                    > "$TRAINING_LOG" 2>&1
                
                # Report result
                ACC=$(grep "Final Best Val Acc:" "$TRAINING_LOG" | tail -n 1 | awk '{print $NF}')
                echo "   ✅ Finished $EXP_NAME | Accuracy: $ACC%"
            else
                echo "   ❌ Error: Sampling failed for $EXP_NAME"
            fi
        done
    done
done

echo "🎉 ALL GRID SEARCH EXPERIMENTS COMPLETED!"
