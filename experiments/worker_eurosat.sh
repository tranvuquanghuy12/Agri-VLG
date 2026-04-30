#!/bin/bash
PYTHON="/opt/anaconda3/envs/swatai/bin/python"
DATASET="eurosat_ms"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/EuroSAT/eurosat_ms"
PREFIX="Grid_T0.85_K10"
SHOTS=(4 8 16)
OUTPUT_ROOT="output/PARALLEL_TABLE/${DATASET}"
mkdir -p $OUTPUT_ROOT

for SHOT in "${SHOTS[@]}"
do
    echo "🚀 Đang chạy bộ Full Benchmarking cho ${DATASET} - ${SHOT} shots..."

    # 1. SWAT (Baseline) - GAT False, Method Cutmix
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/SHOT_${SHOT}/SWAT_Baseline" \
        --data_source fewshot+retrieved --method cutmix --use_gat False \
        --shots $SHOT --epochs 20 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

    # 2. FT on few-shot (ours) - GAT True, Source fewshot, Method finetune
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/SHOT_${SHOT}/Argi_FT_FewShot" \
        --data_source fewshot --method finetune --use_gat True \
        --shots $SHOT --epochs 20 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

    # 3. FT on retrieved (ours) - GAT True, Source retrieved, Method finetune
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/SHOT_${SHOT}/Argi_FT_Retrieved" \
        --data_source retrieved --method finetune --use_gat True \
        --shots $SHOT --epochs 20 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

    # 4. AgriVLG_Cutmix - GAT True, Method cutmix
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/SHOT_${SHOT}/AgriVLG_Cutmix" \
        --data_source fewshot+retrieved --method cutmix --use_gat True \
        --shots $SHOT --epochs 20 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

    # 5. Ours_Final (No Cutmix) - GAT True, Method finetune
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/SHOT_${SHOT}/Ours_Final" \
        --data_source fewshot+retrieved --method finetune --use_gat True \
        --shots $SHOT --epochs 20 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

    # 6. CMLP (Cross-Modal Linear Probing) - GAT False, Source fewshot, Method probing
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/SHOT_${SHOT}/CMLP" \
        --data_source fewshot --method probing --use_gat False \
        --shots $SHOT --epochs 20 --lr_classifier 0.001 --bsz 32

    # 7. FLYP (Finetune Language-Image Pre-training) - GAT False, Source fewshot, Method finetune
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "${PREFIX}.txt" \
        --folder "$OUTPUT_ROOT/SHOT_${SHOT}/FLYP" \
        --data_source fewshot --method finetune --use_gat False \
        --shots $SHOT --epochs 20 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
done
