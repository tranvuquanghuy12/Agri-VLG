#!/bash/bin
# Semi-Aves Full Comparative Table (4, 8, 16 shots)
PYTHON="/opt/anaconda3/envs/swatai/bin/python"
DATASET="semi_aves"
DATASET_ROOT="/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset"
PREFIX="Final_T0.85_K3"
OUTPUT_ROOT="output/SEMIAVES_FULL_TABLE"

SHOTS=(4 8 16)

for SHOT in "${SHOTS[@]}"; do
    OUTPUT_DIR="$OUTPUT_ROOT/SHOT_$SHOT"
    mkdir -p $OUTPUT_DIR
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 ĐANG XỬ LÝ SEMI-AVES - MỐC $SHOT SHOTS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. FT-Fewshot
    echo "🔹 1. Chạy FT-Fewshot..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "none" \
        --folder "$OUTPUT_DIR/SHOT_${SHOT}_FT_Fewshot" \
        --data_source fewshot --method cutmix --use_gat False \
        --shots $SHOT --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

    # 2. FLYP
    echo "🔹 2. Chạy FLYP..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "none" \
        --folder "$OUTPUT_DIR/SHOT_${SHOT}_FLYP" \
        --data_source fewshot --method flyp --use_gat False \
        --shots $SHOT --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32

    # 3. SWAT (Baseline)
    echo "🔹 3. Chạy SWAT (Baseline)..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
        --folder "$OUTPUT_DIR/SHOT_${SHOT}_SWAT" \
        --data_source fewshot+retrieved --method cutmix --use_gat False \
        --shots $SHOT --epochs 50 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
        
    # 4. Agri-VLG
    echo "🔹 4. Chạy Agri-VLG..."
    $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
        --retrieved_path ./data/ --retrieval_split "$PREFIX.txt" \
        --folder "$OUTPUT_DIR/SHOT_${SHOT}_AgriVLG" \
        --data_source fewshot+retrieved --method cutmix --use_gat True \
        --shots $SHOT --epochs 100 --lr_gat 0.001 --lr_classifier 0.001 --lr_backbone 0.00001 --bsz 32
done
echo "🎉 HOÀN THÀNH TỔNG BIỂU SEMI-AVES!"
