#!/bin/bash

# Cấu hình môi trường
PYTHON=/opt/anaconda3/envs/swatai/bin/python
DATASET=semi_aves
DATASET_ROOT=/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/semi-aves
OUTPUT_ROOT=output/RECOVERY_GRID_SEMIAVES

mkdir -p $OUTPUT_ROOT

# Danh sách tham số Grid Search
TAUS=(0.5 0.7 0.85 0.9)
KS=(1 3 5 10)

echo "🚀 BẮT ĐẦU CHẠY GRID SEARCH RECOVERY CHO SEMI-AVES..."
echo "🔥 CẤU HÌNH: prompt_name=most_common_name | Zero-shot target: ~43.7%"

for T in "${TAUS[@]}"; do
    for K in "${KS[@]}"; do
        CASE_NAME="Grid_T${T}_K${K}"
        echo "----------------------------------------------------"
        echo "🔹 Đang xử lý: Tau=$T, K=$K"
        
        # 1. Chạy Sampling (nếu chưa có file Grid_T..._K...)
        RETRIEVAL_FILE="Grid_T${T}_K${K}.txt"
        if [ ! -f "data/semi_aves/$RETRIEVAL_FILE" ]; then
            echo "   -> Đang cào dữ liệu (Sampling)..."
            $PYTHON sample_retrieval.py --dataset $DATASET --dataset_path $DATASET_ROOT \
                --database LAION400M --temperature $T --K $K --prefix Grid
        fi

        # 2. Chạy Training Full các Stage
        echo "   -> Đang huấn luyện (Training)..."
        $PYTHON main.py --dataset $DATASET --dataset_path $DATASET_ROOT \
            --retrieved_path ./data/ --retrieval_split $RETRIEVAL_FILE \
            --folder $OUTPUT_ROOT/$CASE_NAME \
            --data_source fewshot+retrieved --method finetune --use_gat True \
            --shots 4 --epochs 20 --prompt_name most_common_name --recal_prompt \
            --log_mode file
            
        echo "✅ Hoàn thành $CASE_NAME"
    done
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 TẤT CẢ ĐÃ XONG! Sếp dùng lệnh sau để xem bảng tổng hợp:"
echo "grep \"CSV Summary\" $OUTPUT_ROOT/*/output_semi_aves/*/main.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
