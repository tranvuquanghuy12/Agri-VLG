#!/bin/bash

PYTHON=/opt/anaconda3/envs/swatai/bin/python

echo "🔥 BẮT ĐẦU CHIẾN DỊCH TỔNG LỰC (SEQUENTIAL MODE) 🔥"

# 1. TLU States No Cutmix (Bù phần thiếu)
echo "----------------------------------------------------"
echo "🚀 [1/5] Chạy TLU States No Cutmix (4 & 8 shots)..."
bash worker_tlu_states_nocutmix_4_8.sh

# 2. Semi-Aves Full Table (Bảng so sánh chính)
echo "----------------------------------------------------"
echo "🚀 [2/5] Chạy Semi-Aves Full Table (4, 8, 16 shots)..."
bash worker_semiaves_full_table.sh

# 3. TLU States Baselines (Đối đầu SOTA)
echo "----------------------------------------------------"
echo "🚀 [3/5] Chạy TLU States Baselines..."
bash worker_tlu_states_baselines.sh

# 4. Hồi sinh Model Flowers102 Best
echo "----------------------------------------------------"
echo "🚀 [4/5] Hồi sinh Flowers102 Best Model..."
$PYTHON main.py --dataset flowers102 --dataset_path data/flowers102 \
    --data_source fewshot+retrieved --method cutmix --use_gat True \
    --shots 16 --epochs 20 --prompt_name most_common_name --prefix REGEN_BEST \
    --folder output/REGEN_FLOWERS_BEST

# 5. Hồi sinh Model Semi-Aves Best
echo "----------------------------------------------------"
echo "🚀 [5/5] Hồi sinh Semi-Aves Best Model..."
$PYTHON main.py --dataset semi_aves --dataset_path /home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/semi-aves \
    --retrieved_path ./data/ --retrieval_split Grid_T0.9_K10.txt \
    --data_source fewshot+retrieved --method finetune --use_gat True \
    --shots 16 --epochs 20 --prompt_name most_common_name --prefix REGEN_BEST \
    --folder output/REGEN_SEMIAVES_BEST

echo "✅ TẤT CẢ ĐÃ HOÀN TẤT! AGRI-VLG ĐÃ SẴN SÀNG CHINH PHỤC THẾ GIỚI! 🏆"
