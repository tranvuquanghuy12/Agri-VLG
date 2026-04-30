import os

dataset_path = "/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/semi-aves"
output_dir = "/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT/data/semi-aves"

print("Đang phân tích tập trainval_images để tạo kho ảnh Retrieval...")

# 1. Đọc list ảnh Validation để loại trừ (không cho học trộm)
val_file = os.path.join(output_dir, 'val.txt')
exclude_set = set()
if os.path.exists(val_file):
    with open(val_file, 'r') as f:
        for line in f:
            path = line.strip().split()[0]
            exclude_set.add(path)

# Thư mục gốc chứa ảnh train
trainval_dir = os.path.join(dataset_path, "trainval_images")
retrieval_lines = []

# 2. Duyệt qua toàn bộ 200 class folder (từ 0 đến 199)
valid_extensions = ('.jpg', '.jpeg', '.png')
total_images = 0

for cls_folder in sorted(os.listdir(trainval_dir)):
    cls_path = os.path.join(trainval_dir, cls_folder)
    
    # Chỉ quét nếu là thư mục số (class)
    if os.path.isdir(cls_path) and cls_folder.isdigit():
        cls_id = int(cls_folder)
        imgs = sorted(os.listdir(cls_path))
        
        for img in imgs:
            if img.lower().endswith(valid_extensions):
                rel_path = f"trainval_images/{cls_folder}/{img}"
                total_images += 1
                
                # Bỏ qua những ảnh đã dành cho Validation
                if rel_path not in exclude_set:
                    # Gán is_fewshot = 0 (Unlabeled / Pool)
                    retrieval_lines.append(f"{rel_path} {cls_id} 0")

out_file = os.path.join(output_dir, 'T2T_Filtered_SemiAves.txt')
with open(out_file, 'w') as f:
    f.write("\n".join(retrieval_lines))

print("-" * 50)
print(f"✅ ĐÃ HÔ BIẾN THÀNH CÔNG!")
print(f"Tổng số ảnh trong thư mục: {total_images}")
print(f"Đã đưa vào kho ảnh SSL (Retrieval Pool): {len(retrieval_lines)}")
print(f"Lưu tại: {out_file}")
print("-" * 50)
