import os
import random

# --- CONFIGURATION (Adjust paths if needed) ---
# Server path to the image files
dataset_path = "/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset/EuroSAT/eurosat_ms"
# Server path where SWAT-GAT labels are stored
output_dir = "/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT/data/eurosat_ms"

os.makedirs(output_dir, exist_ok=True)

# Fixed class order (alphabetical) to match dataset_utils expectations
classes = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]
class_to_id = {cls: i for i, cls in enumerate(classes)}

val_lines = []
test_lines = []
all_fewshot_lines = {4: [], 8: [], 16: []}
retrieval_lines = [] # For T2T_Filtered.txt (RAL / Ours)

shot_values = [4, 8, 16]
max_shots = max(shot_values)
val_shots = 50  # Number of images for validation per class
test_shots = 100 # Number of images for testing per class
seed = 1

random.seed(seed)

print(f"Scanning directory: {dataset_path}")

for cls in classes:
    cls_id = class_to_id[cls]
    cls_path = os.path.join(dataset_path, cls)
    
    if not os.path.exists(cls_path):
        print(f"Warning: Directory {cls_path} not found. Skipping...")
        continue
        
    imgs = sorted([f for f in os.listdir(cls_path) if f.endswith('.tif')])
    print(f"Class {cls}: Found {len(imgs)} images.")
    
    random.shuffle(imgs)
    
    # 1. Fewshot Sample (Nests: 4 in 8, 8 in 16)
    for s in shot_values:
        current_fs = imgs[:s]
        for img in current_fs:
            all_fewshot_lines[s].append(f"{cls}/{img} {cls_id} 1")
    
    # 2. Validation Sample (50 images) - Taken after the max shots
    current_val = imgs[max_shots : max_shots + val_shots]
    for img in current_val:
        val_lines.append(f"{cls}/{img} {cls_id} 1")
        
    # 3. Test Sample (100 images)
    current_test = imgs[max_shots + val_shots : max_shots + val_shots + test_shots]
    for img in current_test:
        test_lines.append(f"{cls}/{img} {cls_id} 1")

    # 4. Retrieval Pool (All other images for T2T_Filtered.txt)
    # These are treated as "Unlabeled" (is_fewshot = 0) for RAL/Ours
    pool_imgs = imgs[max_shots + val_shots + test_shots:]
    for img in pool_imgs:
        retrieval_lines.append(f"{cls}/{img} {cls_id} 0")

# Sorting by label ID for consistency
for s in shot_values:
    all_fewshot_lines[s].sort(key=lambda x: int(x.split(' ')[1]))
val_lines.sort(key=lambda x: int(x.split(' ')[1]))
test_lines.sort(key=lambda x: int(x.split(' ')[1]))
retrieval_lines.sort(key=lambda x: int(x.split(' ')[1]))

# Save files
for s in shot_values:
    fs_path = os.path.join(output_dir, f'fewshot{s}_seed{seed}.txt')
    with open(fs_path, 'w') as f:
        f.write('\n'.join(all_fewshot_lines[s]))
    print(f"- {os.path.basename(fs_path)}: {len(all_fewshot_lines[s])} lines")

    # For 16-shot, also create the generic train.txt
    if s == 16:
        with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(all_fewshot_lines[s]))

with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_lines))

with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_lines))

# SAVE THE KEY FILE FOR RAL/OURS
with open(os.path.join(output_dir, 'T2T_Filtered.txt'), 'w') as f:
    f.write('\n'.join(retrieval_lines))
    print(f"- T2T_Filtered.txt: {len(retrieval_lines)} lines (For RAL/SWAT-GAT)")

print("-" * 30)
print(f"SUCCESS! Created all label and retrieval files in: {output_dir}")
print(f"- val.txt: {len(val_lines)} lines")
print(f"- test.txt: {len(test_lines)} lines")
print("-" * 30)
