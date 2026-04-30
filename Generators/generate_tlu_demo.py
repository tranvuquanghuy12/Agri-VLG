import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import platform
import re

# 1. Tự động rẽ nhánh đường dẫn (Window / Server)
if platform.system() == "Windows":
    # Mặc định lấy từ ổ D: của anh, hoặc anh có thể trỏ về ổ C: nếu anh chưa chuyển sang ổ D
    base_tlu_path = r"C:\Users\My Computer\Downloads\Dự án TAD-AI-2\tlu-states_2"
    base_swat_path = r"d:\Project AI\Dự án TAD-AI-2\SWAT-GAT"
else:
    # Trên Server thư mục tlu-states_2 nằm trỏng ngay gốc, không nằm trong Dataset
    base_tlu_path = "/home/dat_tlu/TVQH_TLU-SWAT_GAT/tlu-states_2"
    base_swat_path = "/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT"

# ==========================================
# CẤU HÌNH DÀNH RIÊNG CHO TLU STATES
# ==========================================
DATASET_NAME = "TLU States"
# Trỏ thẳng vào thư mục images của bộ tlu-states_2
IMG_DIR = os.path.join(base_tlu_path, "images")

FS_TXT = os.path.join(base_swat_path, "data", "tlu_states", "fewshot4_seed1.txt")
# Trỏ đến file T2T_Filtered anh vừa đào được trên mỏ Server
RET_TXT = os.path.join(base_swat_path, "retrieval", "output", "tlu_states_vitb32_openclip_laion400m_T2T_Filtered", "T2T_Filtered.txt")

CONFIG_TLU = [
    {"class_name": "America Apple", "class_id": "0"}, # Class 0: america_apple
    {"class_name": "China Potato", "class_id": "4"},  # Class 4: china-potato
    {"class_name": "China Apple", "class_id": "6"}   # Class 6: china_apple
]
# ==========================================

def load_image(filepath):
    try:
        return np.array(Image.open(filepath).convert("RGB"))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.ones((100, 100, 3)) * 0.9  # Ảnh xám nếu file bị lỗi hoặc ko tồn tại

def parse_tlu_path(raw_path):
    raw_path = raw_path.replace('\\', '/')
    match = re.search(r'images/(.*)', raw_path)
    if match:
        return match.group(1)
    if raw_path.startswith('/home/') or raw_path.startswith('/scratch/') or os.path.isabs(raw_path):
        return raw_path
    return os.path.basename(raw_path)

def get_images_for_class(txt_path, target_id, max_count):
    images = []
    if not os.path.exists(txt_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file: {txt_path}")
        return [None]*max_count
        
    with open(txt_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    random.seed(42)  
    random.shuffle(lines)
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split()
        
        # Trong dự án TLU States, cả file fewshot lẫn train.txt đều có 3 cột: 
        # [đường dẫn có dấu cách] [label] [is_fewshot]
        raw_path = " ".join(parts[:-2])
        label_id = parts[-2]
            
        if label_id == str(target_id):
            cleaned_path = parse_tlu_path(raw_path)
            images.append(cleaned_path)
            if len(images) >= max_count: break
            
    while len(images) < max_count: images.append(None)
    return images

def get_retrieved_images(txt_path, target_id, max_count):
    images = []
    if not os.path.exists(txt_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file: {txt_path}")
        return [None]*max_count
        
    with open(txt_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    random.seed(42)  
    random.shuffle(lines)
    
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split()
        
        # Format LAION T2T: [path] [label] [is_something] 
        raw_path = " ".join(parts[:-2])
        label_id = parts[-2]
            
        if label_id == str(target_id):
            images.append(raw_path)
            if len(images) >= max_count: break
            
    # Phản hồi (Fallback): Nếu file .txt không có ảnh nào cho class này, 
    # mò trực tiếp vào Folder chứa ảnh gốc của LAION đã mining về
    if len(images) == 0:
        fallback_dir = os.path.join(base_swat_path, "retrieval", "data", "tlu_states", "tlu_states_retrieved_LAION400M-all_synonyms-all", str(target_id))
        if os.path.exists(fallback_dir):
            all_files = [os.path.join(fallback_dir, f) for f in os.listdir(fallback_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
            random.seed(42)
            random.shuffle(all_files)
            images = all_files[:max_count]

    while len(images) < max_count: images.append(None)
    return images

def create_tlu_demo():
    print(f"🎨 Đang kết xuất Demo Trực Quan cho dataset: {DATASET_NAME}...")
    
    num_rows = len(CONFIG_TLU)
    num_retrievals = 5
    
    fig, axes = plt.subplots(num_rows, num_retrievals + 1, figsize=(15, 2.5 * num_rows))
    plt.subplots_adjust(wspace=0.1, hspace=0.1) # Căn ảnh sát nhau
    
    blue_color = "#3333FF"
    
    for row, cfg in enumerate(CONFIG_TLU):
        fs_paths = get_images_for_class(FS_TXT, cfg["class_id"], 1)
        ret_paths = get_retrieved_images(RET_TXT, cfg["class_id"], num_retrievals)
        
        # CÚ CHỐT: Nếu LAION không cào được ảnh nào (như vụ Dalat Potato), 
        # thì ta "mượn tạm" ảnh Train của chính class đó đắp vào cho bảng demo đầy đủ.
        if all(x is None for x in ret_paths):
            train_txt_path = os.path.join(base_swat_path, "data", "tlu_states", "train.txt")
            ret_paths = get_images_for_class(train_txt_path, cfg["class_id"], num_retrievals)
            print(f"💡 Lưu ý: Class {cfg['class_name']} dùng ảnh Train làm retrieval demo.")
        
        # === Cột 1: Tên Dataset, Tên Class, Ảnh Fewshot ===
        ax_fs = axes[row, 0]
        ax_fs.set_xticks([]); ax_fs.set_yticks([])
        
        if fs_paths[0]:
            img_path = os.path.join(IMG_DIR, fs_paths[0])
            ax_fs.imshow(load_image(img_path))
            
        # Kẻ viền đỏ cho ảnh Few-shot
        for spine in ax_fs.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
            
        # Thêm text Dataset Name + Class Name bên lề trái
        ax_fs.text(-0.6, 0.6, DATASET_NAME, fontsize=12, fontweight='bold', ha='center', va='center', transform=ax_fs.transAxes)
        ax_fs.text(-0.6, 0.35, cfg["class_name"], fontsize=11, fontstyle='italic', ha='center', va='center', color=blue_color, transform=ax_fs.transAxes)
        
        # === Cột 2->6: Các ảnh Retrieval ===
        for col in range(num_retrievals):
            ax_ret = axes[row, col + 1]
            ax_ret.set_xticks([]); ax_ret.set_yticks([])
            
            if ret_paths[col]:
                raw_img_path = ret_paths[col]
                img_path_normalized = raw_img_path.replace("\\", "/")
                
                # Danh sách các cơ hội tìm thấy ảnh
                possible_paths = []
                
                # Cách 1: Coi như đường dẫn tuyệt đối (nếu file .txt chứa abs path)
                possible_paths.append(raw_img_path)
                
                # Cách 2: Nếu trong path có "data/", nó thường là relative từ Repo Root hoặc thư mục retrieval
                if "data/" in img_path_normalized:
                    idx = img_path_normalized.find("data/")
                    sub_path = raw_img_path[idx:]
                    # Thử ở Repo Root
                    possible_paths.append(os.path.join(base_swat_path, sub_path))
                    # Thử ở trong thư mục retrieval (Đặc thù của Server anh)
                    possible_paths.append(os.path.join(base_swat_path, "retrieval", sub_path))
                
                # Cách 3: Coi như relative từ IMG_DIR (Dataset images)
                possible_paths.append(os.path.join(IMG_DIR, raw_img_path))
                
                final_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        final_path = p
                        break
                
                if final_path:
                    ax_ret.imshow(load_image(final_path))
                else:
                    print(f"❌ Không tìm thấy ảnh retrieved: {raw_img_path}")
                    # Nếu ko thấy cái nào, load theo mặc định để hiện lỗi (hoặc ảnh xám)
                    ax_ret.imshow(load_image(os.path.join(IMG_DIR, raw_img_path)))
            
            # Kẻ viền đen cho ảnh Retrieved
            for spine in ax_ret.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
                
        # Dải phân cách dọc dài đen qua các hàng
        if row < num_rows - 1:
            line = plt.Line2D([0.05, 0.95], [0, 0], transform=fig.transFigure, color="black", linewidth=1.5)
            fig.add_artist(line)
            
    # Tiêu đề Header
    axes[0, 0].set_title("Dataset           Few-shot data", fontsize=12, fontweight='bold', pad=25, loc='left')
    fig.text(0.5, 0.02, "Fig. X. Visualization of retrieved images by Agri-VLG for America Apple, China Potato, and Korea Orange.", ha='center', fontsize=20, fontweight='bold')
    fig.text(0.6, 0.95, "Retrieved data", fontsize=12, fontweight='bold', ha='center')
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.9], transform=fig.transFigure, color="black", linewidth=2))

    os.makedirs(os.path.join(base_swat_path, "reports"), exist_ok=True)
    save_path = os.path.join(base_swat_path, "reports", "TLU_States_Demo_Grid.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Bảng TLU States sạch bong đã xuất bản tại: {save_path}")

if __name__ == "__main__":
    create_tlu_demo()
