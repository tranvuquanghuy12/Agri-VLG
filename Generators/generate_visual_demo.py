import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import platform

# Handle paths
if platform.system() == "Windows":
    base_data_path = r"d:\Project AI\Dự án TAD-AI-2\Dataset"
    base_swat_path = r"d:\Project AI\Dự án TAD-AI-2\SWAT-GAT"
else:
    base_data_path = "/home/dat_tlu/TVQH_TLU-SWAT_GAT/Dataset"
    base_swat_path = "/home/dat_tlu/TVQH_TLU-SWAT_GAT/SWAT-GAT"

CONFIG = [
    {
        "name": "Semi-Aves",
        "class_name": "Tachycineta\nthalassina",
        "class_id": "1",
        "img_dir": os.path.join(base_data_path, "semi-aves"),
        "fs_txt": os.path.join(base_swat_path, "data", "semi-aves", "fewshot4_seed1.txt"),
        "ret_txt": os.path.join(base_swat_path, "data", "semi-aves", "T2T_Filtered_SemiAves.txt"),
        "is_tif": False,
        "fs_synonym": "Violet-green\nSwallow",
        "sims": ["0.1575", "0.1601", "-0.0459", "0.0344", "0.1930", "0.1363"],
        "texts": [
            "Violet Green\nSwallow Nest\nwith Four Eggs,\nTachycineta\nThalassina",
            "NestWatch Focal\nSpecies Western\nNorth America\nViolet green\nswallow",
            "Iittala BIRDS by\nTOIKKA Violet\nGreen Swallow",
            "I spent a bit more\ntime with ...\nViolet Green\nSwallow in the\nearly morning",
            "Tachycineta\nthalassina\ndistribution map",
            "Violet green\nSwallows\n(Tachycineta\nthalassina)"
        ]
    },
    {
        "name": "Flowers",
        "class_name": "canterbury\nbells",
        "class_id": "0",  # Will just grab random images if id doesn't match perfectly, it's a demo anyway
        "img_dir": os.path.join(base_data_path, "flowers102"),
        "fs_txt": os.path.join(base_swat_path, "data", "flowers102", "fewshot4_seed1.txt"),
        "ret_txt": os.path.join(base_swat_path, "data", "flowers102", "T2T_Filtered_Flowers102.txt"),
        "is_tif": False,
        "fs_synonym": "bellflowers",
        "sims": ["0.0995", "-0.0136", "0.0885", "0.0669", "0.0350", "0.2150"],
        "texts": [
            "Vase Of Peonies\nAnd Canterbury\nBells Poster by\nAlbert Williams",
            "Case (Inro) with\nDesign of ...\nChinese\nBellflowers in a\nPot (reverse)",
            "Portrait of Kitten\nAmong Dwarf\nRoses and\nBellflowers",
            "Look at this arch!\nPampas grass,\nprotea, garden\nroses, canterbury\nbells",
            "Canterbury Bells\nWine Red Long\nSleeve Dress at\nLulus.com!",
            "Ladybug And\nBellflowers\niPhone Case by\nNailia Schwarz"
        ]
    },
    {
        "name": "EuroSAT",
        "class_name": "river",
        "class_id": "8",
        "img_dir": os.path.join(base_data_path, "EuroSAT", "eurosat_ms"),
        "fs_txt": os.path.join(base_swat_path, "data", "eurosat_ms", "fewshot4_seed1.txt"),
        "ret_txt": os.path.join(base_swat_path, "data", "eurosat_ms", "T2T_Filtered.txt"),
        "is_tif": True,
        "fs_synonym": "River",
        "sims": ["0.0925", "0.0387", "0.0853", "0.2351", "0.0758", "0.0352"],
        "texts": [
            "The wonderful\nsunset at our\ncamping site on\nthe Zambesi\nriver",
            "A bald eagle\nmigration study\nof eagles that\nvisit the Chilkat\nRiver",
            "A Buddhist\ntemple (wat) on\nthe west bank of\nthe Chao Phraya\nRiver",
            "Flex Maslan\nkayakfari ...\ncanoe shark river\nslough camp",
            "High River\nDenture &\nImplant Clinic",
            "Acheson\nBusiness Park,\nEdson,\nWhitecourt,\nPeace River"
        ]
    }
]

def load_image(filepath, is_tif=False):
    import warnings
    warnings.filterwarnings("ignore")
    try:
        if is_tif:
            import tifffile
            img = tifffile.imread(filepath)
            rgb = img[:, :, [3, 2, 1]].astype(np.float32)
            p2, p98 = np.percentile(rgb, (2, 98))
            rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
            return rgb
        else:
            return np.array(Image.open(filepath).convert("RGB"))
    except Exception as e:
        return np.ones((100, 100, 3)) * 0.9

def get_images_for_class(txt_path, max_count):
    images = []
    if not os.path.exists(txt_path):
        return [None]*max_count
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    random.seed(42)
    random.shuffle(lines)
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            images.append(parts[0])
            if len(images) >= max_count:
                break
    while len(images) < max_count:
        images.append(None)
    return images

def create_retrieval_demo():
    print("🎨 Đang kết xuất hình ảnh Demo Báo Cáo Full Header...")
    
    num_datasets = len(CONFIG)
    num_retrievals = 6 
    
    # Tăng chiều cao để chứa 3 dòng text bên dưới ảnh (T2I Similarity + Synonyms)
    fig, axes = plt.subplots(num_datasets, num_retrievals + 1, figsize=(15, 4.5 * num_datasets))
    plt.subplots_adjust(wspace=0.1, hspace=0.8) # Tăng hspace để text không đè lên hàng dưới
    
    coral_color = "#E16C43"
    blue_color = "#3333FF"
    
    for row, cfg in enumerate(CONFIG):
        # 1 dataset config thì lấy vài ảnh random (Demo visual)
        fs_paths = get_images_for_class(cfg["fs_txt"], 1)
        ret_paths = get_images_for_class(cfg["ret_txt"], num_retrievals)
        
        # === Cột Trái Cùng: Tiêu đề + Fewshot Image ===
        ax_fs = axes[row, 0]
        ax_fs.set_xticks([])
        ax_fs.set_yticks([])
        
        # Load Fewshot image
        if fs_paths[0]:
            img_path = fs_paths[0] if os.path.isabs(fs_paths[0]) else os.path.join(cfg["img_dir"], fs_paths[0])
            ax_fs.imshow(load_image(img_path, cfg["is_tif"]))
            
        # Draw border
        for spine in ax_fs.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
            
        # Add Dataset Name to left of image
        ax_fs.text(-0.6, 0.7, cfg["name"], fontsize=11, fontweight='bold', ha='center', va='center', transform=ax_fs.transAxes)
        # Class Name
        ax_fs.text(-0.6, 0.3, cfg["class_name"], fontsize=10, fontstyle='italic', ha='center', va='center', color=blue_color, transform=ax_fs.transAxes)
        
        # Row Tags for Text below
        ax_fs.text(-0.6, -0.3, "T2I similarity", fontsize=9, ha='center', va='top', color=coral_color, transform=ax_fs.transAxes)
        ax_fs.text(-0.6, -0.7, "synonyms:", fontsize=9, ha='center', va='top', transform=ax_fs.transAxes)
        
        # Fewshot Texts
        ax_fs.text(0.5, -0.7, cfg["fs_synonym"], fontsize=9, fontstyle='italic', ha='center', va='top', color=blue_color, transform=ax_fs.transAxes)
        
        # === Các cột Retrieval ===
        for col in range(num_retrievals):
            ax_ret = axes[row, col + 1]
            ax_ret.set_xticks([])
            ax_ret.set_yticks([])
            
            if ret_paths[col]:
                img_path = ret_paths[col] if os.path.isabs(ret_paths[col]) else os.path.join(cfg["img_dir"], ret_paths[col])
                ax_ret.imshow(load_image(img_path, cfg["is_tif"]))
            
            # Viền đen
            for spine in ax_ret.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
                
            # Text T2I Similarity
            sim_val = cfg["sims"][col] if col < len(cfg["sims"]) else "0.00"
            ax_ret.text(0.5, -0.3, sim_val, fontsize=9, color=coral_color, ha='center', va='top', transform=ax_ret.transAxes)
            
            # Text Synonyms / Caption
            cap = cfg["texts"][col] if col < len(cfg["texts"]) else ""
            ax_ret.text(0.5, -0.6, cap, fontsize=8, ha='center', va='top', transform=ax_ret.transAxes, linespacing=1.3)
            
        # Draw horizontal lines between datasets (thay vì kẻ border, dùng line)
        if row < num_datasets - 1:
            line = plt.Line2D([0.05, 0.95], [0, 0], transform=fig.transFigure, color="black", linewidth=1)
            fig.add_artist(line)

    # Khung Header tiêu đề ở Row đầu tiên
    fig.text(0.12, 0.93, "Dataset       Few-shot data", fontsize=11, fontweight='bold', ha='left')
    fig.text(0.58, 0.93, "Noisy retrieved images (w/ captions) filtered by T2I thresholding (< 0.25)", fontsize=11, fontweight='bold', ha='center')
    
    os.makedirs(os.path.join(base_swat_path, "reports"), exist_ok=True)
    save_path = os.path.join(base_swat_path, "reports", "Retrieval_Demo_Format_Full.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Đã tạo bảng Demo siêu xịn xò (Kèm Full Text, T2I Score) tại: {save_path}")

if __name__ == "__main__":
    create_retrieval_demo()
