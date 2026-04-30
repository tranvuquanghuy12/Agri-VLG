import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Add project root to path so 'utils' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import set_model
from utils.dataloader import set_dataloaders
from utils.parser import get_args
from utils.logger import get_logger

def plot_attention_weights(attention_matrix, class_name, output_path):
    """
    Visualizes attention between samples in a batch.
    attention_matrix: shape [batch_size, batch_size]
    """
    print(f"🎨 Generating Attention Map for {class_name}...")
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attention_matrix, cmap='YlOrRd')
    
    plt.colorbar(im, label='Attention Weight')
    ax.set_title(f"GAT Attention Weights ({class_name})", fontsize=14)
    ax.set_xlabel("Keys (Neighbors)", fontsize=12)
    ax.set_ylabel("Queries (Samples)", fontsize=12)
    
    # Label few-shot vs retrieved if possible
    # For now, just a generic heatmap
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Attention map saved to: {output_path}")

def main():
    # Use existing parser for consistency
    args = get_args() 
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Override checkpoint if provided
    # We'll just use args.ckpt_path or similar if it's already in the parser
    # But usually visual scripts need a specific file.
    
    logger = get_logger('output', 'gat_attn', 'console')
    
    # 1. Load Model
    model, preprocess, tokenizer = set_model(args, logger)
    # Load from a specific path if provided in sys.argv, otherwise try to find best model
    import sys
    ckpt_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--checkpoint' and i+1 < len(sys.argv):
            ckpt_path = sys.argv[i+1]
            break
            
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        if 'clip' in checkpoint:
            model.load_state_dict(checkpoint['clip'])
        else:
            model.load_state_dict(checkpoint) # Direct state dict
    model.eval()
    
    # 2. Get Data
    args.data_source = 'fewshot+retrieved'
    train_loader, _, _ = set_dataloaders(args, model, None, preprocess, logger)
    ret_loader, fs_loader = train_loader
    
    # Grab one batch from fewshot and some from retrieved
    fs_batch = next(iter(fs_loader))
    ret_batch = next(iter(ret_loader))
    
    imgs_fs, labels_fs, _ = fs_batch
    imgs_ret, labels_ret, _ = ret_batch
    
    # Combine them to see cross-attention
    combined_imgs = torch.cat([imgs_fs[:4], imgs_ret[:12]], dim=0).to(args.device)
    
    # 3. Forward pass with attention extraction
    with torch.no_grad():
        _, (edge_index, alpha) = model.encode_image(combined_imgs, return_attention=True)
    
    # alpha shape: [num_edges, num_heads]
    # We want to convert this back to a matrix [batch_size, batch_size]
    batch_size = combined_imgs.size(0)
    # Average across heads
    alpha_avg = alpha.mean(dim=1).cpu().numpy()
    
    attn_matrix = np.zeros((batch_size, batch_size))
    # edge_index is [2, num_edges]
    for i in range(len(alpha_avg)):
        u, v = edge_index[0, i], edge_index[1, i]
        attn_matrix[u, v] = alpha_avg[i]
    
    os.makedirs('reports', exist_ok=True)
    plot_attention_weights(attn_matrix, args.dataset, 'reports/GAT_Attention_Map.png')

if __name__ == "__main__":
    main()
    # Example: python Generators/generate_gat_attention.py --checkpoint Weights/stage1_model_best.pth --dataset tlu_states
