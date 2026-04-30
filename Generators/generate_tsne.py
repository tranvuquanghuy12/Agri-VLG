import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
import sys

# Add project root to path so 'utils' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.models import set_model
from utils.dataloader import set_dataloaders
from utils.parser import get_args
from utils.logger import get_logger

def plot_tsne(features, labels, data_sources, output_path):
    print("🎨 Generating TSNE plot...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    # Map sources to colors/markers
    # data_sources: 0 for retrieved, 1 for few-shot
    unique_sources = np.unique(data_sources)
    colors = ['#1f77b4', '#d62728'] # Blue for retrieved, Red for few-shot
    labels_map = {0: 'Retrieved (Web)', 1: 'Few-shot (Real)'}
    
    for src in unique_sources:
        idx = (data_sources == src)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                    c=colors[src], label=labels_map[src], 
                    alpha=0.6, edgecolors='w', s=50)
    
    plt.title("TSNE Visualization: Domain Alignment (Retrieved vs Few-shot)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ TSNE plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='tlu_states')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to plot per source')
    args = get_args() # Use existing parser for consistency
    
    # Override some args for visualization
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger = get_logger('output', 'tsne', 'console')
    
    # 1. Load Model
    model, preprocess, tokenizer = set_model(args, logger)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['clip'])
    model.eval()
    
    # 2. Get Dataloaders
    # We need both few-shot and retrieved data
    args.data_source = 'fewshot+retrieved'
    train_loader, _, _ = set_dataloaders(args, model, None, preprocess, logger)
    
    # train_loader is a tuple (retrieved_loader, fewshot_loader)
    ret_loader, fs_loader = train_loader
    
    all_features = []
    all_labels = []
    all_sources = [] # 0 for ret, 1 for fs
    
    print("📥 Extracting features...")
    with torch.no_grad():
        # Sample from retrieved
        count = 0
        for imgs, labels, _ in ret_loader:
            imgs = imgs.to(args.device)
            feats = model.encode_image(imgs)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            all_sources.append(np.zeros(len(labels)))
            count += len(labels)
            if count >= args.num_samples: break
            
        # Sample from few-shot
        count = 0
        for imgs, labels, _ in fs_loader:
            imgs = imgs.to(args.device)
            feats = model.encode_image(imgs)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            all_sources.append(np.ones(len(labels)))
            count += len(labels)
            if count >= args.num_samples: break
            
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    sources = np.concatenate(all_sources, axis=0)
    
    os.makedirs('reports', exist_ok=True)
    plot_tsne(features, labels, sources, 'reports/TSNE_Domain_Alignment.png')

if __name__ == "__main__":
    # Note: This script needs to be run with the same args as training
    # Example: python Generators/generate_tsne.py --checkpoint Weights/stage1_model_best.pth --dataset tlu_states
    main()
