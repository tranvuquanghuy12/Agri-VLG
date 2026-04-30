# Agri-VLG: Graph Attention Network for Retrieval-Augmented Visual Learning

Agri-VLG is a state-of-the-art framework for fine-grained visual classification, leveraging **Graph Attention Networks (GAT)** and **CutMix augmentation** to achieve superior performance in low-data regimes. This repository contains the official implementation for the paper.

## 🌟 Overview
Agri-VLG addresses the challenge of fine-grained classification with limited data by augmenting visual features with retrieved knowledge from large-scale databases (LAION-400M). Relational dependencies between retrieved samples are modeled using a GAT, providing a more robust feature alignment compared to standard fine-tuning.

## 🚀 Performance Highlights
| Dataset | Shots | Zero-shot | Agri-VLG (Ours) | Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Flowers102** | 16 | 42.8% | **98.1%** | +55.3% |
| **TLU States** | 16 | 30.5% | **76.5%** | +46.0% |

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dat_tlu/Agri-VLG.git
   cd Agri-VLG
   ```
2. Setup the environment:
   ```bash
   conda create -n agrivlg python=3.9
   conda activate agrivlg
   pip install -r requirements.txt
   ```

## 📊 Data Preparation
- **Metadata:** All label mappings and dataset splits are provided in the `data/` directory.
- **Images:** Due to size constraints, please download the raw images from the official sources:
  - [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
  - [TLU States](https://github.com/dat_tlu/TLU-States)
  - [Semi-Aves](https://github.com/visipedia/semi-aves)

## 🏎️ Reproducing Results
To reproduce the 16-shot results on TLU States:
```bash
bash experiments/reproduce_tlustates_s16.sh
```

To aggregate all experimental results into a single table:
```bash
python extract_results.py --dir output/
```

## 🖼️ Visualizations
Visit the `Generators/` folder to run scripts for GAT Attention Maps and TSNE visualizations.

## 📜 Citation
If you find our work useful, please cite our paper:
```bibtex
@inproceedings{dat2026agrivlg,
  title={Agri-VLG: Graph Attention Network for Retrieval-Augmented Visual Learning},
  author={Dat Tlu, et al.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
