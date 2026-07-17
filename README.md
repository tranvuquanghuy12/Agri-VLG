<div align="center">
<h1>Agri-VLG: Robust Identification of Counterfeit <br>Agricultural Products using Few-Shot Annotated Data</h1>

[Paper](https://doi.org/10.1109/ACCESS.2024.0429000)

**Dat Tran, Vu Quang Huy Tran, Manh Dung Nguyen, and Hoai Nam Vu**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![GAT](https://img.shields.io/badge/GAT-Graph_Attention-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![CLIP](https://img.shields.io/badge/Backbone-OpenCLIP_ViT--B/32-lightgrey.svg)](https://github.com/mlfoundations/open_clip)

</div>

This repository contains the official PyTorch implementation for the paper: **Agri-VLG: Robust Identification of Counterfeit Agricultural Products using Few-Shot Annotated Data and Open-Source Knowledge** (IEEE Access, 2024).

![Agri-VLG Architecture](assets\Kientructongthe.png)

## 📌 Project Overview

**Agri-VLG** is a robust framework that synergizes few-shot annotated data with open-source knowledge for precise counterfeit agricultural product identification. It addresses the challenges of data scarcity and domain gaps by incorporating:

1. **VLM-Driven Retrieval**: Utilizing Vision-Language Models (VLM) to retrieve high-quality semantic candidates from open data (e.g., LAION-400M).
2. **GAT-Based Feature Refinement**: Employing a Graph Attention Network (GAT) to facilitate cross-image interaction, refining visual features through relational reasoning.
3. **Two-Stage Training Strategy**:
   - *Stage 1 (Representation Learning)*: Finetuning to align representations.
   - *Stage 2 (Balanced Classifier Adaptation)*: Freezing the backbone and retraining the classifier to shift from imbalanced to balanced predictions.

---

## 📊 Performance Highlights

Based on our comprehensive evaluation (refer to Table 2 in the paper), **Agri-VLG** consistently outperforms traditional and few-shot baselines.

| Dataset | Shots | CLIP (Zero-shot) | Agri-VLG (Ours) | Gain |
| :--- | :---: | :---: | :---: | :---: |
| **TLU-States** | 16 | 25.8% | **72.5%** | +46.7% |
| **TLU-Fruit** | 16 | 28.4% | **74.1%** | +45.7% |

---

## ⚙️ Installation

### Clone the Repository

Make sure to clone the repository with its submodules by using:

```bash
git clone --recursive https://github.com/tranvuquanghuy12/Agri-VLG.git
cd Agri-VLG
```

### Python Environment

The codebase was successfully run with Python 3.10 and CUDA 11.8. We recommend using Anaconda.

1. **Create the environment:**

   ```bash
   conda env create -f environment.yml
   ```

   *Alternatively, create it manually:*

   ```bash
   conda create -n swat python=3.10
   conda activate swat
   pip install -r requirements.txt
   ```

### 📊 Data Preparation & Checkpoints

- **Metadata:** All label mappings and dataset splits are provided in the `data/` directory.
- **Images:** Due to size constraints, please download the raw images from the official sources into the `Dataset` folder:
  - [TLU States](https://github.com/dat_tlu/TLU-States) (Our primary agricultural dataset)
  - [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
  - [Semi-Aves](https://github.com/visipedia/semi-aves)
- **LAION-400M Candidates:** Ensure access to the LAION-400M dataset or the necessary pre-computed web-retrieved candidates in the `data` directory.
- **Checkpoints:** Download the pre-trained weights for the Vision Transformer (ViT-B) and place them in the `Weights` directory.

---

## 🚀 Running Agri-VLG

### 1. Manual Two-Stage Training

Fine-tune the model using the dual-stage graph-augmented pipeline on the TLU-States dataset (e.g., for 16-shot):

```bash
python main.py \
    --method cutmix \
    --data_source fewshot+retrieved \
    --dataset tlu_states \
    --shots 16 \
    --use_gat True \
    --epochs 200 \
    --folder output/Agri-VLG_Refined
```

*(Note: Adding `--skip_stage2` will execute only Stage 1: Representation Learning).*

### 2. Unified Experiment Suite & Baselines

To run master scripts that automatically evaluate and compare all baseline methods (Zero-Shot CLIP, Retrieval Baseline, Few-Shot Baseline, etc.) against Agri-VLG:

```bash
python SGR_Baseline_Master.py
```

### 3. Training Dynamics & Performance Charts

To evaluate the performance and plot the training dynamics (Accuracy & Loss curves) on the TLU-States dataset for 4-shot, 8-shot, and 16-shot configurations:

```bash
python generate_tlu_dynamics.py
python plot_training_dynamics.py
```

---

## 📁 Repository Organization

- `utils/models.py`: Defines the `SWATModelWithGAT` (Agri-VLG) architecture with graph attention layers.
- `utils/training.py`: Implements the optimization logic, including CutMix data augmentation and the two-stage process.
- `utils/losses.py`: Contains the `BalancedSoftmaxLoss` used in Stage 2.
- `retrieval/sample_retrieval.py`: Logic for similarity matching and web-guided candidate filtering.
- `SGR_Baseline_Master.py`: Centralized tool for multi-method benchmarking and result aggregation.

---

## 📫 Contact

For any questions related to our paper, please send an email to <dat.trananh@tlu.edu.vn>.

## 📖 BibTeX

If you find our work useful, please consider citing our paper:

```bibtex
@article{tran2024agrivlg,
  title     = {{Agri-VLG: Robust Identification of Counterfeit Agricultural Products using Few-Shot Annotated Data and Open-Source Knowledge}},
  author    = {Tran, Dat and Tran, Vu Quang Huy and Nguyen, Manh Dung and Vu, Hoai Nam},
  journal   = {IEEE Access},
  year      = {2024},
  doi       = {10.1109/ACCESS.2024.0429000}
}
```

## 📜 Copyright & License

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: <https://www.apache.org/licenses/LICENSE-2.0>

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: <https://creativecommons.org/licenses/by/4.0/legalcode>
