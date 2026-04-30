#!/bin/bash
# =========================================================
# Agri-VLG COMPLETE EXPERIMENT SUITE
# =========================================================

echo "🚀 Starting COMPLETE EXPERIMENT SUITE..."

bash experiments/run_01_priority_baselines.sh
bash experiments/run_02_sensitivity_tau.sh
bash experiments/run_03_sensitivity_k.sh
bash experiments/run_04_ablation_synergy.sh
bash experiments/run_05_complexity.sh
bash experiments/run_06_visualizations.sh

echo "✅ ALL EXPERIMENTS COMPLETED. Check 'output/' and 'reports/' folders."
