#!/bin/bash
# Zero-shot NLI (SOCCAT's base model, no fine-tuning): scores all 5 folds and writes
# the cross-fold files. Inference only, about 412,000 pairs: one GPU job is enough.
#
# Submit from this folder (embedding-based_classifiers/zero_shot_nli/):
#     sbatch run_predict.sbatch
#
# Adapt the lines marked ADAPT to your cluster. Finished folds are skipped, so a
# timed-out job can just be resubmitted.
#
#SBATCH --job-name=zero_shot_nli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --partition=gpu             # ADAPT: your GPU partition (remove one '#' to activate)
##SBATCH --account=my_project        # ADAPT: if your cluster requires an account

set -euo pipefail
mkdir -p logs

# ADAPT: load your environment (python with torch, transformers, sentencepiece, pandas, scikit-learn)
# module load python/3.11 cuda/12.1
# source ~/venvs/group-identification-gpu/bin/activate   # see ../requirements.txt for how to create it

# If the compute nodes have no internet access, download the model once on a login node:
#     python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; m='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'; M.from_pretrained(m); T.from_pretrained(m)"
# and uncomment:
# export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python predict_cv.py --fp16
