#!/bin/bash
# Fine-tuned multi-label classifier, 5-fold CV: one fold per array task (one GPU each).
#
# Submit from this folder (embedding-based_classifiers/fine_tuned_classifier/):
#     jid=$(sbatch --parsable run_train.sbatch)
# then, to write the cross-fold files once all 5 folds are done:
#     sbatch --dependency=afterok:$jid --wrap "python train_cv.py --finalize" --time=00:15:00 --mem=16G
# (or simply run `python train_cv.py --finalize` afterwards; it needs no GPU).
#
# Adapt the lines marked ADAPT to your cluster. Finished folds are skipped, so a
# failed or timed-out task can just be resubmitted, e.g. sbatch --array=3 run_train.sbatch
#
#SBATCH --job-name=ft_classifier
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_fold%a_%j.out
#SBATCH --error=logs/%x_fold%a_%j.err
##SBATCH --partition=gpu             # ADAPT: your GPU partition (remove one '#' to activate)
##SBATCH --account=my_project        # ADAPT: if your cluster requires an account

set -euo pipefail
mkdir -p logs

# ADAPT: load your environment (python with torch, transformers, sentencepiece, pandas, scikit-learn)
# module load python/3.11 cuda/12.1
# source ~/venvs/group-identification-gpu/bin/activate   # see ../requirements.txt for how to create it

# If the compute nodes have no internet access, download the model once on a login node:
#     python -c "from transformers import AutoModel, AutoTokenizer as T; m='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'; AutoModel.from_pretrained(m); T.from_pretrained(m)"
# and uncomment:
# export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID, fold $SLURM_ARRAY_TASK_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# checkpoints on the node-local disk if available (deleted after the fold)
CKPT_DIR="${TMPDIR:-$PWD/data/ckpt}/ft_classifier_fold${SLURM_ARRAY_TASK_ID}"

python train_cv.py --folds "$SLURM_ARRAY_TASK_ID" --ckpt_dir "$CKPT_DIR"
