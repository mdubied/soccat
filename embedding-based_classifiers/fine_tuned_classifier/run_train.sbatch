#!/bin/bash
# Fine-tuned classifier, 5-fold CV (one fold per array task).
# Submit from this folder: sbatch run_train.sbatch
# Afterwards (no GPU needed): python train_cv.py --finalize
#SBATCH --job-name=ft_classifier
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_309_general
#SBATCH --output=logs/%x_fold%a_%j.out
#SBATCH --error=logs/%x_fold%a_%j.err

set -euo pipefail
mkdir -p logs

module purge
source /scratch/md6382/venvs/soccat-gpu/bin/activate

echo "Job $SLURM_JOB_ID, fold $SLURM_ARRAY_TASK_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

CKPT_DIR="${TMPDIR:-$PWD/data/ckpt}/ft_classifier_fold${SLURM_ARRAY_TASK_ID}"

python -u train_cv.py --folds "$SLURM_ARRAY_TASK_ID" --ckpt_dir "$CKPT_DIR"
