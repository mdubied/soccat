#!/bin/bash
# Zero-shot NLI: scores all 5 folds and writes the cross-fold files.
# Submit from this folder: sbatch run_predict.sbatch
#SBATCH --job-name=zero_shot_nli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_309_general
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

module purge
source /scratch/md6382/venvs/soccat-gpu/bin/activate

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -u predict_cv.py --fp16
