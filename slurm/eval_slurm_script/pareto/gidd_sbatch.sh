#!/usr/bin/env bash
#SBATCH --job-name=gidd-nvib
#SBATCH --partition=gpu
#SBATCH --account=balm
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

set -euo pipefail

# ============================================================
# Environment
# ============================================================

eval "$(conda shell.bash hook)"
conda activate gidd

cd /idiap/temp/mnafez/research/gidd/


# ============================================================
# Paths
# ============================================================

CHECKPOINT="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib-3-4-5-6-7-8"

OUTPUT_DIR="/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd-nvib-3-4-5-6-7-8"

LOG_FILE="${OUTPUT_DIR}/log-gidd-nvib.log"


# ============================================================
# Create output directory
# ============================================================

mkdir -p "${OUTPUT_DIR}"


# ============================================================
# Run evaluation
# ============================================================

/idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd_interactive.sh \
    --checkpoint "${CHECKPOINT}" \
    --activate-nvib-noise \
    --output-dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1