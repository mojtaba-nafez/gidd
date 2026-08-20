#!/usr/bin/env bash
#SBATCH --job-name=gidd-nvib
#SBATCH --partition=gpu
#SBATCH --account=balm
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=/idiap/temp/mnafez/research/gidd/logs-eval-slurm/%x-%j.out
#SBATCH --error=/idiap/temp/mnafez/research/gidd/logs-eval-slurm/%x-%j.err

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

CHECKPOINT="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib"

OUTPUT_DIR="/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/mauve/gidd-nvib"

LOG_FILE="${OUTPUT_DIR}/log-gidd-nvib.log"


# ============================================================
# Create output directory
# ============================================================

mkdir -p "${OUTPUT_DIR}"


# ============================================================
# Run evaluation
# ============================================================

/idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/mauve/gidd_mauve.sh \
    --checkpoint "${CHECKPOINT}" \
    --activate-nvib-noise \
    --output-dir "${OUTPUT_DIR}" --temp 0.1 \
    > "${LOG_FILE}" 2>&1




CHECKPOINT="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small"

OUTPUT_DIR="/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/mauve/gidd"

LOG_FILE="${OUTPUT_DIR}/log-gidd.log"


# ============================================================
# Create output directory
# ============================================================

mkdir -p "${OUTPUT_DIR}"


# ============================================================
# Run evaluation
# ============================================================

/idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/mauve/gidd_mauve.sh \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" --temp 0.3 \
    > "${LOG_FILE}" 2>&1