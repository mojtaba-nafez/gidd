#!/usr/bin/env bash
#SBATCH --job-name=gidd_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=0-20:00:00
#SBATCH --output=logs-eval-slurm/%x-%j.out
#SBATCH --error=logs-eval-slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

CHECKPOINT_PATH="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small"
OUTPUT_DIR="/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd"
NAME_PREFIX=""
LATENT_NOISE=False
ACTIVATE_NVIB_NOISE=False

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --name-prefix) NAME_PREFIX="$2"; shift 2 ;;
        --latent-noise) LATENT_NOISE=True; shift ;;
        --activate-nvib-noise) ACTIVATE_NVIB_NOISE=True; shift ;;
        -h|--help)
            echo "Usage: sbatch main_idiap.sh [--checkpoint PATH] [--output-dir DIR] [--name-prefix NAME] [--latent-noise] [--activate-nvib-noise]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

CHECKPOINT_PATH="${CHECKPOINT_PATH%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
NAME_PREFIX="${NAME_PREFIX:-$(basename "$CHECKPOINT_PATH")}"

[[ -e "$CHECKPOINT_PATH" ]] || {
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
}

mkdir -p logs-eval-slurm "$OUTPUT_DIR"


echo "======= Conda and CUDA ======="
module load CUDA
source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate gidd
echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "================================"

CHECKPOINT_NAME="$(basename "${CHECKPOINT_PATH%/}")"

echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "NVCC: $(command -v nvcc || echo 'not found')"


python gidd/eval/generate_samples.py path="$CHECKPOINT_PATH" \
    samples_path="${OUTPUT_DIR}/base_samples_${CHECKPOINT_NAME}.pt" num_samples=1024 num_denoising_steps=128 batch_size=16

TEMPS=(0.05 0.1 0.3 0.5 1.0)
DENOISING_STEPS=(32 64 128 256 384 512)

for TEMP in "${TEMPS[@]}"; do
    for NUM_DENOISING_STEPS in "${DENOISING_STEPS[@]}"; do

        CORRECTED_PATH="${OUTPUT_DIR}/base_correct_${CHECKPOINT_NAME}_tmp-${TEMP}_step${NUM_DENOISING_STEPS}.pt"
        METRICS_PATH="${OUTPUT_DIR}/samples_${CHECKPOINT_NAME}_tmp-${TEMP}_step${NUM_DENOISING_STEPS}.json"

        echo "Running temp=$TEMP, denoising_steps=$NUM_DENOISING_STEPS"

        python gidd/eval/self_correction.py \
            path="$CHECKPOINT_PATH" \
            samples_path="${OUTPUT_DIR}/base_samples_${CHECKPOINT_NAME}.pt" \
            corrected_samples_path="$CORRECTED_PATH" \
            batch_size=16 \
            num_denoising_steps="$NUM_DENOISING_STEPS" \
            temp="$TEMP" \
            latent_noise="$LATENT_NOISE" \
            activate_nvib_noise="$ACTIVATE_NVIB_NOISE"

        python gidd/eval/generative_ppl.py \
            samples_path="$CORRECTED_PATH" \
            model_tokenizer=gpt2 \
            pretrained_model=google/gemma-2-9b \
            batch_size=1 \
            metrics_path="$METRICS_PATH"

    done
done