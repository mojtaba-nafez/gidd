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

set -e

mkdir -p logs-eval-slurm

# =========================================================
# Usage:
#
# sbatch -p gpu -A balm main_idiap.sh \
#   <checkpoint_path> \
#   <output_dir> \
#   [name_prefix]
#
# Example:
#
# sbatch -p gpu -A balm main_idiap.sh \
#   /path/to/checkpoint \
#   /path/to/output \
#   nvib
#
# If no prefix is given:
# baseline.pt
# baseline_correct.pt
#
# If prefix=nvib:
# nvib_baseline.pt
# nvib_baseline_correct.pt
# =========================================================

# -----------------------------
# Input arguments
# -----------------------------
CHECKPOINT_PATH=$1
OUTPUT_DIR=$2
NAME_PREFIX=$3

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage:"
    echo "sbatch -p gpu -A balm main_idiap.sh <checkpoint_path> <output_dir> [name_prefix]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# -----------------------------
# Naming logic
# -----------------------------
if [ -n "$NAME_PREFIX" ]; then
    PREFIX="${NAME_PREFIX}_"
else
    PREFIX=""
fi

# -----------------------------
# Derived paths
# -----------------------------
BASE_SAMPLES="${OUTPUT_DIR}/${PREFIX}baseline.pt"
BASE_METRICS="${OUTPUT_DIR}/${PREFIX}baseline.json"

CORRECT_SAMPLES="${OUTPUT_DIR}/${PREFIX}baseline_correct.pt"
CORRECT_METRICS="${OUTPUT_DIR}/${PREFIX}baseline_correct.json"

CORRECT_N12_SAMPLES="${OUTPUT_DIR}/${PREFIX}baseline_correct_N12.pt"
CORRECT_N12_METRICS="${OUTPUT_DIR}/${PREFIX}baseline_correct_N12.json"


CORRECT_SAMPLES_NVIB_NOISE="${OUTPUT_DIR}/${PREFIX}baseline_correct_nvib_noise.pt"
CORRECT_METRICS_NVIB_NOISE="${OUTPUT_DIR}/${PREFIX}baseline_correct_nvib_noise.json"


CORRECT_512_SAMPLES="${OUTPUT_DIR}/${PREFIX}baseline_correct_512.pt"
CORRECT_512_METRICS="${OUTPUT_DIR}/${PREFIX}baseline_correct_512.json"

CORRECT_N12_512_SAMPLES="${OUTPUT_DIR}/${PREFIX}baseline_correct_N12_512.pt"
CORRECT_N12_512_METRICS="${OUTPUT_DIR}/${PREFIX}baseline_correct_N12_512.json"

CORRECT_512_NVIB_NOISE="${OUTPUT_DIR}/${PREFIX}baseline_correct_512_nvib_noise.pt"
CORRECT_512_NVIB_NOISE_METRICS="${OUTPUT_DIR}/${PREFIX}baseline_correct_512_nvib_noise.json"

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

# =========================================================
# 1. Generate samples
# =========================================================

python gidd/eval/generate_samples.py \
    path="$CHECKPOINT_PATH" \
    samples_path="$BASE_SAMPLES" \
    num_samples=1024 \
    num_denoising_steps=128 \
    batch_size=16

# =========================================================
# 2. Evaluate baseline
# =========================================================

python gidd/eval/generative_ppl.py \
    samples_path="$BASE_SAMPLES" \
    model_tokenizer=gpt2 \
    pretrained_model=google/gemma-2-9b \
    batch_size=1 \
    metrics_path="$BASE_METRICS"

# =========================================================
# 3. Self correction
# =========================================================

python gidd/eval/self_correction.py \
    path="$CHECKPOINT_PATH" \
    samples_path="$BASE_SAMPLES" \
    corrected_samples_path="$CORRECT_SAMPLES" \
    batch_size=16 \
    num_denoising_steps=128 \
    temp=0.1

python gidd/eval/generative_ppl.py \
    samples_path="$CORRECT_SAMPLES" \
    model_tokenizer=gpt2 \
    pretrained_model=google/gemma-2-9b \
    batch_size=1 \
    metrics_path="$CORRECT_METRICS"

# =========================================================
# 4. Self correction + latent noise
# =========================================================

python gidd/eval/self_correction.py \
    path="$CHECKPOINT_PATH" \
    samples_path="$BASE_SAMPLES" \
    corrected_samples_path="$CORRECT_SAMPLES_NVIB_NOISE" \
    batch_size=16 \
    num_denoising_steps=128 \
    temp=0.1 \
    latent_noise=True

python gidd/eval/generative_ppl.py \
    samples_path="$CORRECT_SAMPLES_NVIB_NOISE" \
    model_tokenizer=gpt2 \
    pretrained_model=google/gemma-2-9b \
    batch_size=1 \
    metrics_path="$CORRECT_METRICS_NVIB_NOISE"


# =========================================================
# 5. Self correction + nvib noise
# =========================================================

python gidd/eval/self_correction.py \
    path="$CHECKPOINT_PATH" \
    samples_path="$BASE_SAMPLES" \
    corrected_samples_path="$CORRECT_N12_SAMPLES" \
    batch_size=16 \
    num_denoising_steps=128 \
    temp=0.1 \
    activate_nvib_noise=True

python gidd/eval/generative_ppl.py \
    samples_path="$CORRECT_N12_SAMPLES" \
    model_tokenizer=gpt2 \
    pretrained_model=google/gemma-2-9b \
    batch_size=1 \
    metrics_path="$CORRECT_N12_METRICS"

# =========================================================
# 6. Self correction 512
# =========================================================

python gidd/eval/self_correction.py \
    path="$CHECKPOINT_PATH" \
    samples_path="$BASE_SAMPLES" \
    corrected_samples_path="$CORRECT_512_SAMPLES" \
    batch_size=16 \
    num_denoising_steps=512 \
    temp=0.1

python gidd/eval/generative_ppl.py \
    samples_path="$CORRECT_512_SAMPLES" \
    model_tokenizer=gpt2 \
    pretrained_model=google/gemma-2-9b \
    batch_size=1 \
    metrics_path="$CORRECT_512_METRICS"

# =========================================================
# 7. Self correction 512 + latent noise
# =========================================================

python gidd/eval/self_correction.py \
    path="$CHECKPOINT_PATH" \
    samples_path="$BASE_SAMPLES" \
    corrected_samples_path="$CORRECT_N12_512_SAMPLES" \
    batch_size=16 \
    num_denoising_steps=512 \
    temp=0.1 \
    latent_noise=True

python gidd/eval/generative_ppl.py \
    samples_path="$CORRECT_N12_512_SAMPLES" \
    model_tokenizer=gpt2 \
    pretrained_model=google/gemma-2-9b \
    batch_size=1 \
    metrics_path="$CORRECT_N12_512_METRICS"


# =========================================================
# 8. Self correction 512 + nvib noise
# =========================================================

python gidd/eval/self_correction.py \
    path="$CHECKPOINT_PATH" \
    samples_path="$BASE_SAMPLES" \
    corrected_samples_path="$CORRECT_512_NVIB_NOISE" \
    batch_size=16 \
    num_denoising_steps=512 \
    temp=0.1 \
    activate_nvib_noise=True

python gidd/eval/generative_ppl.py \
    samples_path="$CORRECT_512_NVIB_NOISE" \
    model_tokenizer=gpt2 \
    pretrained_model=google/gemma-2-9b \
    batch_size=1 \
    metrics_path="$CORRECT_512_NVIB_NOISE_METRICS"


echo "================================"
echo "All evaluations finished."
echo "Results saved to:"
echo "$OUTPUT_DIR"
echo "================================"