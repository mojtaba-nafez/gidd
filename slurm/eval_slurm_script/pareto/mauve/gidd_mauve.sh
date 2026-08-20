#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_PATH="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small"
OUTPUT_DIR="/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd"
PROJECT_DIR="/idiap/temp/mnafez/research/gidd"
NAME_PREFIX=""

TEMP=0.1
LATENT_NOISE=False
ACTIVATE_NVIB_NOISE=False

usage() {
    echo "Usage: bash $0 [options]"
    echo
    echo "Options:"
    echo "  --checkpoint PATH"
    echo "  --output-dir DIR"
    echo "  --name-prefix NAME"
    echo "  --temp VALUE              Temperature (default: 0.1)"
    echo "  --latent-noise"
    echo "  --activate-nvib-noise"
    echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT_PATH="${2:?Missing value for --checkpoint}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?Missing value for --output-dir}"
            shift 2
            ;;
        --name-prefix)
            NAME_PREFIX="${2:?Missing value for --name-prefix}"
            shift 2
            ;;
        --temp)
            TEMP="${2:?Missing value for --temp}"
            shift 2
            ;;
        --latent-noise)
            LATENT_NOISE=True
            shift
            ;;
        --activate-nvib-noise)
            ACTIVATE_NVIB_NOISE=True
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

CHECKPOINT_PATH="${CHECKPOINT_PATH%/}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
NAME_PREFIX="${NAME_PREFIX:-$(basename "$CHECKPOINT_PATH")}"

[[ -e "$CHECKPOINT_PATH" ]] || {
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
}

[[ -d "$PROJECT_DIR" ]] || {
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "======= Conda and CUDA ======="

module load CUDA
source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate gidd

echo "Python: $(command -v python)"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "NVCC: $(command -v nvcc || echo 'not found')"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Name prefix: $NAME_PREFIX"
echo "Temperature: $TEMP"
echo "Latent noise: $LATENT_NOISE"
echo "NVIB noise: $ACTIVATE_NVIB_NOISE"
echo "================================"

# nvidia-smi

CHECKPOINT_NAME="$NAME_PREFIX"
SAMPLES_PATH="${OUTPUT_DIR}/base_samples_${CHECKPOINT_NAME}.pt"

NOISE_SUFFIX=""
[[ "$LATENT_NOISE" == True ]] && NOISE_SUFFIX+="_latent"
[[ "$ACTIVATE_NVIB_NOISE" == True ]] && NOISE_SUFFIX+="_nvib"

echo "Generating base samples..."

# python gidd/eval/generate_samples.py \
#     path="$CHECKPOINT_PATH" \
#     samples_path="$SAMPLES_PATH" \
#     num_samples=5000 \
#     num_denoising_steps=128 \
#     batch_size=32

DENOISING_STEPS=(32 64 128 256 384 512)

for NUM_DENOISING_STEPS in "${DENOISING_STEPS[@]}"; do
    CORRECTED_PATH="${OUTPUT_DIR}/base_correct_${CHECKPOINT_NAME}_tmp-${TEMP}_step${NUM_DENOISING_STEPS}${NOISE_SUFFIX}.pt"
    METRICS_PATH="${OUTPUT_DIR}/samples_${CHECKPOINT_NAME}_tmp-${TEMP}_step${NUM_DENOISING_STEPS}${NOISE_SUFFIX}.json"

    echo "================================================"
    echo "Temperature: $TEMP"
    echo "Denoising steps: $NUM_DENOISING_STEPS"
    echo "Corrected samples: $CORRECTED_PATH"
    echo "================================================"

    # python gidd/eval/self_correction.py \
    #     path="$CHECKPOINT_PATH" \
    #     samples_path="$SAMPLES_PATH" \
    #     corrected_samples_path="$CORRECTED_PATH" \
    #     batch_size=32 \
    #     num_denoising_steps="$NUM_DENOISING_STEPS" \
    #     temp="$TEMP" \
    #     latent_noise="$LATENT_NOISE" \
    #     activate_nvib_noise="$ACTIVATE_NVIB_NOISE"

    python mauve_compute.py \
        samples_path="$CORRECTED_PATH" \
        model_tokenizer=gpt2 \
        pretrained_model=google/gemma-2-9b \
        batch_size=8 \
        metrics_path="$METRICS_PATH"

    echo "Completed: temp=$TEMP, steps=$NUM_DENOISING_STEPS"
done

echo "All evaluations completed successfully."