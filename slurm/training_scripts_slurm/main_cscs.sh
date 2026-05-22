#!/usr/bin/env bash
#SBATCH --job-name=gidd_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=normal
#SBATCH --time=0-08:00:00
#SBATCH --output=logs-train-slurm/%x-%j.out
#SBATCH --error=logs-train-slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

############################
# DEFAULTS
############################
GPUS=${GPUS:-4}
CONFIG_NAME=${CONFIG_NAME:-gidd}

############################
# OPTIONAL ARG PARSING
############################
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

############################
# ENV SETUP
############################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gidd

echo "==================="
echo "Python: $(which python)"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "==================="

############################
# FIX WORKING DIRECTORY (IMPORTANT)
############################

echo "Before cd:"
echo "PWD=$(pwd)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"

cd /mnt/home/NLU/gidd
# cd "${SLURM_SUBMIT_DIR}"

echo "After cd:"
pwd
ls -l gidd/train.py || true

############################
# MASTER PORT (avoid collisions)
############################
export MASTER_PORT=$((29500 + RANDOM % 1000))

echo "==================="
echo "HOSTNAME: $(hostname)"
echo "WHOAMI: $(whoami)"
echo "GPUs: $GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_PORT: $MASTER_PORT"
echo "==================="

echo "Running nvidia-smi..."
nvidia-smi

############################
# PYTHON PATH SAFETY (optional but recommended)
############################
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
############################
# RESOLVE ENTRY FILE SAFELY
############################

if [[ -f "train.py" ]]; then
  ट्रेन_ENTRY="train.py"
elif [[ -f "gidd/train.py" ]]; then
  TRAIN_ENTRY="gidd/train.py"
else
  echo "ERROR: Cannot find train.py"
  find . -maxdepth 3 -name "train.py"
  exit 1
fi

echo "Using training entry: $TRAIN_ENTRY"

############################
# RUN
############################

echo "Running torchrun..."

torchrun \
  --nnodes=1 \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" \
  "$TRAIN_ENTRY" \
  --config-name "$CONFIG_NAME" \
  "$@"