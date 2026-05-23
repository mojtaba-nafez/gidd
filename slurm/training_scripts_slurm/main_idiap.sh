#!/usr/bin/env bash
#SBATCH --job-name=gidd_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=0-35:00:00
#SBATCH --output=logs-train-slurm/%x-%j.out
#SBATCH --error=logs-train-slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

############################
# DEFAULTS
############################
GPUS=${GPUS:-1}
CONFIG_NAME=${CONFIG_NAME:-gidd}

############################
# OPTIONAL ARG PARSING
# (only for clean overrides)
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
echo "======= ENV ======="
module load CUDA

source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate gidd

echo "Python: $(which python)"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "GPUs: $GPUS"
echo "==================="

############################
# MASTER PORT (avoid collisions)
############################
export MASTER_PORT=$((29500 + RANDOM % 1000))

############################
# RUN
############################
echo "Running torchrun..."

torchrun \
  --nnodes=1 \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" \
  gidd/train.py \
  --config-name "$CONFIG_NAME" \
  "$@"