#!/usr/bin/env bash
#SBATCH --job-name=gidd_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=0-30:00:00
#SBATCH --output=logs-train-slurm/%x-%j.out
#SBATCH --error=logs-train-slurm/%x-%j.err
#SBATCH --requeue

set -e

echo "======= Conda and CUDA ======="
module load CUDA

source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate gidd

echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
echo "================================"

# Avoid port conflicts
export MASTER_PORT=$((29500 + RANDOM % 1000))

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port=$MASTER_PORT \
  gidd/train.py \
  --config-name gidd \
  model.p_uniform=0.2 \
  logging.run_name=small-gidd-owt-pu0.2