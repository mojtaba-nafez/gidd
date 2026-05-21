#!/bin/bash
#SBATCH --job-name=lmeval
#SBATCH -A balm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00

set -e

TASK_NAME="$1"
MODEL_PATH="$2"

if [ -z "$TASK_NAME" ] || [ -z "$MODEL_PATH" ]; then
  echo "Usage: sbatch run_lmeval.sh <task_name> <model_path>"
  echo "Example: sbatch run_lmeval.sh hellaswag ./weights/model"
  exit 1
fi

mkdir -p logs

exec > "logs/${TASK_NAME}-${SLURM_JOB_ID}.out" 2>&1

echo "Running task: $TASK_NAME"
echo "Model path: $MODEL_PATH"

echo "=======Conda and CUDA=========="
module load CUDA
source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate gidd
echo "Conda activated: $(which python)"
echo "CUDA_HOME is: $CUDA_HOME"
echo "NVCC is at: $(which nvcc)"
echo "============================"

python -m lm_eval \
  --model gidd \
  --tasks "$TASK_NAME" \
  --model_args "model_path=${MODEL_PATH},num_denoising_steps=128"