#!/usr/bin/env bash
#SBATCH --job-name=gidd_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=0-13:00:00
#SBATCH --output=logs-eval-slurm/%x-%j.out
#SBATCH --error=logs-eval-slurm/%x-%j.err
#SBATCH --requeue

set -e

mkdir -p logs-eval-slurm

echo "======= Conda and CUDA ======="

module load CUDA

source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate gidd

echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
echo "================================"

# =========================================================
# AR 181M (our checkpoints)
# =========================================================

temps=(0.7 0.75 0.8 0.85 0.9 0.95 1.0)

for temp in "${temps[@]}"; do
    echo "======= Running temperature: $temp ======="

    python gidd/eval/generate_samples.py \
        path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/ar-baseline-small-181M" \
        samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/ar/samples${temp}.pt" \
        num_samples=1024 \
        num_denoising_steps=512 \
        batch_size=16 \
        +temperature=$temp

    python gidd/eval/generative_ppl.py \
        samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/ar/samples${temp}.pt" \
        model_tokenizer=gpt2 \
        pretrained_model=google/gemma-2-9b \
        batch_size=1 \
        metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/ar/samples${temp}.json"
done