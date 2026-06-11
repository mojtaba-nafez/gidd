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


python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/ar-baseline-small-181M" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-baseline-181M/samples.pt" num_samples=1024 num_denoising_steps=512 batch_size=16

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-baseline-181M/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-baseline-181M/samples.json"



python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/weights/mdlm-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples.pt" num_samples=1024 num_denoising_steps=128 batch_size=16

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples.json"



python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/weights/mdlm-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples_512.pt" num_samples=1024 num_denoising_steps=512 batch_size=16

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples_512.json"



python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/weights/ar-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.pt" num_samples=1024 num_denoising_steps=512 batch_size=16

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.json"

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=gpt2-large batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples_gpt2_large.json"


echo "================================"
echo "All evaluations finished."
echo "Results saved to:"
echo "$OUTPUT_DIR"
echo "================================"