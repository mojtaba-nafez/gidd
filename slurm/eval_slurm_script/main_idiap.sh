#!/usr/bin/env bash
#SBATCH --job-name=gidd_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=0-10:00:00
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


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.pt" batch_size=16 num_denoising_steps=512 temp=0.5

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.json

python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.pt" batch_size=16 num_denoising_steps=512 temp=0.5 latent_noise=True

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.json

python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.pt" batch_size=16 num_denoising_steps=512 temp=0.1

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.json

python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.pt" batch_size=16 num_denoising_steps=512 temp=0.1 latent_noise=True

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.json
