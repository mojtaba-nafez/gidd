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

echo "======= Conda and CUDA ======="
module load CUDA
source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate gidd

echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
echo "================================"

python gidd/eval/get_ppl_training_data.py model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=training_dataset.json +training.train_batch_size=1 +training.eval_batch_size=1 +data.test_size=100000 +data.dataset_name=Skylion007/openwebtext +data.dataset_subset=null +data.trust_remote_code=true +data.tokenizer_name=gpt2 +model.max_seq_len=512 +data.sequence_packing=false +data.max_add_padding=0 +data.cache_dir=./cache +data.num_workers=8
