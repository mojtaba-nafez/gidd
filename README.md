# CSCS-Base Document

# Training:

## Intractive Session Training:

```
srun --environment=gidd  -A go082  --pty bash
source ~/miniconda3/bin/activate
conda activate gidd
```

```
torchrun --nnodes 1 --nproc_per_node 4 --master_port 29501 gidd/train.py --config-name gidd model.p_uniform=0.2 model.nvib_layers=[4,6,8] logging.run_name="'small-nvib-gidd+-owt-pu=0.2'" model=small training.num_train_steps=125000
```

## Baseline Replication (same parameter numbers)

```
sbatch --environment=gidd  -A go082 slurm/training_scripts_slurm/main_cscs.sh \
  model.p_uniform=0.2 \
  model.nvib_layers=[] \
  logging.run_name=small-nvib-gidd+-owt-pu0.2 model=small \
  training.num_train_steps=125000 model=small
```

## NVIB Training

```
sbatch --environment=gidd  -A go082 slurm/training_scripts_slurm/main_cscs.sh \
  model.p_uniform=0.2 \
  model.nvib_layers=[4,6,8] \
  logging.run_name=small-nvib-gidd+-owt-pu0.2 model=small \
  training.num_train_steps=125000 model=small
```



# Evaluation

### Baseline: original checkpoint

```
python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/weights/gidd-base-pu-0.2" samples_path=samples.pt num_samples=16 num_denoising_steps=128 batch_size=16

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=samples.json
```



### Baseline: Our checkpoint

```
python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" num_samples=1024 num_denoising_steps=128 batch_size=16


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.json"


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct.pt" batch_size=16 num_denoising_steps=128 temp=0.5


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct.json


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12.pt" batch_size=16 num_denoising_steps=128 temp=0.5


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12.json
```

##### 512 step

```
python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.pt" batch_size=16 num_denoising_steps=512 temp=0.5


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.json


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.pt" batch_size=16 num_denoising_steps=512 temp=0.5 latent_noise=True


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.json
```




### NVIB: Our checkpoint

```
python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" num_samples=1024 num_denoising_steps=128 batch_size=16


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.json"


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct.pt" batch_size=16 num_denoising_steps=128 temp=0.1


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct.json



python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12.pt" batch_size=16 num_denoising_steps=128 temp=0.1


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12.json
```


##### 512 step

```
python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.pt" batch_size=16 num_denoising_steps=512 temp=0.1


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.json


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.pt" batch_size=16 num_denoising_steps=512 temp=0.1 latent_noise=True


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.json
```



### NVIB: Our checkpoint  -- Temporal Runs


