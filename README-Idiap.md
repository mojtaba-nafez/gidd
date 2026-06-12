# Training:

## Intractive Session Training:

```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 29501 gidd/train.py --config-name gidd model.p_uniform=0.2 model.nvib_layers=[4,6,8] logging.run_name="'small-nvib-gidd+-owt-pu=0.2'" training.train_batch_size=8 model=small
```


## Sbatch Job Sumbission

### Baseline Replication (same parameter numbers)

```
sbatch -p gpu -A balm --gres=gpu:1 slurm/training_scripts_slurm/main_idiap.sh --gpus 1 \
  model.p_uniform=0.2 \
  model.nvib_layers=[] \
  logging.run_name=small-nvib-gidd+-owt-pu0.2
```

### NVIB Training
```
sbatch -p gpu -A balm --gres=gpu:h100:1 /idiap/temp/mnafez/research/gidd/slurm/training_scripts_slurm/main_idiap.sh --gpus 1 \
  model.p_uniform=0.2 \
  model.nvib_layers=[3,4,5,6,7,8] \
  logging.run_name=small-nvib-gidd+-owt-pu0.2 model=small
```

# Evaluation


### Sbatch Job Sumbission

```
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/main_idiap.sh /idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small /idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs
```
  
```
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/main_idiap.sh \
/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib-3-4-5-6-7-8 \
/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs
```


## Intractive Session Training:

For exact and update detail and arguemnt to pass refer to main_idiap.sh content.
an general example for sample generation, self-correction, ppl calculaiton:

```
python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/weights/gidd-base-pu-0.2" samples_path=samples.pt num_samples=16 num_denoising_steps=128 batch_size=16


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/weights/gidd-base-pu-0.2" samples_path="samples.pt " corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct.pt" batch_size=16 num_denoising_steps=128 temp=0.5  latent_noise=False activate_nvib_noise=False


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=samples.json
```


##### AR

check /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/ar_idiap.sh !





### NVIB: Our checkpoint  -- Temporal Runs


python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib-3-4-5-6-7-8" samples_path=samples.pt num_samples=32 num_denoising_steps=128 batch_size=16


python gidd/eval/generative_ppl.py samples_path="samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=samples.json

 python gidd/eval/entropy-analysis.py samples_path="samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530

----

python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib-3-4-5-6-7-8" samples_path="samples.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/baseline_correct.pt" batch_size=16 num_denoising_steps=128 temp=0.1  latent_noise=False activate_nvib_noise=True

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=baseline_correct.json


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530

====

python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-13-block" samples_path="samples.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/baseline_correct2.pt" batch_size=16 num_denoising_steps=128 temp=0.1  latent_noise=False activate_nvib_noise=False

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/baseline_correct2.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=baseline_correct2.json