# Training:

## Intractive Session Training:


```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 29501 gidd/train.py --config-name gidd model.p_uniform=0.2 model.nvib_layers=[4,6,8] logging.run_name="'small-nvib-gidd+-owt-pu=0.2'" training.train_batch_size=8 model=small
```


## Baseline Replication (same parameter numbers)

```
sbatch -p gpu -A balm --gres=gpu:1 slurm/training_scripts_slurm/main_idiap.sh --gpus 1 \
  model.p_uniform=0.2 \
  model.nvib_layers=[] \
  logging.run_name=small-nvib-gidd+-owt-pu0.2
```

## NVIB Training
```
sbatch -p gpu -A balm --gres=gpu:h100:1 /idiap/temp/mnafez/research/gidd/slurm/training_scripts_slurm/main_idiap.sh --gpus 1 \
  model.p_uniform=0.2 \
  model.nvib_layers=[3,4,5,6,7,8] \
  logging.run_name=small-nvib-gidd+-owt-pu0.2 model=small
```



# Evaluation


## Baseline: original checkpoint

### sbatch

```
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/main_idiap.sh /idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small /idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs
```
  
```
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/main_idiap.sh \
/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib-3-4-5-6-7-8 \
/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs
```


### srun
```
python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/weights/gidd-base-pu-0.2" samples_path=samples.pt num_samples=16 num_denoising_steps=128 batch_size=16

python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=samples.json
```



## Baseline: Our checkpoint

```
python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" num_samples=1024 num_denoising_steps=128 batch_size=16


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.json"


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct.pt" batch_size=16 num_denoising_steps=128 temp=0.5


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct.json


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12.pt" batch_size=16 num_denoising_steps=128 temp=0.5


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12.json
```

### 512 step

```
python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.pt" batch_size=16 num_denoising_steps=512 temp=0.5


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_512.json


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.pt" batch_size=16 num_denoising_steps=512 temp=0.5 latent_noise=True


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline_correct_N12_512.json
```




## NVIB: Our checkpoint

```
python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" num_samples=1024 num_denoising_steps=128 batch_size=16


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.json"


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct.pt" batch_size=16 num_denoising_steps=128 temp=0.1


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct.json



python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12.pt" batch_size=16 num_denoising_steps=128 temp=0.1


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12.json
```


### 512 step

```
python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.pt" batch_size=16 num_denoising_steps=512 temp=0.1


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_512.json


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.pt" batch_size=16 num_denoising_steps=512 temp=0.1 latent_noise=True


python gidd/eval/generative_ppl.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib/nvib_baseline_correct_N12_512.json
```



### NVIB: Our checkpoint  -- Temporal Runs


```
python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib-3-4-5-6-7-8" samples_path="nvib_baseline.pt" num_samples=32 num_denoising_steps=128 batch_size=16


python gidd/eval/generative_ppl.py samples_path="nvib_baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path="nvib_baseline.json"


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib-3-4-5-6-7-8"  samples_path="nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/nvib_baseline_correct.pt" batch_size=16 num_denoising_steps=128 temp=0.1 

python gidd/eval/generative_ppl.py samples_path="nvib_baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=nvib_baseline_correct.json
```





```
python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib" samples_path="nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/nvib_baseline_correct_N12.pt" batch_size=16 num_denoising_steps=128 temp=0.1 latent_noise=True

python gidd/eval/generative_ppl.py samples_path="nvib_baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=nvib_baseline_correct_N12.json 

```






```
python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/pt-p-0.2-small-nvib" samples_path="nvib_baseline.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/nvib_baseline_correct.pt" batch_size=16 num_denoising_steps=128 temp=0.1 latent_noise=True

python gidd/eval/generative_ppl.py samples_path="nvib_baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=nvib_baseline_correct.json

```





```
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/main_idiap.sh \
/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib \
/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib_cscs nvib



sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/main_idiap_3.sh \
/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib \
/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_nvib_cscs_scale-0.04_samplingscale0.2_nvib_noise


python gidd/eval/get_ppl_training_data.py model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=training_dataset.json +training.train_batch_size=1 +training.eval_batch_size=1 +data.test_size=100000 +data.dataset_name=Skylion007/openwebtext +data.dataset_subset=null +data.trust_remote_code=true +data.tokenizer_name=gpt2 +model.max_seq_len=512 +data.sequence_packing=false +data.max_add_padding=0 +data.cache_dir=./cache +data.num_workers=8


sbatch -p gpu -A balm --gres=gpu:0 /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/temp.sh 

```

python gidd/eval/generate_samples.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-13-block" samples_path="samples.pt" num_samples=32 num_denoising_steps=128 batch_size=16

python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-13-block" samples_path="samples.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples.pt" batch_size=16 num_denoising_steps=128 temp=0.1 latent_noise=True

python gidd/eval/generative_ppl.py samples_path="corrected_samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json 


python gidd/eval/self_correction.py path="/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-13-block" samples_path="samples.pt" corrected_samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples.pt" batch_size=16 num_denoising_steps=128 temp=0.1 latent_noise=False

python gidd/eval/generative_ppl.py samples_path="corrected_samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json 
