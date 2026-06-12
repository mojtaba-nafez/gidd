

### Baseline + Sample Generation Step
```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10
```

### Baseline + 128 setp + Self-Correction

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10

```

### NVIB-3-4-5-6-7-8 + Sample Generation Step

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10
```




### NVIB-3-4-5-6-7-8 + Self-Correction Step

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10

```


### Original GIDD Checkpoint + Sample Generation Setp

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/samples_1024_original.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/samples_1024_original.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/samples_1024_original.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/samples_1024_original.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/samples_1024_original.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10
```


### Original GIDD Checkpoint + Self-Correction 

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/corrected_samples_original_temp0-5.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/corrected_samples_original_temp0-5.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/corrected_samples_original_temp0-5.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/corrected_samples_original_temp0-5.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/corrected_samples_original_base_checkpoint/corrected_samples_original_temp0-5.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10
```


### Original GIDD Architecture + Our CSCS Replication  + Sample Generation Setp

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10
```


### Original GIDD Architecture + Our CSCS Replication  + Self.Correction Setp

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10

```


### Original GIDD Architecture + Idiap Our Replication  + Sample Generation Setp

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10
```



### NVIB-3-4-5-6-7-8 + Self-Correction Step NVIB NOISE + 128

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline_correct_N12.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10

```



### Original GIDD Checkpoint + Small Model + Sample Generation Setp

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10




### Original GIDD Checkpoint + Small Model + Self-Correction Setp

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline_correct.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10



### MDLM + Small Model + Official Checkpoint

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/mdlm-small-official-checkpoint-gidd/samples_512.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10



### AR + Small Model + Official Checkpoint

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=450 +entropy_sample_len_up_threshold=530


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=450


python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=10 +entropy_sample_len_up_threshold=100

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/ar-small-official-checkpoint-gidd/samples.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=10


================

### Debug

##### gidd small official checkpoint + sample gen step

python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


##### gidd small our nvib-3-4-5-6-7-8 + sample gen step

 python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/nvib-3-4-5-6-7-8-cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530


#####  gidd + our cscs replication  + sample gen step

```
python gidd/eval/entropy-analysis.py samples_path="/idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_cscs/baseline.pt" model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=1 metrics_path=corrected_samples.json +entropy_sample_len_low_threshold=0 +entropy_sample_len_up_threshold=530

