# Eval


```
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/baseline_eval-idiap-2.sh \
    /idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-13-block \
    /idiap/temp/mnafez/research/gidd/gemma_metrics/our_pt_baseline_13block_cscs
```



```
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/baseline_eval-idiap.sh \
    /idiap/temp/mnafez/research/gidd/weights/gidd-small-pu-0.2 \
    /idiap/temp/mnafez/research/gidd/gemma_metrics/gidd-small-pu-0.2-original-checkpoint
```


```
sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/ar_idiap.sh
```