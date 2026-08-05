# Pareto Plot Experiments

```bash
sbatch -p gpu -A balm \
  /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd.sh \
  --checkpoint "/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small" \
  --output-dir "/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd"
```


```bash
sbatch -p gpu -A balm --gres=gpu:h100:1 \
  /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd.sh \
  --checkpoint "/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small" \
  --output-dir "/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd"
```


