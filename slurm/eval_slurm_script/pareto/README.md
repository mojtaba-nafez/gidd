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


```bash
/idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd_interactive.sh \
    --checkpoint "/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small" \
    --output-dir "/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd" \
    > /idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd/log-gidd.log 2>&1
```

/idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd_interactive.sh \
    --checkpoint "/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small" \
    --output-dir "/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd" \
    > /idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd/log-gidd.log 2>&1

/idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd_interactive.sh \
    --checkpoint "/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib" \
    --output-dir "/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd-nvib" \
    > /idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd-nvib/log-gidd-nvib.log 2>&1


srun -J gidd-nvib -p gpu -A balm \
    -t 08:00:00 \
    --gres=gpu:h100:1 \
    --mem=80GB \
    --cpus-per-task=8 \
    bash -lc '
        eval "$(conda shell.bash hook)"
        conda activate gidd
        cd /idiap/temp/mnafez/research/gidd/

        /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd_interactive.sh \
            --checkpoint "/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib" \
            --output-dir "/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd-nvib"
    ' > /idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd-nvib/log-gidd-nvib2.log 2>&1



====

srun -J gidd-nvib -p gpu -A balm \
    -t 08:00:00 \
    --gres=gpu:h100:1 \
    --mem=80GB \
    --cpus-per-task=8 \
    bash -lc '
        eval "$(conda shell.bash hook)"
        conda activate gidd
        cd /idiap/temp/mnafez/research/gidd/

        /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd_interactive.sh \
            --checkpoint "/idiap/temp/mnafez/research/gidd/our-pt-checkpoints/cscs-trained/pt-p-0.2-small-nvib-3-4-5-6-7-8" --activate-nvib-noise \
            --output-dir "/idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd-nvib-3-4-5-6-7-8" > /idiap/temp/mnafez/research/gidd/gemma_metrics/pareto-experiments/gidd-nvib-3-4-5-6-7-8/log-gidd-nvib.log 2>&1
    '


sbatch -p gpu -A balm /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/ar.sh


sbatch /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/gidd_sbatch.sh


sbatch /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/mauve/sbatch.sh

sbatch \
  --partition=cpu \
  --gres=gpu:0 \
  --mem=80G \
  --cpus-per-task=16 \
  --time=40:00:00 \
  /idiap/temp/mnafez/research/gidd/slurm/eval_slurm_script/pareto/mauve/sbatch.sh