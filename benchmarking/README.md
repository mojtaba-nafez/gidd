
# Commands (Their Checkpoints)

```
sbatch benchmarking/run_lmeval.sh hellaswag ./weights/gidd-base-pu-0.2
sbatch benchmarking/run_lmeval.sh arc_easy ./weights/gidd-base-pu-0.2
sbatch benchmarking/run_lmeval.sh arc_challenge ./weights/gidd-base-pu-0.2
sbatch benchmarking/run_lmeval.sh boolq ./weights/gidd-base-pu-0.2
sbatch benchmarking/run_lmeval.sh piqa ./weights/gidd-base-pu-0.2
sbatch benchmarking/run_lmeval.sh winogrande ./weights/gidd-base-pu-0.2
sbatch benchmarking/run_lmeval.sh AraDiCE_openbookqa_eng ./weights/gidd-base-pu-0.2
```




# Commands (My Checkpoints)

```
sbatch --time=02:00:00 --gres=gpu:rtx3090:1 benchmarking/run_lmeval.sh arc_easy ./our-pt-checkpoints/pt-p-0.2-small
sbatch --time=01:00:00 benchmarking/run_lmeval.sh arc_challenge ./our-pt-checkpoints/pt-p-0.2-small
sbatch --time=01:30:00 benchmarking/run_lmeval.sh boolq ./our-pt-checkpoints/pt-p-0.2-small
sbatch --time=01:00:00 benchmarking/run_lmeval.sh piqa ./our-pt-checkpoints/pt-p-0.2-small
sbatch --time=01:00:00 benchmarking/run_lmeval.sh winogrande ./our-pt-checkpoints/pt-p-0.2-small
sbatch --time=01:00:00 benchmarking/run_lmeval.sh AraDiCE_openbookqa_eng ./our-pt-checkpoints/pt-p-0.2-small
sbatch --time=8:00:00 benchmarking/run_lmeval.sh hellaswag ./our-pt-checkpoints/pt-p-0.2-small
```


# Commands (My NVIB Checkpoints)

```
sbatch --time=02:00:00 --gres=gpu:rtx3090:1 benchmarking/run_lmeval.sh arc_easy ./our-pt-checkpoints/pt-p-0.2-small-nvib
sbatch --time=01:00:00 --gres=gpu:rtx3090:1 benchmarking/run_lmeval.sh arc_challenge ./our-pt-checkpoints/pt-p-0.2-small-nvib
sbatch --time=01:30:00 --gres=gpu:rtx3090:1 benchmarking/run_lmeval.sh boolq ./our-pt-checkpoints/pt-p-0.2-small-nvib
sbatch --time=01:00:00 --gres=gpu:rtx3090:1 benchmarking/run_lmeval.sh piqa ./our-pt-checkpoints/pt-p-0.2-small-nvib
sbatch --time=01:00:00 --gres=gpu:rtx3090:1 benchmarking/run_lmeval.sh winogrande ./our-pt-checkpoints/pt-p-0.2-small-nvib
sbatch --time=01:00:00 --gres=gpu:rtx3090:1 benchmarking/run_lmeval.sh AraDiCE_openbookqa_eng ./our-pt-checkpoints/pt-p-0.2-small-nvib
sbatch --time=8:00:00 --gres=gpu:rtx3090:1 benchmarking/run_lmeval.sh hellaswag ./our-pt-checkpoints/pt-p-0.2-small-nvib
```

 