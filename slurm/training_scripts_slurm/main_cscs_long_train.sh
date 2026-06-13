#!/usr/bin/env bash
#SBATCH --job-name=gidd_training
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --time=0-12:00:00
#SBATCH --output=logs-train-slurm/%x-%j.out
#SBATCH --error=logs-train-slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

############################
# DEFAULTS
############################
GPUs_PER_NODE=${GPUs_PER_NODE:-8}
NODES=${NODES:-2}
CONFIG_NAME=${CONFIG_NAME:-gidd}

############################
# OPTIONAL ARG PARSING
############################
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus_per_node)
      GPUs_PER_NODE="$2"
      shift 2
      ;;
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --nodes)
      NODES="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

############################
# ENV SETUP
############################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gidd

echo "==================="
echo "Python: $(which python)"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "==================="

############################
# FIX WORKING DIRECTORY (IMPORTANT)
############################

echo "Before cd:"
echo "PWD=$(pwd)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"

cd /mnt/home/NLU/gidd
# cd "${SLURM_SUBMIT_DIR}"

echo "After cd:"
pwd
ls -l gidd/train.py || true

############################
# MASTER PORT (avoid collisions)
############################
# export MASTER_PORT=$((29500 + RANDOM % 1000))
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "==================="
echo "HOSTNAME: $(hostname)"
echo "WHOAMI: $(whoami)"
echo "GPUs_PER_NODE: $GPUs_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "MASTER_PORT: $MASTER_PORT"
echo "==================="

echo "Running nvidia-smi..."
nvidia-smi

##########################
# Debug #
##########################
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NODEID=${SLURM_NODEID:-unset}"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

scontrol show hostnames "$SLURM_JOB_NODELIST"

echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_NODEID=$SLURM_NODEID"
hostname

# Use the HSN interface for inter-node communication
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# Explicitly use the high-speed network hostname
MASTER_ADDR_HSN="${MASTER_ADDR}-hsn1"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$SLURM_NODEID"

############################
# PYTHON PATH SAFETY (optional but recommended)
############################
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
############################
# RESOLVE ENTRY FILE SAFELY
############################

if [[ -f "train.py" ]]; then
  TRAIN_ENTRY="train.py"
elif [[ -f "gidd/train.py" ]]; then
  TRAIN_ENTRY="gidd/train.py"
else
  echo "ERROR: Cannot find train.py"
  find . -maxdepth 3 -name "train.py"
  exit 1
fi

echo "Using training entry: $TRAIN_ENTRY"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "@: $@"
############################
# RUN
############################

echo "Running torchrun..."

# Resolve master IP directly from the HSN interface to avoid DNS issues
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR_HSN="${MASTER_ADDR}-hsn1"

# Resolve to IP to bypass any DNS ambiguity
MASTER_IP=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_ADDR_HSN}'))" 2>/dev/null \
            || getent hosts "${MASTER_ADDR_HSN}" | awk '{print $1}')
echo "MASTER_IP=$MASTER_IP"

HSN_IFACE=$(awk 'NR>2 {gsub(/:.*/, "", $1); if ($1 ~ /^hsn/) print $1}' /proc/net/dev | head -n1)
echo "HSN_IFACE: $HSN_IFACE"

export GLOO_SOCKET_IFNAME=$HSN_IFACE
export NCCL_SOCKET_IFNAME=$HSN_IFACE

echo "=== CONTAINER ENV CHECK ==="
srun --ntasks=$NODES bash -c '
  echo "--- Node: $(hostname) ---"
  echo "PWD: $(pwd)"
  echo "ls /mnt/home/NLU/gidd:"
  ls /mnt/home/NLU/gidd 2>/dev/null || echo "MISSING /mnt/home/NLU/gidd"
  echo "which python: $(which python 2>/dev/null || echo MISSING)"
  echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
  echo "cat /proc/net/dev interfaces:"
  awk "NR>2 {gsub(/:.*/, \"\", \$1); print \$1}" /proc/net/dev
'
echo "==========================="

# Capture all remaining args before the srun call
HYDRA_ARGS="$@"
echo "HYDRA_ARGS: $HYDRA_ARGS"

srun bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate gidd
  cd /mnt/home/NLU/gidd
  echo "Node: $(hostname), Python: $(which python), torchrun: $(which torchrun)"
  torchrun \
    --nnodes='"$NODES"' \
    --nproc_per_node='"$GPUs_PER_NODE"' \
    --node_rank=$SLURM_NODEID \
    --master_addr='"$MASTER_IP"' \
    --master_port='"$MASTER_PORT"' \
    --rdzv_backend=static \
    gidd/train.py \
    --config-name '"$CONFIG_NAME"' \
    '"$HYDRA_ARGS"'
'