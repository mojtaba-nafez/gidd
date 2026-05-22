# How to use this repo on Clariden

Run the commands below to get the repo, and create the data and ouputs directory. The directories are set to be in the scratch directory but linked to the repo directory via a softlink, due to the fact that our personal storage limit is quite small, but scratch directory has a lot of space (but subject to auto-cleaning every 30/90 days, so don't put things there for too long!) We will also use a softlink .cache.

```bash
cd ~
git clone https://github.com/mojtaba-nafez/gidd.git
cd gidd
mkdir -p $SCRATCH/gidd-data/training-data $SCRATCH/gidd-data/outputs
ln -s $SCRATCH/gidd-data/training-data .
ln -s $SCRATCH/gidd-data/outputs .
# below creates a softlink for .cache from $SCRATCH, otherwise we are going to run out of memory very soon.
rm -rf .cache # this clears the cache, normally this is OK, maybe slows things down a bit, but cache will be recovered very quickly.
mkdir -p $SCRATCH/.cache
ln -s $SCRATCH/.cache .
```

## I. Reuse a docker image

### Step 1

You can copy my compiled docker container by running the following (currently gidd.sqsh has been removed!)

```bash
mkdir -p /iopsstor/scratch/cscs/[your_username]/ce-images
lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M \
  $SCRATCH/ce-images
cp /capstor/store/cscs/swissai/a137/fwu/DLM-docker/gidd.sqsh /iopsstor/scratch/cscs/[your_username]/ce-images
```

### Step 2

Put the following in a text file `gidd.toml` located at `~/.edf`. Remember to fill in your username.

```bash
image = "/iopsstor/scratch/cscs/[your_username]/ce-images/gidd.sqsh"

mounts = [
  "/users/[your_username]:/mnt/home",
  "/iopsstor/scratch/cscs/[your_username]:/mnt/scratch",
  "/iopsstor/scratch/cscs/[your_username]:/iopsstor/scratch/cscs/[your_username]"  # for symlink resolution
]

workdir = "/mnt/home/NLU"  # <-- leading slash

[env]
HOME = "/mnt/home"
SCRATCH = "/mnt/scratch"
NCCL_DEBUG = "INFO"
CUDA_CACHE_DISABLE = "1"
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"
MPICH_GPU_SUPPORT_ENABLED = "0"

[annotations]
"com.hooks.aws_ofi_nccl.enabled" = "true"
"com.hooks.aws_ofi_nccl.variant" = "cuda12"
```

My case:

```bash
image = "/iopsstor/scratch/cscs/mojtaba_nafez/ce-images/gidd.sqsh"

mounts = [
  "/users/mojtaba_nafez:/mnt/home",
  "/iopsstor/scratch/cscs/mojtaba_nafez:/mnt/scratch",
  "/iopsstor/scratch/cscs/mojtaba_nafez:/iopsstor/scratch/cscs/mojtaba_nafez"  # for symlink resolution
]

workdir = "/mnt/home/NLU"  # <-- leading slash

[env]
HOME = "/mnt/home"
SCRATCH = "/mnt/scratch"
NCCL_DEBUG = "INFO"
CUDA_CACHE_DISABLE = "1"
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"
MPICH_GPU_SUPPORT_ENABLED = "0"

[annotations]
"com.hooks.aws_ofi_nccl.enabled" = "true"
"com.hooks.aws_ofi_nccl.variant" = "cuda12"
```


## II. GIDD docker image on Clariden

This section is about how I generate the `gidd.sqsh` file. I have already shared with you this file in our project folder `/capstor/store/cscs/swissai/a137/gidd`. `Section I` have already helped you copy it into your own space. Please go to `Section III` first to use take the docker image for a spin. If things breaks down for you, come back to this section to compile the `gidd.sqsh` file for yourself.

I followed the [CSCS documentation here](https://docs.cscs.ch/tutorials/ml/llm-inference/). Some modifications are made.

### Step 1

````bash
cd $SCRATCH
mkdir -p dlm_docker_building
cd dlm_docker_building
````

### Step 2

Put the following in a text file with the name `Dockerfile`:

(make sure that you put the requirements.txt in the same directory as Dockerfile)

```bash
FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG DEBIAN_FRONTEND=noninteractive

# ---- System packages ----
RUN apt-get update && apt-get install -y \
    git git-lfs \
    curl ca-certificates \
    openssh-server openssh-client \
    sudo tmux \
    vim htop \
    zip unzip \
    libx11-6 \
    build-essential \
    cmake ninja-build pkg-config \
 && rm -rf /var/lib/apt/lists/*

# ---- pip tooling ----
RUN python -m pip install --upgrade pip setuptools wheel

# Disable any global pip config/constraints injected by the base image/site
ENV PIP_CONSTRAINT=
ENV PIP_CONFIG_FILE=/dev/null

# ---- Install from requirements (keep NGC torch; build CUDA extensions from git) ----
COPY requirements.txt /tmp/requirements.txt

# Filter out torch + the two CUDA extensions we build from git source
RUN grep -vE '^\s*(torch==|causal-conv1d==|mamba-ssm==)' /tmp/requirements.txt > /tmp/req_no_torch.txt
RUN python -m pip install --no-cache-dir -r /tmp/req_no_torch.txt

# ---- causal-conv1d from source (aarch64 PyPI sdist can be missing csrc files) ----
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git /causal-conv1d && \
    cd /causal-conv1d && \
    git checkout v1.4.0 && \
    python -m pip install --no-cache-dir --no-build-isolation .

# ---- mamba-ssm from source (avoid x86_64-only wheel URLs; ensure submodules) ----
RUN git clone --recursive https://github.com/state-spaces/mamba.git /mamba && \
    cd /mamba && \
    git checkout v1.2.0.post1 && \
    python -m pip install --no-cache-dir --no-build-isolation .

```
requirement.txt of a uni-d2 repo (uni-d2 is unified dllm project that implemeted all dllm in one repo and for gidd i also use that.):
```
# Generic
hydra-core==1.3.2
tqdm==4.67.1
black==25.1.0
termcolor==3.0.1
seaborn==0.13.2
blobfile==3.0.0
fsspec
git-lfs==1.6
jinja2==3.1.5
pyyaml==6.0.2
torchmetrics==1.7.1

# ML
torch==2.7.0
einops==0.8.1
fancy-einsum==0.0.3
numpy==1.26.4
mauve-text==0.4.0
git+https://github.com/EleutherAI/lm-evaluation-harness
sentencepiece==0.2.0
lightning==2.5.1
pandas==2.2.1
scikit-learn==1.4.0
timm==1.0.15
jupyter==1.1.1
ocifs==1.3.2
causal-conv1d==1.4.0
mamba-ssm==1.2.0.post1

# Liger Kernel (optional, for optimized cross-entropy on CUDA)
liger-kernel>=0.4.0

# Huggingface
huggingface-hub==0.30.2
safetensors==0.5.2
tokenizers==0.20.3
evaluate==0.4.3
datasets==3.5.0
transformers==4.45.0
accelerate==1.6.0

# Monitoring / configuration
tensorboard==2.16.2
wandb==0.19.9
omegaconf==2.3.0
rich==13.9.4

# Documentation
mkdocs
mkdocs-material
mkdocstrings[python]
mkdocs-include-markdown-plugin
```
### Step 3

Put the following:

```bash
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs-1.13"
```

to `$HOME/.config/containers/storage.conf`.

### Step 4

```bash
$ mkdir -p $SCRATCH/ce-images
$ lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M \
  $SCRATCH/ce-images
```

### Step 5

```bash
[clariden-lnXXX]$ mkdir -p $SCRATCH/ce-images
[clariden-lnXXX]$ lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M \
  $SCRATCH/ce-images
```

### Step 6

(Inside `$SCRATCH/dlm_docker_building`)

```bash
[clariden-lnXXX]$ srun -A a137 --pty bash
[nidYYYYYY]$ podman build -t ngc-pytorch:25.04 . 
# ... lots of output here ...
[nidYYYYYY]$ enroot import -x mount \
  -o $SCRATCH/ce-images/gidd.sqsh \
  podman://ngc-pytorch:25.04 
# ... more output here ...
```

The above should take ~10 mins, after which we can check:

```bash
[clariden-lnXXX]$ ls $SCRATCH/ce-images
gidd.sqsh
```

### Step 7

Put the following in a text file `gidd.toml` located at `~/.edf`. Remember to fill in your username. (You have already done this step in `Section I`)

```bash
image = "/iopsstor/scratch/cscs/[your_username]/ce-images/gidd.sqsh"

mounts = [
  "/users/[your_username]:/mnt/home",
  "/iopsstor/scratch/cscs/[your_username]:/mnt/scratch",
  "/iopsstor/scratch/cscs/[your_username]:/iopsstor/scratch/cscs/[your_username]"  # for symlink resolution
]

workdir = "/mnt/home/NLU"  # <-- leading slash

[env]
HOME = "/mnt/home"
SCRATCH = "/mnt/scratch"
NCCL_DEBUG = "INFO"
CUDA_CACHE_DISABLE = "1"
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"
MPICH_GPU_SUPPORT_ENABLED = "0"

[annotations]
"com.hooks.aws_ofi_nccl.enabled" = "true"
"com.hooks.aws_ofi_nccl.variant" = "cuda12"
```

```bash
image = "/iopsstor/scratch/cscs/mojtaba_nafez/ce-images/gidd.sqsh"

mounts = [
  "/users/mojtaba_nafez:/mnt/home",
  "/iopsstor/scratch/cscs/mojtaba_nafez:/mnt/scratch",
  "/iopsstor/scratch/cscs/mojtaba_nafez:/iopsstor/scratch/cscs/mojtaba_nafez"  # for symlink resolution
]

workdir = "/mnt/home/NLU"  # <-- leading slash

[env]
HOME = "/mnt/home"
SCRATCH = "/mnt/scratch"
NCCL_DEBUG = "INFO"
CUDA_CACHE_DISABLE = "1"
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"
MPICH_GPU_SUPPORT_ENABLED = "0"

[annotations]
"com.hooks.aws_ofi_nccl.enabled" = "true"
"com.hooks.aws_ofi_nccl.variant" = "cuda12"
```

## III. Configure other packages

First start an interactive job session within the uni-d2 docker container:

```bash
srun --environment=gidd  -A go082   --pty bash
```

If you cannot start the environment, the docker image construction did not go through. If you get the job up and running, do the following, which allows us to install addition packages on top of the docker environment.

### Step 1

Construct a virtual environment.

```bash
python -m venv --system-site-packages venv-docker
source venv-docker/bin/activate
```

or install conda env (not: aarch64 architecture of amd):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
bash miniconda.sh -b -p ~/miniconda3
source ~/miniconda3/bin/activate
```

setup the conda:

```bash
conda create -n gidd python=3.10 -y
conda activate gidd
pip install -r requirements.txt && pip install -e .
```
Check the installation of torch.
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

if that's get wrong!

```
pip uninstall -y torch torchvision torchaudio
export CUDA_HOME=/usr/local/cuda-13.1
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
pip install torch torchvision torchaudio
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Step 2

Install necessary packages: this is from the original README file of the repo, I made some changes to accommodate for our docker setup.

```bash
pip install -e . --no-deps
pip install flash-attn --no-build-isolation
pip install liger-kernel
```

The original repo is a bit unorganized. Chances are, `flash-attn` and `liger-kernel` have already been baked into the docker image at this point. But just in case, we run the above.

## IV. Run a training code

Now that we have finished with environment setup, we will run a piece of training code. The following training code is the original one from`README`.

(Make sure you are still in the active job with the `venv-docker` or  conda evn `gidd` activated.)

Run:

```bash
torchrun --nnodes 1 --nproc_per_node 4 --master_port 29501 gidd/train.py --config-name gidd model.p_uniform=0.2 model.nvib_layers=[4,6,8] logging.run_name="'small-nvib-gidd+-owt-pu=0.2'" model=small training.num_train_steps=125000
```

Things should run :)

## V. A prototypical slurm script
