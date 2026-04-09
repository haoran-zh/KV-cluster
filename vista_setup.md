# RLKV Setup Guide For Vista (GH200)

## Scope

This guide documents the setup that was used to bring RLKV training up on the Vista cluster at TACC for GH200 nodes.

It covers:

- the `rlkv` conda environment
- SGLang and AReaL installation on Vista
- Vista-specific runtime and Slurm fixes
- Hugging Face and W&B auth handling
- PCA initialization for Learned-Loki
- smoke testing on `gh-dev`
- formal training submission on `gh`

This guide assumes the project lives at:

```bash
/work/11240/hz8556/vista/RLKV
```

and that you are using the current checkout of this repo, which already contains the Vista fixes described below.

## Vista-Specific Constraints

### 1. Use `gh` for real training

`gh-dev` is useful for short validation because it usually starts faster, but it is not suitable for the full decoupled RLKV training flow on Vista:

- `gh-dev` allows only one active job for this QoS
- the Slurm launcher submits two jobs:
  - `llm_server`
  - `trainer`
- once the rollout server starts, the trainer submission on `gh-dev` hits `QOSMaxJobsPerUserLimit`

Use:

- `gh-dev` for rollout/bootstrap smoke tests
- `gh` for the actual training run

### 2. Two nodes are required

For the current config:

```yaml
allocation_mode: sglang:d1+fsdp:d1
cluster.n_nodes=2
cluster.n_gpus_per_node=1
```

Vista GH200 nodes expose one GPU per node for this workflow. Running rollout and trainer on the same GPU led to duplicate-GPU/NCCL issues during online weight sync. The working deployment model is:

- node 1: SGLang rollout server
- node 2: FSDP trainer

### 3. Vista Slurm behavior is different from the repo defaults

On Vista:

- `GresTypes = (null)`, so `--gres=gpu:...` must be omitted
- `--mem` and `--mem-per-cpu` requests in this launcher path are rejected on GH partitions
- `gh-dev` enforces a one-job QoS limit

The current repo checkout has Slurm fixes for this already.

## Code Fixes Required On This Checkout

These files were patched to make RLKV work on Vista:

- `AReaL/areal/learned_loki/forward.py`
  - Learned-Loki scalar parameters are stored as 1-element tensors so FSDP2 can wrap them.
- `sglang/python/sglang/srt/layers/quantization/awq.py`
- `sglang/python/sglang/srt/layers/quantization/gptq.py`
  - `fused_marlin_moe` is treated as optional so non-quantized Qwen runs do not fail when the installed `sgl-kernel` lacks that symbol.
- `AReaL/areal/utils/slurm.py`
  - omits `--gres` on Vista
  - supports disabling Slurm memory flags
  - supports Vista-compatible helper `srun` commands
- `AReaL/areal/launcher/slurm.py`
  - Slurm-safe job naming
  - `--export=ALL,...` so `PATH` survives inside `srun`
  - Slurm-aware rollout wait loop
  - optional omission of memory flags
- `AReaL/areal/utils/launcher.py`
  - redirects Hugging Face caches into `AREAL_LOCAL_CACHE_DIR`

## 1. Create The Conda Environment

Run this on a Vista login node first:

```bash
conda create -n rlkv python=3.10 -y
conda activate rlkv

conda install -y git
conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit
conda install -y nvidia::cuda-cudart-dev

pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip freeze | grep -iE 'torch|nvidia' > /tmp/constraints.txt
```

## 2. Install SGLang

The pinned `sgl-kernel==0.3.9.post2` is not available directly from pip on this branch. Build the local wheel first.

Do the build on a GH200 compute node if possible.

```bash
cd /work/11240/hz8556/vista/RLKV/sglang

conda install -y -c conda-forge "cmake>=3.26" ninja
pip install -U pip setuptools wheel ninja scikit-build-core uv

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export CUDAToolkit_ROOT="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_BUILD_PARALLEL_LEVEL=2

cd sgl-kernel
python -m uv build --wheel \
  -Cbuild-dir=build \
  -Ccmake.define.CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
  -Ccmake.define.CUDAToolkit_ROOT="$CUDAToolkit_ROOT" \
  -Ccmake.define.CMAKE_CUDA_COMPILER="$CUDACXX" \
  -Ccmake.define.SGL_KERNEL_COMPILE_THREADS=2 \
  --no-build-isolation -v

pip install --force-reinstall dist/sgl_kernel-0.3.9.post2*.whl

cd ../python
pip install -e ".[srt]" -c /tmp/constraints.txt

python -m flashinfer --download-cubin
```

## 3. Install AReaL

```bash
cd /work/11240/hz8556/vista/RLKV/AReaL

pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
pip install "deepspeed>=0.17.2" pynvml -c /tmp/constraints.txt
pip install megatron-core==0.13.1 nvidia-ml-py -c /tmp/constraints.txt
MAX_JOBS=2 pip install "flash-attn<=2.8.1" --no-build-isolation --no-cache-dir

pip install -e evaluation/latex2sympy
pip install -e .[dev] -c /tmp/constraints.txt
```

## 4. Install Block-Sparse-Attention

```bash
cd /work/11240/hz8556/vista/RLKV
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
MAX_JOBS=1 python setup.py install
```

## 5. Configure Auth And Cache Paths

Do not hardcode tokens in the repo. Put them in an activation hook instead.

### Auth file

Create:

```bash
mkdir -p ~/.config/rlkv
cat > ~/.config/rlkv/auth.env <<'EOF'
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_hf_token>
export HUGGINGFACE_HUB_TOKEN=<your_hf_token>
EOF
chmod 600 ~/.config/rlkv/auth.env
```

### Conda activation hook

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/rlkv_auth.sh" <<'EOF'
source ~/.config/rlkv/auth.env
export AREAL_LOCAL_CACHE_DIR=/work/11240/hz8556/vista/.cache/areal
mkdir -p "$AREAL_LOCAL_CACHE_DIR"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/rlkv_auth.sh" <<'EOF'
unset WANDB_API_KEY
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
unset AREAL_LOCAL_CACHE_DIR
EOF
```

Re-activate:

```bash
conda deactivate
conda activate rlkv
```

### Why this matters

The Hugging Face cache must not default to `~/.cache/huggingface` on Vista. During the smoke run, model download failed with:

```text
OSError: [Errno 122] Disk quota exceeded
```

The fix is to keep:

```bash
export AREAL_LOCAL_CACHE_DIR=/work/11240/hz8556/vista/.cache/areal
```

The current repo patch converts that into:

- `HF_HOME`
- `HF_HUB_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `TRANSFORMERS_CACHE`
- `HF_DATASETS_CACHE`

all rooted under `/work`.

## 6. PCA Initialization For Learned-Loki

Before the main RL run, create a PCA-initialized Learned-Loki checkpoint.

Example:

```bash
cd /work/11240/hz8556/vista/RLKV/AReaL
conda activate rlkv

export AREAL_LOCAL_CACHE_DIR=/work/11240/hz8556/vista/.cache/areal

python3 scripts/calibrate_learned_loki.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --dataset Kurt232/Sampled-Laser-Dataset-V4 \
  --dataset-type rl \
  --dataset-split train \
  --output-dir expr/pca-qwen3-smoke \
  --rank 32 \
  --budget-ratio 0.5 \
  --sink-window-size 16 \
  --recent-window-size 64 \
  --gate-temperature 1.0 \
  --threshold-init 0.0 \
  --num-samples 256 \
  --max-length 1024 \
  --dtype bfloat16 \
  --device cuda \
  --attn-implementation sdpa
```

Expected output:

```text
Saved PCA-initialized Learned-Loki checkpoint to expr/pca-qwen3-smoke
```

## 7. Slurm Wrappers

The launcher internally runs `sbatch`, so the easiest way to bind account, partition, wall time, and mail settings is to prepend a small `sbatch` wrapper into `PATH`.

### `gh-dev` wrapper

```bash
mkdir -p /tmp/tacc-bin-ghdev
cat > /tmp/tacc-bin-ghdev/sbatch <<'EOF'
#!/bin/bash
exec /usr/bin/sbatch -A ASC24078 -p gh-dev -t 02:00:00 \
  --mail-user=haoran@utexas.edu --mail-type=BEGIN,FAIL,END "$@"
EOF
chmod +x /tmp/tacc-bin-ghdev/sbatch
```

### `gh` wrapper

```bash
mkdir -p /tmp/tacc-bin-gh
cat > /tmp/tacc-bin-gh/sbatch <<'EOF'
#!/bin/bash
exec /usr/bin/sbatch -A ASC24078 -p gh -t 06:00:00 \
  --mail-user=haoran@utexas.edu --mail-type=BEGIN,FAIL,END "$@"
EOF
chmod +x /tmp/tacc-bin-gh/sbatch
```

## 8. Smoke Test On `gh-dev`

This is only for validating:

- rollout bootstrap
- model download
- SGLang launch
- name registration
- trainer submission

It will not complete end-to-end on Vista because `gh-dev` only allows one active job and the trainer becomes blocked by `QOSMaxJobsPerUserLimit`.

Example launcher script:

```bash
cat > /tmp/run_qwen3_2node_dev.sh <<'EOF'
#!/bin/bash
set -eo pipefail
source /work/11240/hz8556/vista/miniconda3/etc/profile.d/conda.sh
conda activate rlkv
export PATH=/tmp/tacc-bin-ghdev:$PATH
export AREAL_LOCAL_CACHE_DIR=/work/11240/hz8556/vista/.cache/areal
export AREAL_SLURM_DISABLE_MEM=1
export AREAL_SLURM_ROLLOUT_WAIT_TIMEOUT=0
cd /work/11240/hz8556/vista/RLKV/AReaL
exec python3 -m areal.launcher.slurm examples/math/gsm8k_grpo.py \
  --config examples/learned_loki/grpo.yaml \
  experiment_name=learned-loki-grpo \
  trial_name=qwen3_2node_dev6 \
  allocation_mode=sglang:d1+fsdp:d1 \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=1 \
  +launcher.slurm.container_type=none \
  async_training=false \
  ++actor.path=Qwen/Qwen3-4B-Thinking-2507 \
  ++ref.path=Qwen/Qwen3-4B-Thinking-2507 \
  ++sglang.model_path=Qwen/Qwen3-4B-Thinking-2507 \
  ++actor.attn_impl=sdpa \
  ++ref.attn_impl=sdpa \
  ++actor.learned_loki_init_path=/work/11240/hz8556/vista/RLKV/AReaL/expr/pca-qwen3-smoke \
  ++actor.mb_spec.max_tokens_per_mb=2048 \
  ++actor.max_new_tokens=128 \
  ++gconfig.max_new_tokens=128 \
  ++gconfig.n_samples=2 \
  ++actor.group_size=2 \
  ++train_dataset.batch_size=4 \
  ++valid_dataset.batch_size=4 \
  ++rollout.consumer_batch_size=4 \
  ++rollout.max_concurrent_rollouts=4 \
  ++sglang.context_length=1024 \
  ++sglang.max_running_requests=8 \
  ++sglang.mem_fraction_static=0.5 \
  ++stats_logger.wandb.mode=disabled \
  ++stats_logger.swanlab.mode=disabled
EOF

chmod +x /tmp/run_qwen3_2node_dev.sh
setsid /tmp/run_qwen3_2node_dev.sh > \
  /work/11240/hz8556/vista/RLKV/AReaL/expr/experiments/logs/hz8556/learned-loki-grpo/qwen3_2node_dev6/launcher.nohup.log \
  2>&1 < /dev/null &
```

## 9. Formal Training On `gh`

Use the same launcher pattern but point `PATH` at `/tmp/tacc-bin-gh`.

Example:

```bash
cat > /tmp/run_qwen3_2node_gh.sh <<'EOF'
#!/bin/bash
set -eo pipefail
source /work/11240/hz8556/vista/miniconda3/etc/profile.d/conda.sh
conda activate rlkv
export PATH=/tmp/tacc-bin-gh:$PATH
export AREAL_LOCAL_CACHE_DIR=/work/11240/hz8556/vista/.cache/areal
export AREAL_SLURM_DISABLE_MEM=1
export AREAL_SLURM_ROLLOUT_WAIT_TIMEOUT=0
cd /work/11240/hz8556/vista/RLKV/AReaL
exec python3 -m areal.launcher.slurm examples/math/gsm8k_grpo.py \
  --config examples/learned_loki/grpo.yaml \
  experiment_name=learned-loki-grpo \
  trial_name=qwen3_2node_gh6h3 \
  allocation_mode=sglang:d1+fsdp:d1 \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=1 \
  +launcher.slurm.container_type=none \
  async_training=false \
  ++actor.path=Qwen/Qwen3-4B-Thinking-2507 \
  ++ref.path=Qwen/Qwen3-4B-Thinking-2507 \
  ++sglang.model_path=Qwen/Qwen3-4B-Thinking-2507 \
  ++actor.attn_impl=sdpa \
  ++ref.attn_impl=sdpa \
  ++actor.learned_loki_init_path=/work/11240/hz8556/vista/RLKV/AReaL/expr/pca-qwen3-smoke \
  ++actor.mb_spec.max_tokens_per_mb=2048 \
  ++actor.max_new_tokens=128 \
  ++gconfig.max_new_tokens=128 \
  ++gconfig.n_samples=2 \
  ++actor.group_size=2 \
  ++train_dataset.batch_size=4 \
  ++valid_dataset.batch_size=4 \
  ++rollout.consumer_batch_size=4 \
  ++rollout.max_concurrent_rollouts=4 \
  ++sglang.context_length=1024 \
  ++sglang.max_running_requests=8 \
  ++sglang.mem_fraction_static=0.5 \
  ++stats_logger.wandb.mode=disabled \
  ++stats_logger.swanlab.mode=disabled
EOF

chmod +x /tmp/run_qwen3_2node_gh.sh
setsid /tmp/run_qwen3_2node_gh.sh > \
  /work/11240/hz8556/vista/RLKV/AReaL/expr/experiments/logs/hz8556/learned-loki-grpo/qwen3_2node_gh6h3/launcher.nohup.log \
  2>&1 < /dev/null &
```

## 10. Monitoring

### Queue state

```bash
squeue -u hz8556
```

### Slurm accounting

```bash
sacct -j <jobid> -n -P \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,Start,End,AllocNodes,NodeList,Reason
```

### Rollout server logs

```bash
tail -f /work/11240/hz8556/vista/RLKV/AReaL/expr/experiments/logs/hz8556/learned-loki-grpo/<trial>/llm_server.log
```

### Launcher logs

```bash
tail -f /work/11240/hz8556/vista/RLKV/AReaL/expr/experiments/logs/hz8556/learned-loki-grpo/<trial>/launcher.nohup.log
```

### Trainer logs

```bash
tail -f /work/11240/hz8556/vista/RLKV/AReaL/expr/experiments/logs/hz8556/learned-loki-grpo/<trial>/trainer.log
```

### Name resolve registration

```bash
find /work/11240/hz8556/vista/RLKV/AReaL/expr/name_resolve/hz8556/learned-loki-grpo/<trial> -maxdepth 3 -type f
```

## 11. Known Vista Failure Modes And Fixes

### `sgl-kernel==0.3.9.post2` not found from pip

Cause:

- the branch expects `sgl-kernel==0.3.9.post2`
- PyPI exposes newer versions

Fix:

- build the local wheel from `sglang/sgl-kernel`
- install the local wheel before `pip install -e "python[srt]"`

### `Memory specification can not be satisfied`

Cause:

- Vista GH partitions reject the launcher's default `--mem` / `--mem-per-cpu` requests

Fix:

```bash
export AREAL_SLURM_DISABLE_MEM=1
```

### `hostname: No such file or directory`, `bash: No such file or directory`, `python3: No such file or directory`

Cause:

- `--export=<vars>` replaced the full environment in `srun`

Fix:

- use `--export=ALL,<vars>`

### `Disk quota exceeded` while downloading from Hugging Face

Cause:

- default cache in `/home1/.../.cache/huggingface`

Fix:

```bash
export AREAL_LOCAL_CACHE_DIR=/work/11240/hz8556/vista/.cache/areal
```

### `Timeout waiting for rollout servers to be ready`

Cause:

- the original Slurm launcher used a blind 360-second wait and self-cancelled queued rollout jobs

Fix:

```bash
export AREAL_SLURM_ROLLOUT_WAIT_TIMEOUT=0
```

and use the patched `AReaL/areal/launcher/slurm.py`.

### `QOSMaxJobsPerUserLimit` on `gh-dev`

Cause:

- `gh-dev` only allows one active job for this QoS
- RLKV on Slurm needs both `llm_server` and `trainer`

Fix:

- do not use `gh-dev` for the full training run
- use `gh-dev` only for rollout/bootstrap smoke tests
- use `gh` for formal training

### W&B asks for login in "offline" mode

Cause:

- in this codebase, `offline` still reaches `wandb.login()`

Fix:

always launch with:

```bash
++stats_logger.wandb.mode=disabled
++stats_logger.swanlab.mode=disabled
```

## 12. Recommended Operating Procedure

1. Prepare the environment and install all packages.
2. Set auth tokens only through the conda activation hook.
3. Set `AREAL_LOCAL_CACHE_DIR` to `/work`.
4. Build the PCA initialization checkpoint.
5. Run a `gh-dev` smoke test only until:
   - `llm_server` starts
   - rollout server registers
   - trainer job is submitted
6. Cancel the `gh-dev` smoke jobs.
7. Keep the formal `gh` launcher alive and wait for allocation.
8. Once `gh` starts, monitor:
   - `llm_server.log`
   - `launcher.nohup.log`
   - `trainer.log`

## 13. Current Canonical Launch Environment

These environment variables should always be present for Vista training:

```bash
export AREAL_LOCAL_CACHE_DIR=/work/11240/hz8556/vista/.cache/areal
export AREAL_SLURM_DISABLE_MEM=1
export AREAL_SLURM_ROLLOUT_WAIT_TIMEOUT=0
export HF_TOKEN=<your_hf_token>
export HUGGINGFACE_HUB_TOKEN=<your_hf_token>
export WANDB_API_KEY=<your_wandb_key>
```

## 14. Notes

- The `TRANSFORMERS_CACHE` deprecation warning is harmless for now.
- The `pynvml` warning is also not the current blocker.
- The main unresolved part of the formal run is whatever happens after `gh` actually allocates the rollout node and then the trainer node. Everything before that has been validated on Vista.
