# Semantic KV Compression

This document describes the new `semantic_kv` path added on top of the RLKV codebase.

## What Changed

The original RLKV method learns per-head gates and keeps some heads in full precision while compressing the rest. The new scaffold disables that idea for the new path and moves toward a different objective:

1. All KV heads are compressed.
2. Each layer owns a trainable low-rank projection `W^l in R^{r x d}`.
3. Tokens are projected into a semantic space and a small representative subset is retained in the KV cache.
4. Cache filling is token-selective, not head-selective.

The implemented entry points are:

- Evaluation / inference: `base/semantic_kv_cache.py`
- AReaL training scaffold: `AReaL/areal/semantic_kv/`
- GRPO config/script scaffold: `AReaL/examples/semantic_kv/grpo.yaml` and `AReaL/scripts/run_semantic_kv.sh`

## Current Status

Implemented now:

- All-head selective KV compression during HF evaluation.
- Per-layer low-rank projection weights saved to and loaded from `semantic_kv.pt`.
- AReaL-side compressed attention training forward with online cumulative attention importance.
- Straight-through greedy token selection so the GRPO actor loss can update `W` through the compressed forward path.
- Cluster regularization over the retained semantic centers at each compression event.
- SGLang rollout support for the same hard semantic compression policy on the `torch_native` attention backend.
- Server-side allocator reclamation for semantic-KV rollout via per-request, per-layer slot refcounting.
- Best-effort AReaL `past_key_value` support by carrying a compacted runtime semantic state across cached decode calls.

Remaining limitations:

- Semantic-KV rollout currently requires `torch_native`, `page_size=1`, `disable_radix_cache=true`, and `disable_overlap_schedule=true`.
- The rollout backend still approximates importance if it has to reconstruct semantic state after state loss instead of continuing from the live semantic state.
- AReaL cached decode support is best-effort and assumes stable batch ordering across decode steps.
- AReaL semantic-KV training still does not support Ulysses sequence parallelism.
- The explicit online compressed-attention loop is much slower than the optimized dense attention kernels.

For the exact mathematical formulation of the straight-through implementation, see `SEMANTIC_KV_STRAIGHT_THROUGH.md`.

## Loss Choice

We use:

`L_total = L_GRPO + lambda * L_cluster`

For `L_cluster`, the implementation chooses a soft LDA-style ratio:

- Compute projected token features with the per-layer projector.
- Select representative tokens with greedy farthest-point selection.
- Treat the selected projected tokens as cluster centers.
- Use soft assignments from every token to those centers.
- Minimize within-cluster spread while maximizing between-center separation.

Concretely:

`L_cluster = E[within_cluster / (between_cluster + eps)]`

Why this option:

- It matches the final deployment objective directly: we only keep representative tokens.
- It is fully differentiable once centers are fixed by the current selector.
- It avoids collapse more naturally than a pure entropy term.
- It is simpler to wire into the current trainer than contrastive pair mining over rollout traces.

## Training

The current training command is a scaffold for the new path:

```bash
cd AReaL
bash scripts/run_semantic_kv.sh deepseek-ai/DeepSeek-R1-Distill-Llama-8B semantic_kv_llama_r32
```

Useful environment overrides:

```bash
export SEMANTIC_KV_RANK=32
export SEMANTIC_KV_BUDGET_RATIO=0.5
export SEMANTIC_KV_SINK_SIZE=16
export SEMANTIC_KV_RECENT_SIZE=64
export SEMANTIC_KV_CLUSTER_LOSS_SCALE=0.1
export SEMANTIC_KV_CLUSTER_TEMPERATURE=0.1
export SEMANTIC_KV_LR=1e-2
export SEMANTIC_KV_EPOCHS=2
```

Checkpoint output:

- The projector weights are saved as `semantic_kv.pt`.
- The file contains:
  - `semantic_kv_state_dict` for reloading the AReaL model
  - `projection_weight_dict` for HF evaluation
  - the semantic-KV config metadata

## Evaluation

Evaluate a saved `semantic_kv.pt` checkpoint with the new all-head token selector:

```bash
python -u eval/bench/pred.py \
  --model Llama-3.1-8B-R1 \
  --task gsm8k \
  --method semantic_kv \
  --attn_load_dir /path/to/checkpoint_dir \
  --sparsity 0.4 \
  --sink_size 16 \
  --recent_size 64
```

Or run the benchmark script:

```bash
export MODEL=Llama-3.1-8B-R1
export SEMANTIC_KV_DIR=/path/to/checkpoint_dir
bash scripts/run_bench_semantic_kv.sh
```

## What To Verify

For each checkpoint, verify:

1. Accuracy on `gsm8k`, `math_500`, `aime24`, and `mbpp`.
2. Average output length and early-stop behavior in `eval/src/eval.py`.
3. Effective cache ratio:
   `budget_ratio = 1 - sparsity`
4. Whether the retained-token pattern stays stable across long reasoning traces.

## Code Reading Guide

Start here:

- `base/semantic_kv_cache.py`
- `AReaL/areal/semantic_kv/forward.py`
- `AReaL/areal/semantic_kv/loss.py`
- `AReaL/areal/engine/fsdp_engine.py`
- `eval/bench/pred.py`

The most important implementation files are:

- `AReaL/areal/semantic_kv/forward.py`
- `AReaL/areal/semantic_kv/loss.py`
- `sglang/python/sglang/srt/layers/attention/torch_native_backend.py`
- `sglang/python/sglang/srt/model_executor/model_runner.py`
