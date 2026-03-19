#!/bin/bash

set -e

cleanup() {
    echo "Terminating background processes..."
    kill $(jobs -p) 2>/dev/null
    wait
    exit 1
}
trap cleanup SIGINT SIGTERM

root="eval/bench"
model="${MODEL:-Llama-3.1-8B-R1}"
semantic_kv_dir="${SEMANTIC_KV_DIR:-expr/experiments/semantic-kv-grpo/semantic_kv_r32/checkpoints/epoch1epochstep10globalstep10}"
log_dir="$root/logs/$model"
mkdir -p "$log_dir"

tasks=(
    "math_500"
    "gsm8k"
    "aime24"
    "mbpp"
)
sparsities=(0.2 0.4 0.6 0.8)
gpus=(4 5 6 7)
method="semantic_kv"

for i in "${!sparsities[@]}"; do
    sparsity="${sparsities[i]}"
    gpu="${gpus[i]}"
    (
        for task in "${tasks[@]}"; do
            CUDA_VISIBLE_DEVICES=$gpu bash scripts/bench.sh "$model" "$task" "$semantic_kv_dir" "$sparsity" "$method" > "${log_dir}/${model}_${task}_${method}_${sparsity}.log"
        done
    ) &
done

wait
echo "Benchmarks completed."

python -u eval/src/eval.py --model "$model" --results_path "$root/pred"
