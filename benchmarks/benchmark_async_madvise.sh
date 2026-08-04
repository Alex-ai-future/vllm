#!/usr/bin/env bash
set -euo pipefail

# Run this script on a Linux CUDA host from two clean revisions:
#   LABEL=baseline ./benchmarks/benchmark_async_madvise.sh
#   LABEL=candidate ./benchmarks/benchmark_async_madvise.sh
# Compare the JSON files and the debug logs under OUTPUT_DIR.

LABEL="${LABEL:-candidate}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TP="${TP:-1}"
CONTROL_GIB="${CONTROL_GIB:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/results/async_madvise/${LABEL}}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This benchmark requires Linux because SharedOffloadRegion uses /dev/shm." >&2
    exit 2
fi

if [[ ! -d /dev/shm ]]; then
    echo "/dev/shm is unavailable." >&2
    exit 2
fi

available_gib=$(df -Pk /dev/shm | awk 'NR == 2 {printf "%d", $4 / 1024 / 1024}')
stress_gib=$((available_gib / 2))
if (( stress_gib < 8 )); then
    echo "Need at least 8 GiB available in /dev/shm; found ${available_gib} GiB." >&2
    exit 2
fi
if (( stress_gib > 64 )); then
    stress_gib=64
fi

mkdir -p "${OUTPUT_DIR}"

{
    echo "label=${LABEL}"
    echo "model=${MODEL}"
    echo "tensor_parallel_size=${TP}"
    echo "control_gib=${CONTROL_GIB}"
    echo "stress_gib=${stress_gib}"
    echo "kernel=$(uname -a)"
    echo "shm=$(df -h /dev/shm | tail -n 1)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    fi
} > "${OUTPUT_DIR}/metadata.txt"

run_case() {
    local name="$1"
    local offload_gib="$2"
    local output_json="${OUTPUT_DIR}/${name}.json"
    local output_log="${OUTPUT_DIR}/${name}.log"

    echo "Running ${LABEL}/${name}: ${offload_gib} GiB"
    VLLM_LOGGING_LEVEL=DEBUG vllm bench startup \
        --model "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --kv-offloading-size "${offload_gib}" \
        --num-iters-cold 1 \
        --num-iters-warmup 1 \
        --num-iters-warm 5 \
        --output-json "${output_json}" \
        > "${output_log}" 2>&1
}

run_case control "${CONTROL_GIB}"
run_case stress "${stress_gib}"

echo "Results written to ${OUTPUT_DIR}"
