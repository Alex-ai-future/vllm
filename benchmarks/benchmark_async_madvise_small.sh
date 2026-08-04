#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Small paired baseline/candidate startup benchmark for a PVC-backed pod.

ROOT_DIR="${ROOT_DIR:-/home/jovyan/vllm-project}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${ROOT_DIR}/vllm}"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/vllm-baseline}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/vllm-benchmark-results/async-madvise-small}"
CONDA_ENV="${CONDA_ENV:-/home/jovyan/conda-envs/cuda130-build}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TP="${TP:-1}"
GPU="${GPU:-0}"
KV_GIB="${KV_GIB:-1}"
REPEATS="${REPEATS:-3}"
COLD_ITERS="${COLD_ITERS:-1}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
WARM_ITERS="${WARM_ITERS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
RANK_LOCAL_REGISTRATION="${RANK_LOCAL_REGISTRATION:-0}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This benchmark requires Linux." >&2
    exit 2
fi
for root in "${CANDIDATE_DIR}" "${BASELINE_DIR}"; do
    if [[ ! -x "${root}/.venv/bin/python" ]]; then
        echo "Missing Python environment: ${root}/.venv/bin/python" >&2
        exit 2
    fi
done

mkdir -p "${RESULT_DIR}/candidate" "${RESULT_DIR}/baseline"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
export PATH="${CANDIDATE_DIR}/.venv/bin:${BASELINE_DIR}/.venv/bin:/home/jovyan/.local/bin:${PATH}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export PYTHONUNBUFFERED=1

cleanup_new_mmaps() {
    local file
    for file in "$@"; do
        [[ -e "${file}" ]] && rm -- "${file}"
    done
}

if compgen -G "/dev/shm/vllm_offload_*.mmap" > /dev/null; then
    echo "Found existing vLLM mmap files in /dev/shm; clean them first." >&2
    exit 2
fi

{
    echo "model=${MODEL}"
    echo "tensor_parallel_size=${TP}"
    echo "gpu=${GPU}"
    echo "kv_gib=${KV_GIB}"
    echo "repeats=${REPEATS}"
    echo "cold_iters=${COLD_ITERS}"
    echo "warmup_iters=${WARMUP_ITERS}"
    echo "warm_iters=${WARM_ITERS}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "enforce_eager=${ENFORCE_EAGER}"
    echo "rank_local_registration=${RANK_LOCAL_REGISTRATION}"
    echo "kernel=$(uname -a)"
    echo "shm=$(df -h /dev/shm | tail -n 1)"
    echo "cuda_home=${CUDA_HOME}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true
    echo "baseline=$(git -C "${BASELINE_DIR}" rev-parse HEAD)"
    echo "candidate=$(git -C "${CANDIDATE_DIR}" rev-parse HEAD)"
} > "${RESULT_DIR}/metadata.txt"

run_case() {
    local label="$1"
    local root="$2"
    local run_id="$3"
    local output_dir="${RESULT_DIR}/${label}"
    local python_bin="${root}/.venv/bin/python"
    local output_json="${output_dir}/run-${run_id}.json"
    local output_log="${output_dir}/run-${run_id}.log"
    local extra_args=(--max-model-len "${MAX_MODEL_LEN}")
    local before_files=(/dev/shm/vllm_offload_*.mmap)
    local after_files=()
    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
        extra_args+=(--enforce-eager)
    fi
    if [[ "${label}" == "candidate" && "${RANK_LOCAL_REGISTRATION}" == "1" ]]; then
        extra_args+=(
            --kv-transfer-config
            '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"rank_local_registration":true}}'
        )
    fi

    echo "Running ${label} run=${run_id}"
    if compgen -G "/dev/shm/vllm_offload_*.mmap" > /dev/null; then
        echo "A previous run left a vLLM mmap file." >&2
        exit 3
    fi
    local status=0
    if (
        cd "${root}"
        "${python_bin}" -m vllm.entrypoints.cli.main bench startup \
            --model "${MODEL}" \
            --tensor-parallel-size "${TP}" \
            "${extra_args[@]}" \
            --kv-offloading-size "${KV_GIB}" \
            --num-iters-cold "${COLD_ITERS}" \
            --num-iters-warmup "${WARMUP_ITERS}" \
            --num-iters-warm "${WARM_ITERS}" \
            --output-json "${output_json}"
    ) > "${output_log}" 2>&1; then
        :
    else
        status=$?
    fi
    after_files=(/dev/shm/vllm_offload_*.mmap)
    for file in "${after_files[@]}"; do
        [[ " ${before_files[*]} " == *" ${file} "* ]] || cleanup_new_mmaps "${file}"
    done
    if (( status != 0 )); then
        return "${status}"
    fi
}

for run_id in $(seq 1 "${REPEATS}"); do
    if (( run_id % 2 == 1 )); then
        run_case baseline "${BASELINE_DIR}" "${run_id}"
        run_case candidate "${CANDIDATE_DIR}" "${run_id}"
    else
        run_case candidate "${CANDIDATE_DIR}" "${run_id}"
        run_case baseline "${BASELINE_DIR}" "${run_id}"
    fi
done

echo "Results written to ${RESULT_DIR}"
