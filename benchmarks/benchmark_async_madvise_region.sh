#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Paired mmap population/registration benchmark with one fresh Python process
# per measurement. Both worktrees use their production defaults; the baseline
# is the main worktree and the candidate is the current worktree.

ROOT_DIR="${ROOT_DIR:-/home/jovyan/vllm-project}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${ROOT_DIR}/vllm}"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/vllm-baseline}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/vllm-benchmark-results/async-madvise-region-paired}"
CONDA_ENV="${CONDA_ENV:-/home/jovyan/conda-envs/cuda130-build}"
GPU="${GPU:-0}"
REPEATS="${REPEATS:-5}"
NUM_BLOCKS="${NUM_BLOCKS:-512}"
ROW_SIZE_MIB="${ROW_SIZE_MIB:-2}"
NUM_WORKERS="${NUM_WORKERS:-1}"
RANK="${RANK:-0}"
CANDIDATE_RANK_LOCAL_REGISTRATION="${CANDIDATE_RANK_LOCAL_REGISTRATION:-0}"
REGION_SCRIPT="${CANDIDATE_DIR}/benchmarks/benchmark_async_madvise_region.py"

for root in "${CANDIDATE_DIR}" "${BASELINE_DIR}"; do
    if [[ ! -x "${root}/.venv/bin/python" ]]; then
        echo "Missing Python environment: ${root}/.venv/bin/python" >&2
        exit 2
    fi
done
if [[ ! -f "${REGION_SCRIPT}" ]]; then
    echo "Missing region benchmark: ${REGION_SCRIPT}" >&2
    exit 2
fi

mkdir -p "${RESULT_DIR}/candidate" "${RESULT_DIR}/baseline"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export PYTHONUNBUFFERED=1

if compgen -G "/dev/shm/vllm_offload_*.mmap" > /dev/null; then
    echo "Found existing vLLM mmap files in /dev/shm; clean them first." >&2
    exit 2
fi

{
    echo "gpu=${GPU}"
    echo "repeats=${REPEATS}"
    echo "num_blocks=${NUM_BLOCKS}"
    echo "row_size_mib=${ROW_SIZE_MIB}"
    echo "num_workers=${NUM_WORKERS}"
    echo "rank=${RANK}"
    echo "candidate_rank_local_registration=${CANDIDATE_RANK_LOCAL_REGISTRATION}"
    echo "cuda_home=${CUDA_HOME}"
    echo "baseline=$(git -C "${BASELINE_DIR}" rev-parse HEAD)"
    echo "candidate=$(git -C "${CANDIDATE_DIR}" rev-parse HEAD)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true
} > "${RESULT_DIR}/metadata.txt"

cleanup_new_mmaps() {
    local file
    for file in "$@"; do
        [[ -e "${file}" ]] && rm -- "${file}"
    done
}

run_case() {
    local label="$1"
    local root="$2"
    local run_id="$3"
    local output_json="${RESULT_DIR}/${label}/run-${run_id}.json"
    local output_log="${RESULT_DIR}/${label}/run-${run_id}.log"
    local before_files=(/dev/shm/vllm_offload_*.mmap)
    local after_files=()
    local status=0
    echo "Running ${label} run=${run_id}"
    if compgen -G "/dev/shm/vllm_offload_*.mmap" > /dev/null; then
        echo "A previous run left a vLLM mmap file." >&2
        exit 3
    fi
    local extra_args=(
        --num-blocks "${NUM_BLOCKS}"
        --row-size-mib "${ROW_SIZE_MIB}"
        --num-workers "${NUM_WORKERS}"
        --rank "${RANK}"
        --repeats 1
        --output-json "${output_json}"
    )
    if [[ "${label}" == "candidate" && "${CANDIDATE_RANK_LOCAL_REGISTRATION}" == "1" ]]; then
        extra_args+=(--rank-local-registration)
    fi
    if (
        cd "${root}"
        PYTHONPATH="${root}" "${root}/.venv/bin/python" "${REGION_SCRIPT}" \
            "${extra_args[@]}"
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
