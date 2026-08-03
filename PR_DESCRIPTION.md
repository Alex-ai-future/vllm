# [KV Offload] Run ranked mmap population in the background

## Summary

This change moves the ranked `MADV_POPULATE_WRITE` per-block loop in
`SharedOffloadRegion` to a background thread. The worker continues creating
the mmap-backed tensor views while population runs, then waits before creating
transfer handlers so initialization failures retain the existing error
semantics.

The scheduler-side `rank=None` path remains synchronous. This change does not
alter KV cache layout, block ranges, transfer APIs, or load/store readiness.

## Why this is not a duplicate

- #44913 addresses `MADV_POPULATE_WRITE` fallback on Linux kernels older than
  5.14. It is not a prerequisite for this change and is not implemented here.
- #50358 addresses fail-fast `/dev/shm` capacity validation. It is separate
  lifecycle work and should be rebased carefully if it lands first.
- The current change only overlaps ranked page population with existing worker
  initialization work.

## Testing

Commands to run on a Linux test host:

```bash
.venv/bin/python -m pytest \
  tests/v1/kv_offload/cpu/test_shared_offload_region.py \
  tests/v1/kv_offload/cpu/test_gpu_worker.py -v

pre-commit run ruff-check --files \
  vllm/v1/kv_offload/cpu/shared_offload_region.py \
  vllm/v1/kv_offload/cpu/gpu_worker.py \
  tests/v1/kv_offload/cpu/test_shared_offload_region.py \
  tests/v1/kv_offload/cpu/test_gpu_worker.py

pre-commit run ruff-format --files \
  vllm/v1/kv_offload/cpu/shared_offload_region.py \
  vllm/v1/kv_offload/cpu/gpu_worker.py \
  tests/v1/kv_offload/cpu/test_shared_offload_region.py \
  tests/v1/kv_offload/cpu/test_gpu_worker.py
```

The implementation was prepared on macOS, where `/dev/shm`, CUDA, and the
Linux `MADV_POPULATE_WRITE` behavior are unavailable. The commands above must
be run on Linux before submission.

## Benchmark

Run from two clean revisions using the included script:

```bash
LABEL=baseline ./benchmarks/benchmark_async_madvise.sh
LABEL=candidate ./benchmarks/benchmark_async_madvise.sh
```

The script records cold and warm startup JSON, debug logs, kernel/GPU metadata,
and runs both a 4 GiB control configuration and a stress configuration using
up to 50% of available `/dev/shm` (capped at 64 GiB). Compare engine startup,
population, wait, and host-registration timings in the generated files.

Results:

```text
<!-- Fill in baseline/candidate results and first-request smoke-test results
     after running on a Linux CUDA host. -->
```

## AI assistance

AI assistance was used for the implementation, tests, benchmark script, and
this description. The submitting human must review every changed line and run
the relevant tests and benchmark before opening a PR.
