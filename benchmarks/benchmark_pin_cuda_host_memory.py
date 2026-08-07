# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the two CUDA host-registration paths for SharedOffloadRegion.

This benchmark times only the pin operation. Region creation, mmap population,
and cleanup are outside the timed interval. The ``full-mmap`` case is the
previous implementation; ``rank-local-alias`` is the production path exposed
by ``SharedOffloadRegion.pin_cuda_host_memory``.

Example for a 16 GiB shared region with eight rank slots::

    .venv/bin/python benchmarks/benchmark_pin_cuda_host_memory.py \
        --num-blocks 8192 --row-stride-mib 2 --world-size 8 \
        --warmup 1 --runs 3

The benchmark requires a CUDA runtime and a CUDA-capable GPU. ``nvcc`` is not
needed because registration is performed through PyTorch's CUDA runtime API.
It intentionally runs one worker: the optimization changes the registered
virtual-address range per rank, not memory-bandwidth contention. The defaults
model one production rank in an 8-way interleaved, 16 GiB shared region.
"""

from __future__ import annotations

import argparse
import mmap
import statistics
import time
import uuid
from dataclasses import dataclass

import torch

from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

PAGE_SIZE = mmap.PAGESIZE


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs shared by both registration measurements."""

    num_blocks: int
    row_stride_bytes: int
    slot_size_bytes: int
    rank: int


def _cuda_result_code(result: object) -> int:
    value = getattr(result, "value", result)
    return int(value)


def _check_cuda_result(result: object, operation: str) -> None:
    code = _cuda_result_code(result)
    if code != 0:
        raise RuntimeError(f"{operation} failed with CUDA error code {code}")


def _synchronize() -> None:
    torch.accelerator.synchronize()


def _new_region(config: BenchmarkConfig) -> SharedOffloadRegion:
    # Constructing the region intentionally happens before the timer. This
    # benchmark isolates cudaHostRegister, which is the operation being
    # changed in production.
    return SharedOffloadRegion(
        engine_id=f"pin-bench-{uuid.uuid4().hex}",
        num_blocks=config.num_blocks,
        rank=config.rank,
        kv_bytes_per_block=config.row_stride_bytes,
        cpu_page_size=config.slot_size_bytes,
    )


def _measure_full_mmap(config: BenchmarkConfig) -> float:
    """Measure the previous full-mmap cudaHostRegister implementation."""
    region = _new_region(config)
    registered = False
    try:
        _synchronize()
        start = time.perf_counter()
        cudart = torch.cuda.cudart()
        pointer = region._base.data_ptr()
        _check_cuda_result(
            cudart.cudaHostRegister(pointer, region.total_size_bytes, 0),
            "cudaHostRegister(full mmap)",
        )
        registered = True
        _synchronize()
        return (time.perf_counter() - start) * 1000
    finally:
        try:
            if registered:
                _check_cuda_result(
                    torch.cuda.cudart().cudaHostUnregister(pointer),
                    "cudaHostUnregister(full mmap)",
                )
        finally:
            region.cleanup()


def _measure_rank_local_alias(config: BenchmarkConfig) -> float:
    """Measure SharedOffloadRegion's rank-local alias registration path."""
    region = _new_region(config)
    try:
        _synchronize()
        start = time.perf_counter()
        region.pin_cuda_host_memory()
        _synchronize()
        return (time.perf_counter() - start) * 1000
    finally:
        region.cleanup()


def _measure(mode: str, config: BenchmarkConfig, warmup: int, runs: int) -> list[float]:
    measure_once = (
        _measure_full_mmap if mode == "full-mmap" else _measure_rank_local_alias
    )
    for _ in range(warmup):
        measure_once(config)
    return [measure_once(config) for _ in range(runs)]


def _format_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _print_result(
    name: str, samples: list[float], registered_bytes: int
) -> tuple[float, float]:
    mean = statistics.mean(samples)
    median = statistics.median(samples)
    print(
        f"{name}: mean={mean:.2f} ms median={median:.2f} ms "
        f"min={min(samples):.2f} ms max={max(samples):.2f} ms "
        f"registered={_format_bytes(registered_bytes)}"
    )
    return mean, median


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full mmap and rank-local alias host registration."
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=8192,
        help="Number of blocks; defaults to 8192 (16 GiB at 2 MiB rows).",
    )
    parser.add_argument(
        "--row-stride-mib",
        type=int,
        default=2,
        help="Bytes per shared mmap row in MiB; defaults to 2.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=8,
        help="Number of rank slots interleaved in each row; defaults to 8.",
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank slot to benchmark; defaults to 0."
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup iterations per path."
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Measured iterations per path."
    )
    args = parser.parse_args()

    if args.num_blocks <= 0:
        parser.error("--num-blocks must be positive")
    if args.row_stride_mib <= 0:
        parser.error("--row-stride-mib must be positive")
    if args.world_size < 2:
        parser.error("--world-size must be at least 2 for an alias comparison")
    if not 0 <= args.rank < args.world_size:
        parser.error("--rank must be in [0, world-size)")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.runs <= 0:
        parser.error("--runs must be positive")

    row_stride_bytes = args.row_stride_mib * 1024 * 1024
    if row_stride_bytes % args.world_size != 0:
        parser.error("row stride must be divisible by world size")
    slot_size_bytes = row_stride_bytes // args.world_size
    if slot_size_bytes % PAGE_SIZE != 0:
        parser.error("per-rank slot size must be page aligned")

    args.config = BenchmarkConfig(
        num_blocks=args.num_blocks,
        row_stride_bytes=row_stride_bytes,
        slot_size_bytes=slot_size_bytes,
        rank=args.rank,
    )
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this benchmark")

    config: BenchmarkConfig = args.config
    total_bytes = config.num_blocks * config.row_stride_bytes
    alias_bytes = config.num_blocks * config.slot_size_bytes
    print(
        f"config: blocks={config.num_blocks} "
        f"row_stride={_format_bytes(config.row_stride_bytes)} "
        f"total={_format_bytes(total_bytes)} "
        f"rank={config.rank}/{args.world_size} "
        f"alias_size={_format_bytes(alias_bytes)}"
    )
    print("timed operation: cudaHostRegister only")

    full_samples = _measure("full-mmap", config, args.warmup, args.runs)
    alias_samples = _measure("rank-local-alias", config, args.warmup, args.runs)
    full_mean, _ = _print_result("full-mmap", full_samples, total_bytes)
    alias_mean, _ = _print_result("rank-local-alias", alias_samples, alias_bytes)
    print(
        f"alias-vs-full: saved={full_mean - alias_mean:.2f} ms "
        f"speedup={full_mean / alias_mean:.2f}x"
    )


if __name__ == "__main__":
    main()
