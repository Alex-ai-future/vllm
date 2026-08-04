"""Measure mmap population overlap with CUDA host registration."""

import argparse
import json
import mmap as mmap_module
import threading
import time
from pathlib import Path

from vllm.v1.kv_offload.cpu.gpu_worker import pin_mmap_region
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

_MADVISE_TIMINGS: list[tuple[float, float]] = []
_MADVISE_TIMINGS_LOCK = threading.Lock()


class _TimedMmap(mmap_module.mmap):
    """Record mmap.madvise intervals without changing vLLM implementation."""

    def madvise(self, *args):
        start = time.perf_counter()
        try:
            return super().madvise(*args)
        finally:
            end = time.perf_counter()
            with _MADVISE_TIMINGS_LOCK:
                _MADVISE_TIMINGS.append((start, end))


# SharedOffloadRegion resolves mmap.mmap at construction time. Replacing the
# constructor here instruments both the baseline and candidate worktrees.
mmap_module.mmap = _TimedMmap


def measure_once(async_population: bool, num_blocks: int, row_size: int) -> dict:
    region = None
    with _MADVISE_TIMINGS_LOCK:
        _MADVISE_TIMINGS.clear()
    start = time.perf_counter()
    try:
        kwargs = {
            "engine_id": f"region-bench-{time.time_ns()}",
            "num_blocks": num_blocks,
            "rank": 0,
            "kv_bytes_per_block": row_size,
            "cpu_page_size": row_size,
        }
        if async_population:
            kwargs["async_population"] = True
        region = SharedOffloadRegion(**kwargs)
        created = time.perf_counter()

        pin_start = time.perf_counter()
        pin_mmap_region(region)
        pinned = time.perf_counter()

        page_size = region.page_size
        view_size = (row_size // page_size // 28) * page_size
        for _ in range(28):
            region.create_next_view(view_size)
        views_created = time.perf_counter()

        wait_start = time.perf_counter()
        wait_for_population = getattr(region, "wait_for_population", None)
        if wait_for_population is not None:
            wait_for_population()
        waited = time.perf_counter()

        with _MADVISE_TIMINGS_LOCK:
            madvise_timings = list(_MADVISE_TIMINGS)
        if madvise_timings:
            madvise_start = min(start for start, _ in madvise_timings)
            madvise_end = max(end for _, end in madvise_timings)
            madvise_seconds = sum(
                end - start for start, end in madvise_timings
            )
            madvise_wall_seconds = madvise_end - madvise_start
            register_overlap_seconds = max(
                0.0,
                min(madvise_end, pinned)
                - max(madvise_start, pin_start),
            )
        else:
            madvise_start = None
            madvise_end = None
            madvise_seconds = 0.0
            madvise_wall_seconds = 0.0
            register_overlap_seconds = 0.0

        return {
            "async_population": async_population,
            "num_blocks": num_blocks,
            "row_size_bytes": row_size,
            "madvise_call_count": len(madvise_timings),
            "madvise_seconds": madvise_seconds,
            "madvise_wall_seconds": madvise_wall_seconds,
            "region_create_seconds": created - start,
            "cuda_host_register_seconds": pinned - pin_start,
            "madvise_register_overlap_seconds": register_overlap_seconds,
            "view_setup_seconds": views_created - pinned,
            "population_wait_seconds": waited - wait_start,
            "total_seconds": waited - start,
        }
    finally:
        if region is not None:
            region.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async-population", action="store_true")
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--row-size-mib", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    results = [
        measure_once(
            async_population=args.async_population,
            num_blocks=args.num_blocks,
            row_size=args.row_size_mib * 1024 * 1024,
        )
        for _ in range(args.repeats)
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"results": results}, indent=2))
    for result in results:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
