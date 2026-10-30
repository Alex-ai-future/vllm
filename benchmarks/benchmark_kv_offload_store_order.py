# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare store-first and sync-load-first submission using the CPU worker.

The benchmark isolates transfer submission and wait; it does not run a model,
scheduler, token sampling, block reuse, or the connector's lifecycle hooks.
"""

import argparse
import json
import os
import random
import statistics
import time

import torch

import vllm

if extension_path := os.environ.get("VLLM_BENCH_EXTENSION_PATH"):
    vllm.__path__.append(extension_path)

from vllm.v1.kv_offload.base import (  # noqa: E402
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec  # noqa: E402
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-blocks", type=int, default=16)
    parser.add_argument("--store-jobs", type=int, default=8)
    parser.add_argument("--store-blocks", type=int, default=2)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--page-bytes", type=int, default=16384)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        min(
            args.sync_blocks,
            args.store_blocks,
            args.layers,
            args.page_bytes,
            args.iterations,
        )
        < 1
        or args.store_jobs < 0
        or args.warmup < 0
    ):
        parser.error(
            "sizes and iterations must be positive; "
            "job count and warmup cannot be negative"
        )

    blocks = args.sync_blocks + args.store_jobs * args.store_blocks
    if blocks * args.layers * args.page_bytes > 256 * 1024**2:
        parser.error("per-device cache allocation exceeds 256 MiB")
    tensors = [
        torch.full((blocks, args.page_bytes), 7, dtype=torch.int8, device="cuda")
        for _ in range(args.layers)
    ]
    caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=t, page_size_bytes=args.page_bytes)
            for t in tensors
        ],
        group_data_refs=[
            [
                CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=args.page_bytes)
                for i in range(args.layers)
            ]
        ],
    )
    backend = CPUOffloadingWorker(caches, 1, blocks)
    for tensor in backend._load_handler.src_tensors:
        tensor[: args.sync_blocks].fill_(3)
    sync_ids = list(range(args.sync_blocks))
    sync_cpu = CPULoadStoreSpec(sync_ids)
    sync_gpu = GPULoadStoreSpec(
        sync_ids, group_sizes=(len(sync_ids),), block_indices=(0,)
    )
    stores = []
    for job in range(args.store_jobs):
        start = args.sync_blocks + job * args.store_blocks
        ids = list(range(start, start + args.store_blocks))
        stores.append(
            (
                job + 1,
                GPULoadStoreSpec(ids, group_sizes=(len(ids),), block_indices=(0,)),
                CPULoadStoreSpec(ids),
            )
        )

    def run(order):
        backend.get_finished()
        torch.accelerator.synchronize()
        start = time.perf_counter_ns()
        sync_submit_ns = store_submit_ns = 0

        def load():
            nonlocal sync_submit_ns
            begin = time.perf_counter_ns()
            assert backend.submit_load(0, sync_cpu, sync_gpu, expedited=True)
            sync_submit_ns = time.perf_counter_ns() - begin

        def store():
            nonlocal store_submit_ns
            begin = time.perf_counter_ns()
            for job_id, src, dst in stores:
                assert backend.submit_store(job_id, src, dst)
            store_submit_ns = time.perf_counter_ns() - begin

        if order == "store_first":
            store()
            load()
        else:
            load()
            store()
        begin_wait = time.perf_counter_ns()
        backend.wait({0})
        forward_launch = time.perf_counter_ns()
        store_events = [x.end_event for x in backend._store_handler._transfers]
        for event in store_events:
            event.synchronize()
        stores_done = time.perf_counter_ns()
        backend.get_finished()
        return {
            "order": order,
            "pre_forward_ms": (forward_launch - start) / 1e6,
            "sync_submit_ms": sync_submit_ns / 1e6,
            "store_submit_ms": store_submit_ns / 1e6,
            "sync_wait_ms": (forward_launch - begin_wait) / 1e6,
            "stores_done_ms": (stores_done - start) / 1e6,
        }

    rows = []
    rng = random.Random(61961)
    try:
        for iteration in range(args.warmup + args.iterations):
            orders = ["store_first", "load_first"]
            rng.shuffle(orders)
            for order in orders:
                row = run(order)
                if iteration >= args.warmup:
                    rows.append(row)
        torch.accelerator.synchronize()
        assert all(torch.all(t[: args.sync_blocks] == 3).item() for t in tensors)
        assert all(
            torch.all(t[args.sync_blocks :] == 7).item()
            for t in backend._store_handler.dst_tensors
        )
        summary = {}
        for order in ("store_first", "load_first"):
            selected = [row for row in rows if row["order"] == order]
            summary[order] = {
                key: statistics.median(row[key] for row in selected)
                for key in selected[0]
                if key != "order"
            }
        with open(args.output, "w") as output:
            json.dump(
                {"config": vars(args), "summary_median_ms": summary, "samples": rows},
                output,
                indent=2,
            )
        print(json.dumps(summary, indent=2))
    finally:
        backend.shutdown()


if __name__ == "__main__":
    main()
