# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
import os
import threading
import time

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


class SharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
        async_population: bool = False,
        rank_local_registration: bool = False,
    ) -> None:
        self.initialization_start_time = time.perf_counter()
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        self.rank_local_registration = rank_local_registration
        self._rank_local_registration_active = rank_local_registration and (
            rank is not None and cpu_page_size % self.page_size == 0
        )
        if rank_local_registration and not self._rank_local_registration_active:
            logger.warning(
                "Rank-local host registration requires page-aligned worker "
                "slices; falling back to full mmap registration for rank=%d",
                rank,
            )
        self.registration_mode = (
            "rank-local"
            if self._rank_local_registration_active
            else "full-fallback"
            if rank_local_registration
            else "full"
        )
        self._worker_slot_size = cpu_page_size if rank is not None else None
        self._population_thread: threading.Thread | None = None
        self._population_error: Exception | None = None
        self.population_start_time = 0.0
        self.population_end_time = 0.0
        self.population_time_s = 0.0
        self.population_wait_time_s = 0.0
        self.population_barrier_wait_time_s = 0.0
        self.register_time_s = 0.0
        self.ready_barrier_wait_time_s = 0.0
        self.initialization_time_s = 0.0
        self._registered_ranges: list[tuple[int, int]] = []
        self.registration_call_count = 0
        self.registered_bytes = 0
        self.unregister_call_count = 0
        self.unregister_time_s = 0.0
        if rank is not None:
            # byte offset to this worker's first slot within each block row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        try:
            # Exclusive create — only one worker succeeds
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            os.ftruncate(self.fd, self.total_size_bytes)
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.info("Opened existing mmap file %s", self.mmap_path)

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
        _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
        if rank is not None:
            self.population_start_time = time.perf_counter()
            if async_population:
                self._population_thread = threading.Thread(
                    target=self._run_population_in_background,
                    args=(_MADV_POPULATE_WRITE, rank * cpu_page_size, cpu_page_size),
                    name=f"SharedOffloadRegionPopulation-{rank}",
                )
                self._population_thread.start()
            else:
                # Keep the mapping alive after a population failure so the
                # worker can report the failure through the TP collective.
                # Raising here would leave other ranks waiting forever at the
                # collective.
                self._run_population_in_background(
                    _MADV_POPULATE_WRITE, rank * cpu_page_size, cpu_page_size
                )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            self.population_start_time = _t0
            self.mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, self.total_size_bytes)
            self.population_time_s = time.perf_counter() - _t0
            self.population_end_time = time.perf_counter()
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", self.population_time_s
            )

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self.is_pinned: bool = False

    def get_host_registration_ranges(self) -> list[tuple[int, int]]:
        """Return page-aligned mmap ranges for CUDA host registration.

        Rank-local registration covers this worker's slice in each block row.
        The ranges are expanded to page boundaries because CUDA host
        registration requires page-aligned addresses and sizes, then adjacent
        ranges are merged.  Full-region registration returns one range.
        """
        if self.total_size_bytes == 0:
            return []
        if not self._rank_local_registration_active:
            return [(0, self.total_size_bytes)]

        assert self._worker_slot_size is not None
        ranges: list[tuple[int, int]] = []
        for block in range(self.num_blocks):
            raw_start = block * self._row_stride + self.rank * self._worker_slot_size
            raw_end = raw_start + self._worker_slot_size
            start = (raw_start // self.page_size) * self.page_size
            end = (
                (raw_end + self.page_size - 1) // self.page_size
            ) * self.page_size
            assert end <= (block + 1) * self._row_stride
            if ranges and start <= ranges[-1][0] + ranges[-1][1]:
                previous_start, previous_length = ranges[-1]
                previous_end = previous_start + previous_length
                ranges[-1] = (previous_start, max(previous_end, end) - previous_start)
            else:
                ranges.append((start, end - start))
        return ranges

    def unregister_host_registration(self) -> None:
        """Unregister all host ranges while keeping the mmap alive."""
        ranges = list(self._registered_ranges)
        if not ranges and self.is_pinned and self._base is not None:
            ranges = [(self._base.data_ptr(), self.total_size_bytes)]

        if ranges and current_platform.is_cuda_alike():
            _t0 = time.perf_counter()
            cudart = torch.cuda.cudart()
            for ptr, _ in reversed(ranges):
                try:
                    result = cudart.cudaHostUnregister(ptr)
                    self.unregister_call_count += 1
                except Exception:
                    logger.warning(
                        "cudaHostUnregister raised for rank=%d",
                        self.rank,
                        exc_info=True,
                    )
                    continue
                if result.value != 0:
                    logger.warning(
                        "cudaHostUnregister failed for rank=%d (code=%d)",
                        self.rank,
                        result,
                    )
            self.unregister_time_s += time.perf_counter() - _t0

        self._registered_ranges.clear()
        self.registered_bytes = 0
        self.is_pinned = False

    def _populate_worker_pages(
        self, advice: int, worker_offset: int, cpu_page_size: int
    ) -> None:
        """Populate this worker's pages."""
        _t0 = time.perf_counter()
        try:
            page_size = self.page_size
            for block in range(self.num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                assert self.mmap_obj is not None
                self.mmap_obj.madvise(advice, aligned_offset, aligned_length)
        finally:
            self.population_time_s = time.perf_counter() - _t0
            self.population_end_time = time.perf_counter()

        logger.debug(
            "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
            self.num_blocks,
            self.population_time_s,
        )

    def _run_population_in_background(
        self, advice: int, worker_offset: int, cpu_page_size: int
    ) -> None:
        try:
            self._populate_worker_pages(advice, worker_offset, cpu_page_size)
        except Exception as exc:
            self._population_error = exc
            logger.warning(
                "MADV_POPULATE_WRITE loop failed for rank=%d",
                self.rank,
                exc_info=True,
            )

    def wait_for_population(self) -> None:
        """Wait for page population and re-raise its exception, if any."""
        _t0 = time.perf_counter()
        had_thread = self._population_thread is not None
        if had_thread:
            self._population_thread.join()
            self._population_thread = None
        wait_time = time.perf_counter() - _t0
        self.population_wait_time_s += wait_time
        if had_thread:
            logger.debug(
                "Waited for mmap page population rank=%d: %.3f s",
                self.rank,
                wait_time,
            )

        if self._population_error is not None:
            error = self._population_error
            self._population_error = None
            raise error

    def create_next_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_blocks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per block for this  tensor.
        """
        assert self.rank is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        try:
            self.wait_for_population()
        except Exception as error:
            logger.warning(
                "Background mmap page population failed before cleanup for rank=%d",
                self.rank,
                exc_info=(type(error), error, error.__traceback__),
            )

        if self._registered_ranges or self.is_pinned:
            self.unregister_host_registration()
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
