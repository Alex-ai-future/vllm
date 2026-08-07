# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
import os
import time
from typing import Any

import torch

from vllm.distributed.device_communicators.shm_broadcast import (
    check_shm_free_space,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu.host_memory import (
    create_rank_local_alias,
    create_rank_local_alias_view,
    destroy_rank_local_alias,
    rank_local_alias_supported,
)

logger = init_logger(__name__)


def _cuda_result_code(result: Any) -> int:
    value = getattr(result, "value", result)
    return int(value)


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
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self._cpu_page_size = cpu_page_size
        self.total_size_bytes = self.num_blocks * self._row_stride

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each block row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        try:
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
        except FileExistsError:
            # Joiner path — another worker won O_EXCL. Reopen and wait
            # for the file to reach expected size.
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            try:
                _wait_for_file_size(self.fd, self.total_size_bytes)
            except (TimeoutError, OSError):
                os.close(self.fd)
                raise
            logger.info("Opened existing mmap file %s", self.mmap_path)
        else:
            # Creator path. We won O_EXCL, so we own the file: any
            # failure here must clean up so concurrent joiners don't
            # land on a 0-byte stub and spin in _wait_for_file_size
            # for the full 30 s timeout.
            try:
                check_shm_free_space(self.total_size_bytes)
                os.ftruncate(self.fd, self.total_size_bytes)
            except (RuntimeError, OSError):
                os.unlink(self.mmap_path)
                os.close(self.fd)
                raise
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
        _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
        if rank is not None:
            # Populate only this worker's pages (one slot per block row).
            worker_offset = rank * cpu_page_size
            _t0 = time.perf_counter()
            page_size = self.page_size
            for block in range(num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                self.mmap_obj.madvise(
                    _MADV_POPULATE_WRITE, aligned_offset, aligned_length
                )
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
                num_blocks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            self.mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self.is_pinned: bool = False
        self._pinned_ptr: int | None = None
        self._alias_base: int | None = None
        self._alias_size = 0
        self._alias_buffer: Any | None = None
        self._alias_tensor: torch.Tensor | None = None
        self._alias_views: list[torch.Tensor] = []
        self._alias_offset = 0

    def create_next_transfer_view(self, tensor_page_size: int) -> torch.Tensor:
        """Create the next transfer view, using the alias when available."""
        if self._alias_tensor is not None:
            alias_view, new_offset = create_rank_local_alias_view(
                self._alias_tensor,
                self.num_blocks,
                self._cpu_page_size,
                tensor_page_size,
                self._alias_offset,
            )
            self._alias_offset = new_offset
            self._alias_views.append(alias_view)
            return alias_view
        return self.create_next_view(tensor_page_size)

    def _register_rank_local_alias(self, cudart: Any) -> None:
        assert self.fd is not None
        assert self.rank is not None
        alias_size = self.num_blocks * self._cpu_page_size
        alias_base, alias_buffer, alias_tensor = create_rank_local_alias(
            self.fd,
            self.num_blocks,
            self._row_stride,
            self._cpu_page_size,
            self.rank,
        )
        try:
            result = cudart.cudaHostRegister(alias_base, alias_size, 0)
            code = _cuda_result_code(result)
            if code != 0:
                raise RuntimeError(
                    f"cudaHostRegister rank-local alias failed for "
                    f"rank={self.rank} (code={code})"
                )
        except BaseException:
            alias_tensor = None
            alias_buffer = None
            destroy_rank_local_alias(alias_base, alias_size)
            raise

        self._alias_base = alias_base
        self._alias_size = alias_size
        self._alias_buffer = alias_buffer
        self._alias_tensor = alias_tensor
        self._pinned_ptr = alias_base
        self.is_pinned = True
        logger.debug(
            "cudaHostRegister rank=%d rank-local alias %.2f GB (source %.2f GB)",
            self.rank,
            alias_size / 1e9,
            self.total_size_bytes / 1e9,
        )

    def _register_full_mmap(self, cudart: Any) -> None:
        base_ptr = self._base.data_ptr()
        result = cudart.cudaHostRegister(base_ptr, self.total_size_bytes, 0)
        code = _cuda_result_code(result)
        if code != 0:
            logger.warning(
                "cudaHostRegister failed for rank=%d (code=%d) — "
                "transfers will still work but may be slower (unpinned DMA)",
                self.rank,
                code,
            )
            return

        self._pinned_ptr = base_ptr
        self.is_pinned = True
        logger.debug(
            "cudaHostRegister rank=%d full mmap %.2f GB",
            self.rank,
            self.total_size_bytes / 1e9,
        )

    def pin_cuda_host_memory(self) -> None:
        """Register the region with CUDA using its layout-specific path."""
        if self.is_pinned:
            return
        if not current_platform.is_cuda_alike():
            logger.info(
                "Skipping mmap host registration on %s; cudaHostRegister is "
                "only available on CUDA/ROCm.",
                current_platform.device_name,
            )
            return

        cudart = torch.cuda.cudart()
        if current_platform.is_cuda() and self.rank is not None:
            assert self.fd is not None
            if rank_local_alias_supported(
                self.fd,
                self.num_blocks,
                self._row_stride,
                self._cpu_page_size,
                self.rank,
            ):
                self._register_rank_local_alias(cudart)
                return
        self._register_full_mmap(cudart)

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
        if self.is_pinned and self._pinned_ptr is not None:
            if current_platform.is_cuda_alike():
                result = torch.cuda.cudart().cudaHostUnregister(self._pinned_ptr)
                code = _cuda_result_code(result)
                if code != 0:
                    logger.warning(
                        "cudaHostUnregister failed for rank=%d (code=%d)",
                        self.rank,
                        code,
                    )
            self.is_pinned = False
            self._pinned_ptr = None
        if self._alias_base is not None:
            self._alias_views.clear()
            self._alias_tensor = None
            self._alias_buffer = None
            destroy_rank_local_alias(self._alias_base, self._alias_size)
            self._alias_base = None
            self._alias_size = 0
            self._alias_offset = 0
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
