# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level host-memory helpers used by CPU KV offloading."""

import ctypes
import mmap
import sys
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


_LIBC: ctypes.CDLL | None = None
_PROT_NONE = 0
_PROT_READ = 1
_PROT_WRITE = 2
_MAP_ANONYMOUS = getattr(mmap, "MAP_ANONYMOUS", 0x20)
_MAP_FIXED = getattr(mmap, "MAP_FIXED", 0x10)
_MAP_FAILED = ctypes.c_void_p(-1).value


def _get_libc() -> ctypes.CDLL:
    global _LIBC
    if _LIBC is None:
        if sys.platform != "linux":
            raise OSError("rank-local mmap aliases require Linux")
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        libc.mmap.restype = ctypes.c_void_p
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap.restype = ctypes.c_int
        _LIBC = libc
    return _LIBC


def _native_mmap(
    address: int | None,
    size: int,
    protection: int,
    flags: int,
    fd: int,
    offset: int,
) -> int:
    """Call mmap(2) and return the mapped address."""
    requested = None if address is None else ctypes.c_void_p(address)
    result = _get_libc().mmap(
        requested,
        size,
        protection,
        flags,
        fd,
        offset,
    )
    mapped = ctypes.cast(result, ctypes.c_void_p).value
    if mapped is None or mapped == _MAP_FAILED:
        error = ctypes.get_errno()
        raise OSError(error, "mmap failed")
    return mapped


def _native_munmap(address: int, size: int) -> None:
    """Call munmap(2) for a range previously returned by _native_mmap."""
    if _get_libc().munmap(ctypes.c_void_p(address), size) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "munmap failed")


def rank_local_alias_supported(
    fd: int,
    num_blocks: int,
    row_stride: int,
    slot_size: int,
    rank: int,
) -> bool:
    """Return whether the host layout supports a rank-local alias."""
    worker_offset = rank * slot_size
    page_size = mmap.PAGESIZE
    return (
        sys.platform == "linux"
        and num_blocks > 0
        and fd >= 0
        and slot_size > 0
        and slot_size % page_size == 0
        and row_stride % page_size == 0
        and row_stride > slot_size
        and worker_offset >= 0
        and worker_offset + slot_size <= row_stride
    )


def create_rank_local_alias(
    fd: int,
    num_blocks: int,
    row_stride: int,
    slot_size: int,
    rank: int,
) -> tuple[int, Any, torch.Tensor]:
    """Map one rank's interleaved slots into a contiguous virtual range."""
    if not rank_local_alias_supported(fd, num_blocks, row_stride, slot_size, rank):
        raise RuntimeError("rank-local alias layout is not supported")

    alias_size = num_blocks * slot_size
    worker_offset = rank * slot_size
    alias_base = _native_mmap(
        None,
        alias_size,
        _PROT_NONE,
        mmap.MAP_PRIVATE | _MAP_ANONYMOUS,
        -1,
        0,
    )
    alias_buffer: Any | None = None
    alias_tensor: torch.Tensor | None = None
    try:
        for block in range(num_blocks):
            alias_offset = block * slot_size
            file_offset = block * row_stride + worker_offset
            mapped = _native_mmap(
                alias_base + alias_offset,
                slot_size,
                _PROT_READ | _PROT_WRITE,
                mmap.MAP_SHARED | _MAP_FIXED,
                fd,
                file_offset,
            )
            if mapped != alias_base + alias_offset:
                raise RuntimeError(
                    f"rank-local alias mapped at {mapped:#x}, expected "
                    f"{alias_base + alias_offset:#x}"
                )
        alias_buffer = (ctypes.c_char * alias_size).from_address(alias_base)
        alias_tensor = torch.frombuffer(alias_buffer, dtype=torch.int8)
        return alias_base, alias_buffer, alias_tensor
    except BaseException:
        alias_tensor = None
        alias_buffer = None
        destroy_rank_local_alias(alias_base, alias_size)
        raise


def create_rank_local_alias_view(
    alias_tensor: torch.Tensor,
    num_blocks: int,
    slot_size: int,
    tensor_page_size: int,
    storage_offset: int,
) -> tuple[torch.Tensor, int]:
    """Create a transfer view and return it with the next slot offset."""
    new_offset = storage_offset + tensor_page_size
    if new_offset > slot_size:
        raise ValueError(
            f"Rank-local alias offset {new_offset} exceeds slot size "
            f"{slot_size} (overflowed by {new_offset - slot_size} bytes)"
        )
    alias_view = torch.as_strided(
        alias_tensor,
        size=(num_blocks, tensor_page_size),
        stride=(slot_size, 1),
        storage_offset=storage_offset,
    )
    return alias_view, new_offset


def destroy_rank_local_alias(alias_base: int | None, alias_size: int) -> None:
    """Unmap a rank-local alias after its tensor references are released."""
    if alias_base is None:
        return
    try:
        _native_munmap(alias_base, alias_size)
    except Exception:
        logger.warning(
            "Failed to unmap rank-local alias at %#x",
            alias_base,
            exc_info=True,
        )
