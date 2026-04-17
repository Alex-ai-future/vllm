# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GELU activation function IR ops."""

import pytest
import torch

import vllm.kernels  # noqa: F401
from vllm import ir
from vllm.platforms import current_platform


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 8), (4, 16), (17, 64)])
class TestGeluOps:
    """Tests for GELU IR ops."""

    @pytest.mark.parametrize(
        "gelu_op_name", ["gelu_new", "gelu_fast", "quick_gelu"]
    )
    def test_native_semantics(self, gelu_op_name, dtype, shape):
        """Test that IR op matches native implementation."""
        gelu_op = getattr(ir.ops, gelu_op_name)
        x = torch.randn(*shape, dtype=dtype)

        out_ir = gelu_op(x)
        out_native = gelu_op.impls["native"].impl_fn(x)

        torch.testing.assert_close(out_ir, out_native)

    @pytest.mark.parametrize("provider", ["vllm_c"])
    @pytest.mark.parametrize(
        "gelu_op_name", ["gelu_new", "gelu_fast", "quick_gelu"]
    )
    def test_vllm_c_impl(self, gelu_op_name, provider, dtype, shape):
        """Test vllm_c implementation correctness."""
        gelu_op = getattr(ir.ops, gelu_op_name)
        impl = gelu_op.impls[provider]

        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        x = torch.randn(
            *shape, dtype=dtype, device=current_platform.device_type
        )
        out_impl = impl.impl_fn(x)
        out_native = gelu_op.impls["native"].impl_fn(x)

        torch.testing.assert_close(out_impl, out_native)

        # Verify dispatch matches direct call
        with gelu_op.set_priority([provider, "native"]):
            out_dispatch = gelu_op(x)
        torch.testing.assert_close(out_dispatch, out_impl, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "gelu_op_name", ["gelu_new", "gelu_fast", "quick_gelu"]
    )
    def test_torch_opcheck(self, gelu_op_name, dtype, shape):
        """Test torch op integration."""
        gelu_op = getattr(ir.ops, gelu_op_name)
        x = torch.randn(*shape, dtype=dtype)

        with gelu_op.set_priority(["native"]):
            torch.library.opcheck(
                torch.ops.vllm_ir.__getattr__(gelu_op_name), (x,)
            )
