# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GELU activation function IR ops."""

import pytest
import torch
import torch.nn.functional as F

from vllm import ir
from vllm.ir.ops import gelu_new, gelu_fast, quick_gelu
from vllm.platforms import current_platform


class TestGeluNew:
    """Tests for gelu_new IR op."""

    def test_gelu_new_semantics(self):
        """Test that gelu_new IR op matches native PyTorch implementation."""
        x = torch.randn(4, 8, dtype=torch.float32)

        # IR op should match native semantics
        out_ir = gelu_new(x)
        out_native = gelu_new.impls["native"].impl_fn(x)

        torch.testing.assert_close(out_ir, out_native)

    def test_gelu_new_formula(self):
        """Test that gelu_new matches the expected formula."""
        x = torch.randn(4, 8, dtype=torch.float32)

        out = gelu_new(x)

        # Expected formula: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c = (2.0 / torch.pi) ** 0.5
        expected = 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))

        torch.testing.assert_close(out, expected)

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="vllm_c kernels only supported on CUDA-alike platforms",
    )
    def test_gelu_new_vllm_c_kernel(self):
        """Test gelu_new with vllm_c implementation."""
        x = torch.randn(4, 8, dtype=torch.float32, device=current_platform.device_type)

        with gelu_new.set_priority(["vllm_c", "native"]):
            out = gelu_new(x)

        # Should match native semantics
        expected = gelu_new.impls["native"].impl_fn(x)
        torch.testing.assert_close(out, expected)


class TestGeluFast:
    """Tests for gelu_fast IR op."""

    def test_gelu_fast_semantics(self):
        """Test that gelu_fast IR op matches native PyTorch implementation."""
        x = torch.randn(4, 8, dtype=torch.float32)

        # IR op should match native semantics
        out_ir = gelu_fast(x)
        out_native = gelu_fast.impls["native"].impl_fn(x)

        torch.testing.assert_close(out_ir, out_native)

    def test_gelu_fast_formula(self):
        """Test that gelu_fast matches the expected formula."""
        x = torch.randn(4, 8, dtype=torch.float32)

        out = gelu_fast(x)

        # Expected formula: 0.5 * x * (1.0 + tanh(x * 0.7978845608 * (1.0 + 0.044715 * x^2)))
        expected = 0.5 * x * (
            1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
        )

        torch.testing.assert_close(out, expected)

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="vllm_c kernels only supported on CUDA-alike platforms",
    )
    def test_gelu_fast_vllm_c_kernel(self):
        """Test gelu_fast with vllm_c implementation."""
        x = torch.randn(4, 8, dtype=torch.float32, device=current_platform.device_type)

        with gelu_fast.set_priority(["vllm_c", "native"]):
            out = gelu_fast(x)

        # Should match native semantics
        expected = gelu_fast.impls["native"].impl_fn(x)
        torch.testing.assert_close(out, expected)


class TestQuickGelu:
    """Tests for quick_gelu IR op."""

    def test_quick_gelu_semantics(self):
        """Test that quick_gelu IR op matches native PyTorch implementation."""
        x = torch.randn(4, 8, dtype=torch.float32)

        # IR op should match native semantics
        out_ir = quick_gelu(x)
        out_native = quick_gelu.impls["native"].impl_fn(x)

        torch.testing.assert_close(out_ir, out_native)

    def test_quick_gelu_formula(self):
        """Test that quick_gelu matches the expected formula."""
        x = torch.randn(4, 8, dtype=torch.float32)

        out = quick_gelu(x)

        # Expected formula: x * sigmoid(1.702 * x)
        expected = x * torch.sigmoid(1.702 * x)

        torch.testing.assert_close(out, expected)

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="vllm_c kernels only supported on CUDA-alike platforms",
    )
    def test_quick_gelu_vllm_c_kernel(self):
        """Test quick_gelu with vllm_c implementation."""
        x = torch.randn(4, 8, dtype=torch.float32, device=current_platform.device_type)

        with quick_gelu.set_priority(["vllm_c", "native"]):
            out = quick_gelu(x)

        # Should match native semantics
        expected = quick_gelu.impls["native"].impl_fn(x)
        torch.testing.assert_close(out, expected)


class TestGeluDispatch:
    """Tests for GELU IR op dispatching."""

    def test_default_priority(self):
        """Test that GELU ops use native implementation by default."""
        x = torch.randn(4, 8, dtype=torch.float32)

        # All GELU ops should have empty priority by default
        assert gelu_new.get_priority() == []
        assert gelu_fast.get_priority() == []
        assert quick_gelu.get_priority() == []

        # Should dispatch to native
        assert gelu_new.dispatch(x) is gelu_new.impls["native"]
        assert gelu_fast.dispatch(x) is gelu_fast.impls["native"]
        assert quick_gelu.dispatch(x) is quick_gelu.impls["native"]

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="vllm_c kernels only supported on CUDA-alike platforms",
    )
    def test_vllm_c_priority(self):
        """Test that GELU ops use vllm_c when set in priority."""
        x = torch.randn(4, 8, dtype=torch.float32)

        with gelu_new.set_priority(["vllm_c"]):
            assert gelu_new.dispatch(x).provider == "vllm_c"

        with gelu_fast.set_priority(["vllm_c"]):
            assert gelu_fast.dispatch(x).provider == "vllm_c"

        with quick_gelu.set_priority(["vllm_c"]):
            assert quick_gelu.dispatch(x).provider == "vllm_c"

    def test_supported_providers(self):
        """Test that vllm_c is listed as supported provider on CUDA platforms."""
        if current_platform.is_cuda_alike():
            assert "vllm_c" in gelu_new.supported_providers()
            assert "vllm_c" in gelu_fast.supported_providers()
            assert "vllm_c" in quick_gelu.supported_providers()
        else:
            # On non-CUDA platforms, vllm_c should not be supported
            assert "vllm_c" not in gelu_new.supported_providers()
            assert "vllm_c" not in gelu_fast.supported_providers()
            assert "vllm_c" not in quick_gelu.supported_providers()


class TestGeluTorchCompile:
    """Tests for GELU IR ops with torch.compile."""

    def test_gelu_new_compile_basic(self):
        """Test that gelu_new can be compiled with torch.compile."""

        def fn(x):
            return gelu_new(x)

        x = torch.randn(4, 8, dtype=torch.float32)
        compiled_fn = torch.compile(fn, fullgraph=True)

        out_compiled = compiled_fn(x)
        out_eager = fn(x)

        torch.testing.assert_close(out_compiled, out_eager)

    def test_gelu_fast_compile_basic(self):
        """Test that gelu_fast can be compiled with torch.compile."""

        def fn(x):
            return gelu_fast(x)

        x = torch.randn(4, 8, dtype=torch.float32)
        compiled_fn = torch.compile(fn, fullgraph=True)

        out_compiled = compiled_fn(x)
        out_eager = fn(x)

        torch.testing.assert_close(out_compiled, out_eager)

    def test_quick_gelu_compile_basic(self):
        """Test that quick_gelu can be compiled with torch.compile."""

        def fn(x):
            return quick_gelu(x)

        x = torch.randn(4, 8, dtype=torch.float32)
        compiled_fn = torch.compile(fn, fullgraph=True)

        out_compiled = compiled_fn(x)
        out_eager = fn(x)

        torch.testing.assert_close(out_compiled, out_eager)

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="vllm_c kernels only supported on CUDA-alike platforms",
    )
    def test_gelu_new_compile_with_vllm_c(self):
        """Test gelu_new compilation with vllm_c priority."""

        def fn(x):
            return gelu_new(x)

        x = torch.randn(4, 8, dtype=torch.float32, device=current_platform.device_type)

        with gelu_new.set_priority(["vllm_c"]):
            compiled_fn = torch.compile(fn, fullgraph=True)
            out_compiled = compiled_fn(x)
            out_eager = fn(x)

        torch.testing.assert_close(out_compiled, out_eager)
