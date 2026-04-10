# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

import vllm.kernels  # noqa: F401 to register kernels
from vllm import ir
from vllm.compilation.passes.ir.lowering_pass import (
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config
from vllm.ir import ops
from vllm.platforms import current_platform

from ...backend import TestBackend


class Model(nn.Module):
    def __init__(self, hidden_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    def forward(self, x):
        x1 = x + 4.0
        x2 = ops.rms_norm(x1, self.weight, 1e-5)
        x3 = x2 * 5.0
        # no weight
        x4 = ops.rms_norm(x3, None, 1e-5)
        x5 = x4 / 2.0
        # dispatch to native due to variance_size parameter
        x6 = ops.rms_norm(x5, self.weight, 1e-5, self.hidden_size // 2)
        return x6 + 3.0


@pytest.mark.parametrize("rms_provider", ops.rms_norm.supported_providers())
def test_lowering_rms_norm(rms_provider, default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = Model()
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    with (
        ops.rms_norm.set_priority([rms_provider, "native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    selected = lowering_pass.selected_impls["rms_norm"]
    assert len(selected) == 3
    assert selected["rms_norm"] == rms_provider
    assert selected["rms_norm_1"] == rms_provider
    assert selected["rms_norm_2"] == "native"

    # Compiled function guards on global value, avoid recompilation
    with ir.enable_torch_wrap(True):
        output2 = compiled_model(x)

    torch.testing.assert_close(output_unlowered, output)
    torch.testing.assert_close(output_unlowered, output2)


#===================
# GELU Lowering Tests
#===================


class GeluModel(nn.Module):
    """Simple model using GELU IR ops."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        x1 = ops.gelu_new(x)
        x2 = ops.gelu_fast(x1)
        x3 = ops.quick_gelu(x2)
        return x3 + 1.0


class GeluMixedModel(nn.Module):
    """Model mixing GELU IR ops with RMSNorm."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    def forward(self, x):
        x1 = ops.gelu_new(x)
        x2 = ops.rms_norm(x1, self.weight, 1e-5)
        x3 = ops.gelu_fast(x2)
        return x3


@pytest.mark.parametrize("gelu_op_name", ["gelu_new", "gelu_fast", "quick_gelu"])
def test_lowering_gelu_basic(gelu_op_name, default_vllm_config):
    """Test basic lowering for individual GELU ops."""
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)

    gelu_op = getattr(ops, gelu_op_name)
    supported_providers = gelu_op.supported_providers()

    # Test with each supported provider
    for provider in supported_providers:

        class SimpleModel(nn.Module):
            def forward(self, x):
                return gelu_op(x)

        model = SimpleModel()
        x = torch.randn(8, 16, dtype=torch.bfloat16)

        with gelu_op.set_priority([provider, "native"]):
            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            output = compiled_model(x)

        # Check that the correct implementation was selected
        selected = lowering_pass.selected_impls[gelu_op_name]
        assert len(selected) == 1
        assert selected["gelu_new" if gelu_op_name == "gelu_new" else gelu_op_name] == provider

        # Verify output is reasonable (not NaN or Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="vllm_c kernels only supported on CUDA-alike platforms",
)
def test_lowering_gelu_vllm_c_priority(default_vllm_config):
    """Test that GELU ops use vllm_c when set in priority."""
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)

    model = GeluModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)

    with (
        ops.gelu_new.set_priority(["vllm_c"]),
        ops.gelu_fast.set_priority(["vllm_c"]),
        ops.quick_gelu.set_priority(["vllm_c"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        output = compiled_model(x)

    # Check that vllm_c was selected for all GELU ops
    assert lowering_pass.selected_impls["gelu_new"]["gelu_new"] == "vllm_c"
    assert lowering_pass.selected_impls["gelu_fast"]["gelu_fast"] == "vllm_c"
    assert lowering_pass.selected_impls["quick_gelu"]["quick_gelu"] == "vllm_c"

    # Verify correctness by comparing to eager
    with ir.enable_torch_wrap(True):
        expected = model(x)

    torch.testing.assert_close(output, expected)


def test_lowering_gelu_mixed_model(default_vllm_config):
    """Test lowering with mixed GELU and RMSNorm ops."""
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    backend_unlowered = TestBackend()

    model = GeluMixedModel()
    x = torch.randn(8, 16, dtype=torch.bfloat16)

    # Set priority for all ops
    providers_to_test = ["vllm_c"] if current_platform.is_cuda_alike() else ["native"]

    with (
        ops.gelu_new.set_priority(providers_to_test + ["native"]),
        ops.gelu_fast.set_priority(providers_to_test + ["native"]),
        ops.rms_norm.set_priority(["vllm_c", "native"]) if current_platform.is_cuda_alike() else ops.rms_norm.set_priority(["native"]),
        ir.enable_torch_wrap(True),
    ):
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        compiled_unlowered_model = torch.compile(
            model, backend=backend_unlowered, fullgraph=True
        )
        output = compiled_model(x)
        output_unlowered = compiled_unlowered_model(x)

    # Check implementations were selected
    assert "gelu_new" in lowering_pass.selected_impls
    assert "gelu_fast" in lowering_pass.selected_impls
    assert "rms_norm" in lowering_pass.selected_impls

    # Verify correctness with relaxed tolerances for bfloat16
    torch.testing.assert_close(
        output_unlowered, output, rtol=0.1, atol=0.01
    )
