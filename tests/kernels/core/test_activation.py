# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import assert_pluggable_layer_calls_ir_op, opcheck
from vllm import ir
from vllm.model_executor.layers.activation import (
    FatreluAndMul,
    GELU,
    GeluAndMul,
    MulAndSilu,
    FastGELU,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SwigluOAIAndMul,
    SwigluStepAndMul,
    swiglustep_and_mul_triton,
)
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


# TODO: This test validates kernel output correctness, which overlaps with
# the ir.ops routing tests. As custom ops are migrated to PluggableLayer,
# remove cases from here and add them to ACTIVATION_LAYER_CONFIGS.
# Eventually, test_activation_ir_op_routing will be the single source of truth.
@pytest.mark.parametrize(
    "activation",
    [
        "silu_and_mul",
        "mul_and_silu",
        "fatrelu",
        "swigluoai_and_mul",
        "swiglustep_and_mul",
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    default_vllm_config,
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu_and_mul":
        layer = SiluAndMul(compile_native=False)
        fn = torch.ops._C.silu_and_mul
    if activation == "mul_and_silu":
        layer = MulAndSilu()
        fn = torch.ops._C.mul_and_silu
    elif activation == "fatrelu":
        threshold = random.uniform(0, 1)
        layer = FatreluAndMul(threshold)
        fn = torch.ops._C.fatrelu_and_mul
    elif activation == "swigluoai_and_mul":
        layer = SwigluOAIAndMul()
        fn = torch.ops._C.swigluoai_and_mul
    elif activation == "swiglustep_and_mul":
        layer = SwigluStepAndMul()
        fn = swiglustep_and_mul_triton
    out = layer(x)
    ref_out = layer.forward_native(x)
    if activation in ["swigluoai_and_mul", "swiglustep_and_mul"]:
        rtol = {
            # For fp16, change the relative tolerance from 1e-3 to 2e-3
            torch.float16: 2e-3,
            torch.bfloat16: 2e-2,
            torch.float: 1.3e-6,
        }

        def _get_rtol(output) -> float:
            return rtol[output.dtype]

        torch.testing.assert_close(
            out, ref_out, atol=get_default_atol(out), rtol=_get_rtol(out)
        )
    else:
        # The SiluAndMul, MulAndSilu, GELU and FatReLU implementations are
        # equivalent to the native PyTorch implementations, so we can do exact
        # comparison.
        torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)

    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation == "fatrelu":
        opcheck(fn, (out, x, threshold))
    elif activation == "swigluoai_and_mul":
        opcheck(fn, (out, x, layer.alpha, layer.limit))
    elif activation != "swiglustep_and_mul":
        opcheck(fn, (out, x))


# Activation layer test configs (currently only GELU variants)
ACTIVATION_LAYER_CONFIGS = [
    ("GELU", GELU, ir.ops.gelu, {}),
    ("GeluAndMul_none", GeluAndMul, ir.ops.gelu_and_mul, {"approximate": "none"}),
    ("GeluAndMul_tanh", GeluAndMul, ir.ops.gelu_and_mul, {"approximate": "tanh"}),
    ("NewGELU", NewGELU, ir.ops.gelu_new, {}),
    ("FastGELU", FastGELU, ir.ops.gelu_fast, {}),
    ("QuickGELU", QuickGELU, ir.ops.quick_gelu, {}),
]


# This test verifies that PluggableLayer instances correctly dispatch to
# their corresponding ir.ops functions. Add new activations to
# ACTIVATION_LAYER_CONFIGS to automatically get test coverage.
@pytest.mark.parametrize("name, layer_cls, ir_op, kwargs", ACTIVATION_LAYER_CONFIGS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_activation_ir_op_routing(
    default_vllm_config,
    name: str,
    layer_cls: type,
    ir_op,
    kwargs: dict,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test that activation PluggableLayer instances call the corresponding ir.ops function."""
    set_random_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = layer_cls(**kwargs)
    assert_pluggable_layer_calls_ir_op(layer, ir_op, x)
