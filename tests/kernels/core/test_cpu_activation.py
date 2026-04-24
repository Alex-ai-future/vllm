# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE: This file contains CPU-only activation kernel tests.
# GELU-related tests have been removed because GeluAndMul and other GELU
# activations are now PluggableLayers that call ir.ops. The vllm_c kernel
# implementations for GELU are CUDA-only (marked with CUDA_ALIKE in vllm_c.py).
# As GELU and other activations gain CPU ir.ops implementations, add routing
# tests to test_activation.py's test_activation_ir_op_routing.

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import opcheck
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm.model_executor.layers.activation import SiluAndMul

DTYPES = [torch.bfloat16, torch.float32]
NUM_TOKENS = [7, 83]
D = [512, 2048]
SEEDS = [0]


@pytest.mark.parametrize(
    ("activation_cls", "fn"),
    [
        (SiluAndMul, torch.ops._C.silu_and_mul),
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_cpu_silu_and_mul(
    default_vllm_config,
    activation_cls: type[torch.nn.Module],
    fn: object,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    set_random_seed(seed)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)

    layer = activation_cls()
    out = layer(x)
    ref_out = layer.forward_native(x)

    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    raw_out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    opcheck(fn, (raw_out, x))