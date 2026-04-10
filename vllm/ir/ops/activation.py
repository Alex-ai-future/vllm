# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import torch
from torch import Tensor

from ..op import register_op

c_gelu_new = math.sqrt(2.0 / math.pi)


@register_op
def gelu_new(x: Tensor) -> Tensor:
    """
    New GELU activation function.
    
    Formula: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    This is the GELU approximation used in GPT-2 and other transformer models.
    """
    return 0.5 * x * (1.0 + torch.tanh(c_gelu_new * (x + 0.044715 * torch.pow(x, 3.0))))


@register_op
def gelu_fast(x: Tensor) -> Tensor:
    """
    Fast GELU activation function.
    
    Formula: 0.5 * x * (1.0 + tanh(x * 0.7978845608 * (1.0 + 0.044715 * x^2)))
    
    A computationally efficient approximation of the GELU function.
    """
    return 0.5 * x * (
        1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))
    )


@register_op
def quick_gelu(x: Tensor) -> Tensor:
    """
    Quick GELU activation function.
    
    Formula: x * sigmoid(1.702 * x)
    
    A fast approximation of GELU used in various transformer models.
    Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    """
    return x * torch.sigmoid(1.702 * x)
