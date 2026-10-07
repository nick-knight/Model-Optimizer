# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapters for applying per-head transforms to flattened linear outputs."""

from torch import Tensor, nn

__all__ = ["LinearOutputHeadTransform", "set_linear_head_transform"]


class LinearOutputHeadTransform(nn.Module):
    """Reshape a flattened projection output and apply a transform per head."""

    def __init__(self, transform: nn.Module, num_heads: int, head_dim: int) -> None:
        """Initialize a same-shaped output transform for a multi-head linear layer."""
        super().__init__()
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim must be positive")
        self.transform = transform
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, output: Tensor) -> Tensor:
        """Apply the transform with heads moved to its leading tensor axis."""
        expected = self.num_heads * self.head_dim
        if output.shape[-1] != expected:
            raise ValueError(f"expected a final dimension of {expected}, got {output.shape[-1]}")
        per_head = output.unflatten(-1, (self.num_heads, self.head_dim))
        transformed = self.transform(per_head.movedim(-2, 0)).movedim(0, -2)
        if transformed.shape != per_head.shape:
            raise ValueError(
                "head transform must preserve shape; "
                f"got {tuple(transformed.shape)} instead of {tuple(per_head.shape)}"
            )
        return transformed.flatten(-2)


def _apply_output_head_transform(
    module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
) -> Tensor:
    del inputs
    return module._modelopt_output_head_transform(output)


def set_linear_head_transform(
    linear: nn.Linear, transform: nn.Module, num_heads: int, head_dim: int
) -> None:
    """Attach or replace a persistent per-head transform on a linear layer's output."""
    if not isinstance(linear, nn.Linear):
        raise TypeError("linear must be a torch.nn.Linear")
    adapter = LinearOutputHeadTransform(transform, num_heads, head_dim)
    if hasattr(linear, "_modelopt_output_head_transform"):
        linear._modelopt_output_head_transform = adapter
        return
    linear.add_module("_modelopt_output_head_transform", adapter)
    handle = linear.register_forward_hook(_apply_output_head_transform)
    object.__setattr__(linear, "_modelopt_output_head_transform_handle", handle)
