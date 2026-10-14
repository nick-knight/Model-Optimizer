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

"""Adapters for applying per-head transforms at attention boundaries."""

from torch import Tensor, nn

__all__ = [
    "AttentionKeyValueHeadTransform",
    "FlattenedAttentionKeyValueHeadTransform",
    "LinearOutputHeadTransform",
    "set_attention_key_value_transforms",
    "set_flattened_attention_key_value_transforms",
    "set_linear_head_transform",
]


class AttentionKeyValueHeadTransform(nn.Module):
    """Apply same-shaped transforms to explicit key and value head tensors."""

    def __init__(
        self,
        key_transform: nn.Module,
        value_transform: nn.Module,
        *,
        head_axis: int = -2,
    ) -> None:
        """Initialize transforms whose inputs have the head axis moved first."""
        super().__init__()
        self.key_transform = key_transform
        self.value_transform = value_transform
        self.head_axis = head_axis

    def forward(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Transform key and value while preserving their public layouts."""
        return self._apply_transform(self.key_transform, key), self._apply_transform(
            self.value_transform, value
        )

    def _apply_transform(self, transform: nn.Module, values: Tensor) -> Tensor:
        if values.ndim < 2:
            raise ValueError(
                "attention tensors must have a head axis and a feature axis"
            )
        head_axis = self.head_axis % values.ndim
        if head_axis == values.ndim - 1:
            raise ValueError("head_axis cannot be the final feature axis")
        per_head = values.movedim(head_axis, 0)
        transformed = transform(per_head)
        if transformed.shape != per_head.shape:
            raise ValueError(
                "attention head transforms must preserve shape; "
                f"got {tuple(transformed.shape)} instead of {tuple(per_head.shape)}"
            )
        return transformed.movedim(0, head_axis)


def _apply_attention_key_value_transforms(
    module: nn.Module,
    inputs: tuple[Tensor, ...],
    kwargs: dict,
) -> tuple[tuple[Tensor, ...], dict]:
    mutable_inputs = list(inputs)
    kwargs = kwargs.copy()

    def get_argument(index: int, name: str) -> Tensor:
        if index < len(mutable_inputs):
            return mutable_inputs[index]
        if name in kwargs:
            return kwargs[name]
        raise TypeError(f"attention forward call is missing the {name!r} argument")

    key = get_argument(1, "key")
    value = get_argument(2, "value")
    key, value = module._modelopt_key_value_head_transform(key, value)
    if len(mutable_inputs) > 1:
        mutable_inputs[1] = key
    else:
        kwargs["key"] = key
    if len(mutable_inputs) > 2:
        mutable_inputs[2] = value
    else:
        kwargs["value"] = value
    return tuple(mutable_inputs), kwargs


def set_attention_key_value_transforms(
    attention: nn.Module,
    key_transform: nn.Module,
    value_transform: nn.Module,
    *,
    head_axis: int = -2,
) -> None:
    """Attach or replace persistent K/V transforms at an explicit-QKV attention boundary.

    This adapter targets modules such as Megatron-Core's ``TEDotProductAttention``, whose first
    three forward arguments are query, key, and value. The transforms see tensors with the head
    axis moved to axis zero and must preserve shape.
    """
    adapter = AttentionKeyValueHeadTransform(
        key_transform,
        value_transform,
        head_axis=head_axis,
    )
    _set_attention_key_value_adapter(attention, adapter)


class FlattenedAttentionKeyValueHeadTransform(nn.Module):
    """Apply per-head transforms to flattened attention K/V tensors."""

    def __init__(
        self,
        key_transform: nn.Module,
        value_transform: nn.Module,
        *,
        num_heads: int,
        head_dim: int,
        value_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim must be positive")
        if value_head_dim is None:
            value_head_dim = head_dim
        if value_head_dim < 1:
            raise ValueError("value_head_dim must be positive")
        self.key_transform = key_transform
        self.value_transform = value_transform
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim

    def forward(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Transform flattened K/V tensors while preserving their layouts."""
        return self._apply(self.key_transform, key, self.head_dim), self._apply(
            self.value_transform, value, self.value_head_dim
        )

    def _apply(self, transform: nn.Module, values: Tensor, head_dim: int) -> Tensor:
        expected = self.num_heads * head_dim
        if values.shape[-1] != expected:
            raise ValueError(
                f"expected a final dimension of {expected}, got {values.shape[-1]}"
            )
        per_head = values.unflatten(-1, (self.num_heads, head_dim))
        transformed = transform(per_head.movedim(-2, 0)).movedim(0, -2)
        if transformed.shape != per_head.shape:
            raise ValueError(
                "attention head transforms must preserve shape; "
                f"got {tuple(transformed.shape)} instead of {tuple(per_head.shape)}"
            )
        return transformed.flatten(-2)


def set_flattened_attention_key_value_transforms(
    attention: nn.Module,
    key_transform: nn.Module,
    value_transform: nn.Module,
    *,
    num_heads: int,
    head_dim: int,
    value_head_dim: int | None = None,
) -> None:
    """Attach K/V transforms where attention inputs flatten heads and features."""
    adapter = FlattenedAttentionKeyValueHeadTransform(
        key_transform,
        value_transform,
        num_heads=num_heads,
        head_dim=head_dim,
        value_head_dim=value_head_dim,
    )
    _set_attention_key_value_adapter(attention, adapter)


def _set_attention_key_value_adapter(attention: nn.Module, adapter: nn.Module) -> None:
    if hasattr(attention, "_modelopt_key_value_head_transform"):
        attention._modelopt_key_value_head_transform = adapter
        return
    attention.add_module("_modelopt_key_value_head_transform", adapter)
    handle = attention.register_forward_pre_hook(
        _apply_attention_key_value_transforms,
        with_kwargs=True,
    )
    object.__setattr__(attention, "_modelopt_key_value_head_transform_handle", handle)


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
            raise ValueError(
                f"expected a final dimension of {expected}, got {output.shape[-1]}"
            )
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
