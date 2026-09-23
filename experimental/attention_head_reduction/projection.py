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

"""Per-head second-moment calibration and principal-component projection."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

__all__ = ["HeadProjector", "SecondMomentAccumulator", "principal_components"]


class SecondMomentAccumulator:
    """Accumulate uncentered second moments independently for each attention head."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize empty statistics for equally sized attention heads."""
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim must be positive")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.sum = None if device is None else self._empty_sum(torch.device(device))
        self.num_samples = 0

    @torch.no_grad()
    def update(self, values: Tensor, *, head_axis: int = 0) -> None:
        """Add values shaped with a head axis and a final feature axis."""
        if values.ndim < 2:
            raise ValueError("values must have a head axis and a final feature axis")
        head_axis %= values.ndim
        if head_axis == values.ndim - 1:
            raise ValueError("head_axis cannot be the final feature axis")

        values = values.movedim(head_axis, 0)
        if values.shape[0] != self.num_heads or values.shape[-1] != self.head_dim:
            raise ValueError(
                f"expected {self.num_heads} heads of dimension {self.head_dim}, "
                f"got shape {tuple(values.shape)}"
            )

        if self.sum is None:
            self.sum = self._empty_sum(values.device)
        values = values.reshape(self.num_heads, -1, self.head_dim).to(self.sum)
        self.sum.add_(torch.bmm(values.transpose(1, 2), values))
        self.num_samples += values.shape[1]

    def compute(self) -> Tensor:
        """Return the calibrated uncentered second-moment matrices."""
        if self.num_samples == 0:
            raise RuntimeError("cannot compute second moments before observing samples")
        assert self.sum is not None
        return self.sum / self.num_samples

    def _empty_sum(self, device: torch.device) -> Tensor:
        dtype = self.dtype or (torch.float32 if device.type == "mps" else torch.float64)
        return torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)


def principal_components(second_moment: Tensor) -> tuple[Tensor, Tensor]:
    """Return eigenvalues and column eigenvectors in descending eigenvalue order."""
    if second_moment.ndim < 2 or second_moment.shape[-1] != second_moment.shape[-2]:
        raise ValueError("second_moment must contain square matrices")
    second_moment = (second_moment + second_moment.transpose(-1, -2)) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    return eigenvalues.flip(-1), eigenvectors.flip(-1)


class HeadProjector(nn.Module):
    """Emulate independently ranked per-head compression using frozen PCA bases."""

    def __init__(self, basis: Tensor, ranks: Tensor | Sequence[int]) -> None:
        """Initialize from full per-head bases and retained ranks."""
        super().__init__()
        if basis.ndim != 3 or basis.shape[-2] != basis.shape[-1]:
            raise ValueError("basis must have shape (num_heads, head_dim, head_dim)")

        ranks = torch.as_tensor(ranks, dtype=torch.int64)
        if ranks.shape != (basis.shape[0],):
            raise ValueError(f"ranks must have shape ({basis.shape[0]},)")
        rank_values = ranks.tolist()
        if any(rank < 1 or rank > basis.shape[-1] for rank in rank_values):
            raise ValueError("each rank must be between one and head_dim")

        ranks = ranks.to(basis.device)
        max_rank = max(rank_values)
        projection = basis[..., :max_rank].clone()
        retained = torch.arange(max_rank, device=basis.device).unsqueeze(0) < ranks.unsqueeze(1)
        projection.mul_(retained.unsqueeze(1))
        self.register_buffer("projection", projection)
        self.register_buffer("ranks", ranks)

    @classmethod
    def from_second_moment(
        cls, second_moment: Tensor, ranks: Tensor | Sequence[int]
    ) -> "HeadProjector":
        """Construct a projector from calibrated per-head second moments."""
        _, basis = principal_components(second_moment)
        return cls(basis, ranks)

    def compress(self, values: Tensor, *, head_axis: int = 0) -> Tensor:
        """Project values into zero-padded reduced coordinates."""
        values, original_head_axis = self._move_head_axis(values, head_axis)
        projection = self.projection.to(values)
        compressed = torch.einsum("h...d,hdr->h...r", values, projection)
        return compressed.movedim(0, original_head_axis)

    def reconstruct(self, compressed: Tensor, *, head_axis: int = 0) -> Tensor:
        """Reconstruct values from zero-padded reduced coordinates."""
        compressed, original_head_axis = self._move_head_axis(compressed, head_axis)
        if compressed.shape[-1] != self.projection.shape[-1]:
            raise ValueError(
                f"expected reduced dimension {self.projection.shape[-1]}, "
                f"got {compressed.shape[-1]}"
            )
        values = torch.einsum("h...r,hdr->h...d", compressed, self.projection.to(compressed))
        return values.movedim(0, original_head_axis)

    def forward(self, values: Tensor, *, head_axis: int = 0) -> Tensor:
        """Apply projection followed by reconstruction to emulate compression."""
        return self.reconstruct(self.compress(values, head_axis=head_axis), head_axis=head_axis)

    def _move_head_axis(self, values: Tensor, head_axis: int) -> tuple[Tensor, int]:
        if values.ndim < 2:
            raise ValueError("values must have a head axis and a final feature axis")
        head_axis %= values.ndim
        if head_axis == values.ndim - 1:
            raise ValueError("head_axis cannot be the final feature axis")
        if values.shape[head_axis] != self.projection.shape[0]:
            raise ValueError(
                f"expected {self.projection.shape[0]} heads, got {values.shape[head_axis]}"
            )
        return values.movedim(head_axis, 0), head_axis
