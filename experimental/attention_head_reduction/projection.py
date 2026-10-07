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

"""Per-head second-moment calibration and low-rank projection."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "ActivationGradientCalibration",
    "HeadProjector",
    "SecondMomentAccumulator",
    "gradient_weighted_components",
    "principal_components",
]


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
    def update(
        self,
        values: Tensor,
        *,
        head_axis: int = 0,
        sample_weights: Tensor | None = None,
        normalizer: int | float | None = None,
    ) -> None:
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
        sample_shape = values.shape[1:-1]
        values = values.reshape(self.num_heads, -1, self.head_dim).to(self.sum)
        if normalizer is None:
            if sample_weights is not None:
                raise ValueError("normalizer is required with sample_weights")
            normalizer = values.shape[1]
        if normalizer <= 0:
            raise ValueError("normalizer must be positive")
        if sample_weights is None:
            weighted_values = values
        else:
            try:
                sample_weights = torch.broadcast_to(sample_weights, sample_shape)
            except RuntimeError as error:
                raise ValueError(
                    f"sample_weights cannot broadcast to sample shape {tuple(sample_shape)}"
                ) from error
            sample_weights = sample_weights.reshape(1, -1, 1).to(values)
            weighted_values = values * sample_weights
        self.sum.add_(torch.bmm(values.transpose(1, 2), weighted_values))
        self.num_samples += normalizer

    def compute(self) -> Tensor:
        """Return the calibrated uncentered second-moment matrices."""
        if self.num_samples == 0:
            raise RuntimeError("cannot compute second moments before observing samples")
        assert self.sum is not None
        return self.sum / self.num_samples

    def _empty_sum(self, device: torch.device) -> Tensor:
        dtype = self.dtype or (torch.float32 if device.type == "mps" else torch.float64)
        return torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)


class ActivationGradientCalibration(nn.Module):
    """Accumulate per-head second moments of activations and their loss gradients."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        head_axis: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a same-shaped transform that records forward and backward values."""
        super().__init__()
        self.head_axis = head_axis
        self.activation_statistics = SecondMomentAccumulator(
            num_heads, head_dim, device=device, dtype=dtype
        )
        self.gradient_statistics = SecondMomentAccumulator(
            num_heads, head_dim, device=device, dtype=dtype
        )
        self._activation_weights = None
        self._activation_normalizer = None
        self._gradient_normalizer = None

    def set_activation_weights(
        self, sample_weights: Tensor | None, *, normalizer: int | float | None = None
    ) -> None:
        """Configure weighting for the next forward activation update."""
        self._activation_weights = sample_weights
        self._activation_normalizer = normalizer

    def set_gradient_normalizer(self, normalizer: int | float | None) -> None:
        """Configure normalization for subsequent backwards through the current graph."""
        self._gradient_normalizer = normalizer

    def forward(self, values: Tensor) -> Tensor:
        """Record activations and attach a hook that records their loss gradients."""
        if not values.requires_grad:
            raise RuntimeError("activation-gradient calibration requires gradients")
        self.activation_statistics.update(
            values,
            head_axis=self.head_axis,
            sample_weights=self._activation_weights,
            normalizer=self._activation_normalizer,
        )
        values.register_hook(self._record_gradient)
        return values

    def compute(self) -> tuple[Tensor, Tensor]:
        """Return activation and gradient second moments."""
        return self.activation_statistics.compute(), self.gradient_statistics.compute()

    def _record_gradient(self, gradient: Tensor) -> None:
        self.gradient_statistics.update(
            gradient,
            head_axis=self.head_axis,
            normalizer=self._gradient_normalizer,
        )


def principal_components(second_moment: Tensor) -> tuple[Tensor, Tensor]:
    """Return eigenvalues and column eigenvectors in descending eigenvalue order."""
    if second_moment.ndim < 2 or second_moment.shape[-1] != second_moment.shape[-2]:
        raise ValueError("second_moment must contain square matrices")
    second_moment = (second_moment + second_moment.transpose(-1, -2)) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    return eigenvalues.flip(-1), eigenvectors.flip(-1)


def gradient_weighted_components(
    activation_second_moment: Tensor,
    gradient_second_moment: Tensor,
    *,
    damping: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return components minimizing a K-FAC gradient-weighted reconstruction objective.

    The returned compression and reconstruction bases implement the generally oblique
    gradient-metric projection. This solver applies both to squared first-order loss gaps and
    to the local quadratic approximation of output KL divergence. Damping is relative to each
    head's mean gradient eigenvalue.
    """
    if activation_second_moment.shape != gradient_second_moment.shape:
        raise ValueError("activation and gradient second moments must have the same shape")
    if (
        activation_second_moment.ndim < 2
        or activation_second_moment.shape[-1] != activation_second_moment.shape[-2]
    ):
        raise ValueError("second moments must contain square matrices")
    if damping < 0:
        raise ValueError("damping must be non-negative")

    gradient_second_moment = (gradient_second_moment + gradient_second_moment.transpose(-1, -2)) / 2
    gradient_eigenvalues, gradient_eigenvectors = torch.linalg.eigh(gradient_second_moment)
    gradient_eigenvalues = gradient_eigenvalues.clamp_min(0)
    scale = gradient_eigenvalues.mean(dim=-1, keepdim=True)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    gradient_eigenvalues = gradient_eigenvalues + damping * scale
    gradient_eigenvalues = gradient_eigenvalues.clamp_min(
        torch.finfo(gradient_eigenvalues.dtype).eps * scale
    )

    gradient_root = _symmetric_matrix_from_eigendecomposition(
        gradient_eigenvectors, gradient_eigenvalues.sqrt()
    )
    gradient_inverse_root = _symmetric_matrix_from_eigendecomposition(
        gradient_eigenvectors, gradient_eigenvalues.rsqrt()
    )
    activation_second_moment = (
        activation_second_moment + activation_second_moment.transpose(-1, -2)
    ) / 2
    weighted_second_moment = gradient_root @ activation_second_moment @ gradient_root
    eigenvalues, weighted_basis = principal_components(weighted_second_moment)
    compression_basis = gradient_root @ weighted_basis
    reconstruction_basis = gradient_inverse_root @ weighted_basis
    return eigenvalues, compression_basis, reconstruction_basis


def _symmetric_matrix_from_eigendecomposition(eigenvectors: Tensor, eigenvalues: Tensor) -> Tensor:
    return (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


class HeadProjector(nn.Module):
    """Emulate independently ranked per-head compression using frozen bases."""

    def __init__(
        self,
        basis: Tensor,
        ranks: Tensor | Sequence[int],
        *,
        reconstruction_basis: Tensor | None = None,
    ) -> None:
        """Initialize from full per-head compression bases and retained ranks.

        When ``reconstruction_basis`` is omitted, the compression basis is reused to
        implement an orthogonal projection. Distinct bases implement an oblique projection.
        """
        super().__init__()
        if basis.ndim != 3 or basis.shape[-2] != basis.shape[-1]:
            raise ValueError("basis must have shape (num_heads, head_dim, head_dim)")
        if reconstruction_basis is not None and reconstruction_basis.shape != basis.shape:
            raise ValueError("reconstruction_basis must have the same shape as basis")

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
        if reconstruction_basis is not None:
            reconstruction = reconstruction_basis[..., :max_rank].clone()
            reconstruction.mul_(retained.unsqueeze(1).to(reconstruction.device))
            self.register_buffer("reconstruction", reconstruction)
        self.register_buffer("ranks", ranks)

    @classmethod
    def from_second_moment(
        cls, second_moment: Tensor, ranks: Tensor | Sequence[int]
    ) -> "HeadProjector":
        """Construct a projector from calibrated per-head second moments."""
        _, basis = principal_components(second_moment)
        return cls(basis, ranks)

    @classmethod
    def from_squared_loss_gap(
        cls,
        activation_second_moment: Tensor,
        gradient_second_moment: Tensor,
        ranks: Tensor | Sequence[int],
        *,
        damping: float = 1e-6,
    ) -> "HeadProjector":
        """Construct a K-FAC projector for the squared first-order loss-gap objective."""
        return cls.from_gradient_moments(
            activation_second_moment,
            gradient_second_moment,
            ranks,
            damping=damping,
        )

    @classmethod
    def from_kl_divergence(
        cls,
        activation_second_moment: Tensor,
        fisher_second_moment: Tensor,
        ranks: Tensor | Sequence[int],
        *,
        damping: float = 1e-6,
    ) -> "HeadProjector":
        """Construct a K-FAC projector for the local quadratic output-KL objective."""
        return cls.from_gradient_moments(
            activation_second_moment,
            fisher_second_moment,
            ranks,
            damping=damping,
        )

    @classmethod
    def from_gradient_moments(
        cls,
        activation_second_moment: Tensor,
        gradient_second_moment: Tensor,
        ranks: Tensor | Sequence[int],
        *,
        damping: float = 1e-6,
    ) -> "HeadProjector":
        """Construct a K-FAC projector from activation and gradient second moments."""
        _, compression, reconstruction = gradient_weighted_components(
            activation_second_moment,
            gradient_second_moment,
            damping=damping,
        )
        return cls(compression, ranks, reconstruction_basis=reconstruction)

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
        reconstruction = self._buffers.get("reconstruction")
        if reconstruction is None:
            reconstruction = self.projection
        values = torch.einsum("h...r,hdr->h...d", compressed, reconstruction.to(compressed))
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
