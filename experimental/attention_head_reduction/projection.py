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

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = [
    "ActivationGradientCalibration",
    "GradientWeightedDiagnostics",
    "HeadProjector",
    "SecondMomentAccumulator",
    "gradient_weighted_components",
    "gradient_weighted_components_with_diagnostics",
    "principal_components",
]


@dataclass(frozen=True)
class GradientWeightedDiagnostics:
    """Per-matrix numerical diagnostics from a gradient-weighted eigensolve."""

    activation_scale: Tensor
    gradient_scale: Tensor
    activation_precision_epsilon: Tensor
    gradient_precision_epsilon: Tensor
    activation_min_eigenvalue: Tensor
    gradient_min_eigenvalue: Tensor
    transformed_min_eigenvalue: Tensor
    activation_clamped_eigenvalues: Tensor
    gradient_clamped_eigenvalues: Tensor
    transformed_clamped_eigenvalues: Tensor
    condition_number_before_floor: Tensor
    condition_number_after_floor: Tensor
    condition_floored_eigenvalues: Tensor
    zero_gradient_moment: Tensor
    cholesky_jitter: Tensor
    biorthogonality_error: Tensor
    full_rank_projection_error: Tensor


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
        self.accumulation_epsilon = 0.0

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
        values = values.reshape(self.num_heads, -1, self.head_dim)
        if normalizer is None:
            if sample_weights is not None:
                raise ValueError("normalizer is required with sample_weights")
            normalizer = values.shape[1]
        if normalizer <= 0:
            raise ValueError("normalizer must be positive")
        if sample_weights is not None:
            try:
                sample_weights = torch.broadcast_to(sample_weights, sample_shape)
            except RuntimeError as error:
                raise ValueError(
                    f"sample_weights cannot broadcast to sample shape {tuple(sample_shape)}"
                ) from error
        if values.device == self.sum.device:
            values = values.to(self.sum)
        else:
            compute_dtype = torch.float32 if values.device.type == "mps" else self.sum.dtype
            values = values.to(dtype=compute_dtype)
        self.accumulation_epsilon = max(self.accumulation_epsilon, torch.finfo(values.dtype).eps)
        if sample_weights is None:
            weighted_values = values
        else:
            sample_weights = sample_weights.reshape(1, -1, 1).to(values)
            weighted_values = values * sample_weights
        batch_sum = torch.bmm(values.transpose(1, 2), weighted_values)
        if batch_sum.device != self.sum.device:
            # Transfer before widening: MPS cannot cast a resident tensor to float64.
            batch_sum = batch_sum.to(device=self.sum.device)
        self.sum.add_(batch_sum.to(dtype=self.sum.dtype))
        self.num_samples += normalizer

    def compute(self) -> Tensor:
        """Return the calibrated uncentered second-moment matrices."""
        if self.num_samples == 0:
            raise RuntimeError("cannot compute second moments before observing samples")
        assert self.sum is not None
        return self.sum / self.num_samples

    def _empty_sum(self, device: torch.device) -> Tensor:
        if device.type == "mps" and self.dtype in (None, torch.float64):
            device = torch.device("cpu")
        dtype = self.dtype or torch.float64
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
    max_condition_number: float = 1e8,
    activation_precision_epsilon: float | None = None,
    gradient_precision_epsilon: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return components minimizing a K-FAC gradient-weighted reconstruction objective.

    The returned compression and reconstruction bases implement the generally oblique
    gradient-metric projection. This solver applies both to squared first-order loss gaps and
    to the local quadratic approximation of output KL divergence. Damping is relative to each
    head's mean gradient eigenvalue.
    """
    eigenvalues, compression, reconstruction, _ = gradient_weighted_components_with_diagnostics(
        activation_second_moment,
        gradient_second_moment,
        damping=damping,
        max_condition_number=max_condition_number,
        activation_precision_epsilon=activation_precision_epsilon,
        gradient_precision_epsilon=gradient_precision_epsilon,
    )
    return eigenvalues, compression, reconstruction


def gradient_weighted_components_with_diagnostics(
    activation_second_moment: Tensor,
    gradient_second_moment: Tensor,
    *,
    damping: float = 1e-6,
    max_condition_number: float = 1e8,
    psd_tolerance_multiplier: float = 10.0,
    activation_precision_epsilon: float | None = None,
    gradient_precision_epsilon: float | None = None,
) -> tuple[Tensor, Tensor, Tensor, GradientWeightedDiagnostics]:
    """Return gradient-weighted components with numerical diagnostics.

    Moment matrices are validated and normalized before a float64 CPU solve. The regularized
    gradient metric uses a bounded condition number, and its inverse factor is obtained with a
    triangular solve rather than an explicit inverse square root.
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
    if not math.isfinite(max_condition_number) or max_condition_number <= 1:
        raise ValueError("max_condition_number must be finite and greater than one")
    if not math.isfinite(psd_tolerance_multiplier) or psd_tolerance_multiplier < 0:
        raise ValueError("psd_tolerance_multiplier must be finite and non-negative")
    if (
        not activation_second_moment.is_floating_point()
        or not gradient_second_moment.is_floating_point()
    ):
        raise ValueError("second moments must use a floating-point dtype")

    if activation_precision_epsilon is None:
        activation_precision_epsilon = torch.finfo(activation_second_moment.dtype).eps
    if gradient_precision_epsilon is None:
        gradient_precision_epsilon = torch.finfo(gradient_second_moment.dtype).eps
    for name, epsilon in (
        ("activation_precision_epsilon", activation_precision_epsilon),
        ("gradient_precision_epsilon", gradient_precision_epsilon),
    ):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"{name} must be finite and positive")

    output_device = activation_second_moment.device
    output_dtype = activation_second_moment.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        output_dtype = torch.float32
    activation_second_moment = activation_second_moment.to(device="cpu", dtype=torch.float64)
    gradient_second_moment = gradient_second_moment.to(device="cpu", dtype=torch.float64)
    (
        activation_eigenvalues,
        activation_eigenvectors,
        activation_min_eigenvalue,
        activation_clamped_eigenvalues,
    ) = _validated_psd_eigendecomposition(
        activation_second_moment,
        name="activation_second_moment",
        tolerance_multiplier=psd_tolerance_multiplier,
        precision_epsilon=activation_precision_epsilon,
    )
    (
        gradient_eigenvalues,
        gradient_eigenvectors,
        gradient_min_eigenvalue,
        gradient_clamped_eigenvalues,
    ) = _validated_psd_eigendecomposition(
        gradient_second_moment,
        name="gradient_second_moment",
        tolerance_multiplier=psd_tolerance_multiplier,
        precision_epsilon=gradient_precision_epsilon,
    )

    activation_scale = activation_eigenvalues.mean(dim=-1)
    gradient_scale = gradient_eigenvalues.mean(dim=-1)
    zero_gradient_moment = gradient_scale == 0
    safe_activation_scale = torch.where(
        activation_scale > 0, activation_scale, torch.ones_like(activation_scale)
    )
    safe_gradient_scale = torch.where(
        gradient_scale > 0, gradient_scale, torch.ones_like(gradient_scale)
    )
    normalized_activation = _symmetric_matrix_from_eigendecomposition(
        activation_eigenvectors,
        activation_eigenvalues / safe_activation_scale.unsqueeze(-1),
    )
    normalized_gradient_eigenvalues = gradient_eigenvalues / safe_gradient_scale.unsqueeze(-1)
    regularized_gradient_eigenvalues = normalized_gradient_eigenvalues + damping
    gradient_max = regularized_gradient_eigenvalues.amax(dim=-1)
    no_metric = gradient_max == 0
    fallback_eigenvalue = 1 / max_condition_number
    regularized_gradient_eigenvalues = torch.where(
        no_metric.unsqueeze(-1),
        torch.full_like(regularized_gradient_eigenvalues, fallback_eigenvalue),
        regularized_gradient_eigenvalues,
    )
    gradient_max = regularized_gradient_eigenvalues.amax(dim=-1)
    gradient_min = regularized_gradient_eigenvalues.amin(dim=-1)
    condition_number_before_floor = torch.where(
        gradient_min > 0,
        gradient_max / gradient_min,
        torch.full_like(gradient_max, torch.inf),
    )
    numerical_condition_limit = min(
        max_condition_number, 1 / torch.finfo(regularized_gradient_eigenvalues.dtype).eps
    )
    eigenvalue_floor = gradient_max / numerical_condition_limit
    condition_floored_eigenvalues = (
        regularized_gradient_eigenvalues < eigenvalue_floor.unsqueeze(-1)
    ).sum(dim=-1)
    regularized_gradient_eigenvalues = torch.maximum(
        regularized_gradient_eigenvalues, eigenvalue_floor.unsqueeze(-1)
    )
    condition_number_after_floor = gradient_max / regularized_gradient_eigenvalues.amin(dim=-1)
    regularized_gradient = _symmetric_matrix_from_eigendecomposition(
        gradient_eigenvectors, regularized_gradient_eigenvalues
    )
    gradient_factor, cholesky_jitter = _checked_cholesky(
        regularized_gradient,
        gradient_max,
    )
    final_gradient_eigenvalues = regularized_gradient_eigenvalues + cholesky_jitter.unsqueeze(-1)
    condition_number_after_floor = final_gradient_eigenvalues.amax(
        dim=-1
    ) / final_gradient_eigenvalues.amin(dim=-1)

    transformed_second_moment = (
        gradient_factor.transpose(-1, -2) @ normalized_activation @ gradient_factor
    )
    (
        transformed_eigenvalues,
        transformed_basis,
        transformed_min_eigenvalue,
        transformed_clamped_eigenvalues,
    ) = _validated_psd_eigendecomposition(
        transformed_second_moment,
        name="transformed_second_moment",
        tolerance_multiplier=psd_tolerance_multiplier,
        precision_epsilon=torch.finfo(transformed_second_moment.dtype).eps,
    )
    transformed_eigenvalues = transformed_eigenvalues.flip(-1)
    transformed_basis = transformed_basis.flip(-1)
    compression_basis = gradient_factor @ transformed_basis
    reconstruction_basis = torch.linalg.solve_triangular(
        gradient_factor.transpose(-1, -2),
        transformed_basis,
        upper=True,
    )
    _check_finite(compression_basis, "compression_basis")
    _check_finite(reconstruction_basis, "reconstruction_basis")

    dimension = compression_basis.shape[-1]
    identity = torch.eye(dimension, dtype=compression_basis.dtype).expand(
        *compression_basis.shape[:-2], -1, -1
    )
    biorthogonality_error = torch.linalg.matrix_norm(
        compression_basis.transpose(-1, -2) @ reconstruction_basis - identity,
        ord="fro",
    ) / math.sqrt(dimension)
    full_rank_projection = reconstruction_basis @ compression_basis.transpose(-1, -2)
    full_rank_projection_error = torch.linalg.matrix_norm(
        full_rank_projection - identity, ord="fro"
    ) / math.sqrt(dimension)
    validation_tolerance = (
        1000
        * dimension
        * torch.finfo(compression_basis.dtype).eps
        * math.sqrt(numerical_condition_limit)
    )
    if (
        torch.maximum(biorthogonality_error, full_rank_projection_error).amax()
        > validation_tolerance
    ):
        raise RuntimeError("gradient-weighted bases failed numerical projection checks")

    objective_scale = safe_activation_scale * safe_gradient_scale
    transformed_eigenvalues = transformed_eigenvalues * objective_scale.unsqueeze(-1)
    diagnostics = GradientWeightedDiagnostics(
        activation_scale=activation_scale,
        gradient_scale=gradient_scale,
        activation_precision_epsilon=torch.tensor(
            activation_precision_epsilon, dtype=torch.float64
        ),
        gradient_precision_epsilon=torch.tensor(gradient_precision_epsilon, dtype=torch.float64),
        activation_min_eigenvalue=activation_min_eigenvalue,
        gradient_min_eigenvalue=gradient_min_eigenvalue,
        transformed_min_eigenvalue=transformed_min_eigenvalue,
        activation_clamped_eigenvalues=activation_clamped_eigenvalues,
        gradient_clamped_eigenvalues=gradient_clamped_eigenvalues,
        transformed_clamped_eigenvalues=transformed_clamped_eigenvalues,
        condition_number_before_floor=condition_number_before_floor,
        condition_number_after_floor=condition_number_after_floor,
        condition_floored_eigenvalues=condition_floored_eigenvalues,
        zero_gradient_moment=zero_gradient_moment,
        cholesky_jitter=cholesky_jitter,
        biorthogonality_error=biorthogonality_error,
        full_rank_projection_error=full_rank_projection_error,
    )
    return (
        transformed_eigenvalues.to(device=output_device, dtype=output_dtype),
        compression_basis.to(device=output_device, dtype=output_dtype),
        reconstruction_basis.to(device=output_device, dtype=output_dtype),
        diagnostics,
    )


def _validated_psd_eigendecomposition(
    matrix: Tensor,
    *,
    name: str,
    tolerance_multiplier: float,
    precision_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _check_finite(matrix, name)
    matrix = (matrix + matrix.transpose(-1, -2)) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    _check_finite(eigenvalues, f"{name} eigenvalues")
    dimension = matrix.shape[-1]
    spectral_scale = eigenvalues.abs().amax(dim=-1)
    tolerance = (
        tolerance_multiplier
        * dimension
        * precision_epsilon
        * spectral_scale.clamp_min(torch.finfo(matrix.dtype).tiny)
    )
    minimum = eigenvalues[..., 0]
    materially_indefinite = minimum < -tolerance
    if materially_indefinite.any():
        worst = minimum.amin().item()
        worst_tolerance = tolerance.reshape(-1)[minimum.argmin()].item()
        raise ValueError(
            f"{name} is not positive semidefinite: minimum eigenvalue {worst:.6g} "
            f"is below tolerance {-worst_tolerance:.6g}"
        )
    clamped = (eigenvalues < 0).sum(dim=-1)
    return eigenvalues.clamp_min(0), eigenvectors, minimum, clamped


def _checked_cholesky(matrix: Tensor, spectral_scale: Tensor) -> tuple[Tensor, Tensor]:
    dimension = matrix.shape[-1]
    identity = torch.eye(dimension, dtype=matrix.dtype).expand(*matrix.shape[:-2], -1, -1)
    jitter = torch.zeros_like(spectral_scale)
    candidate = matrix
    for attempt in range(5):
        factor, info = torch.linalg.cholesky_ex(candidate)
        failed = info != 0
        if not failed.any():
            return factor, jitter
        next_jitter = (
            100
            * dimension
            * torch.finfo(matrix.dtype).eps
            * spectral_scale.clamp_min(1)
            * 10**attempt
        )
        jitter = torch.where(failed, next_jitter, jitter)
        candidate = matrix + jitter.unsqueeze(-1).unsqueeze(-1) * identity
    raise RuntimeError("regularized gradient metric remained non-positive-definite after jitter")


def _check_finite(values: Tensor, name: str) -> None:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


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
        max_condition_number: float = 1e8,
    ) -> "HeadProjector":
        """Construct a K-FAC projector for the squared first-order loss-gap objective."""
        return cls.from_gradient_moments(
            activation_second_moment,
            gradient_second_moment,
            ranks,
            damping=damping,
            max_condition_number=max_condition_number,
        )

    @classmethod
    def from_kl_divergence(
        cls,
        activation_second_moment: Tensor,
        fisher_second_moment: Tensor,
        ranks: Tensor | Sequence[int],
        *,
        damping: float = 1e-6,
        max_condition_number: float = 1e8,
    ) -> "HeadProjector":
        """Construct a K-FAC projector for the local quadratic output-KL objective."""
        return cls.from_gradient_moments(
            activation_second_moment,
            fisher_second_moment,
            ranks,
            damping=damping,
            max_condition_number=max_condition_number,
        )

    @classmethod
    def from_gradient_moments(
        cls,
        activation_second_moment: Tensor,
        gradient_second_moment: Tensor,
        ranks: Tensor | Sequence[int],
        *,
        damping: float = 1e-6,
        max_condition_number: float = 1e8,
    ) -> "HeadProjector":
        """Construct a K-FAC projector from activation and gradient second moments."""
        _, compression, reconstruction = gradient_weighted_components(
            activation_second_moment,
            gradient_second_moment,
            damping=damping,
            max_condition_number=max_condition_number,
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
