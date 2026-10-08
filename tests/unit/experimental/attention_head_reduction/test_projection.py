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

"""Tests for per-head PCA calibration and projection."""

import pytest
import torch

from experimental.attention_head_reduction import (
    ActivationGradientCalibration,
    HeadProjector,
    SecondMomentAccumulator,
    gradient_weighted_components,
    gradient_weighted_components_with_diagnostics,
    principal_components,
)


def test_second_moment_accumulates_batches_and_accepts_nonleading_head_axis():
    """Calibration should compute exact sample-weighted, per-head second moments."""
    first = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    second = torch.tensor([[[[5.0, 6.0]], [[7.0, 8.0]]]])
    accumulator = SecondMomentAccumulator(2, 2)

    accumulator.update(first, head_axis=1)
    accumulator.update(second, head_axis=1)

    all_values = torch.cat((first, second), dim=0).movedim(1, 0).reshape(2, -1, 2).double()
    expected = torch.bmm(all_values.transpose(1, 2), all_values) / all_values.shape[1]
    torch.testing.assert_close(accumulator.compute(), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_second_moment_widens_on_cpu_after_accumulating_on_mps():
    """MPS statistics should transfer to CPU before widening to unsupported float64."""
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device="mps")
    accumulator = SecondMomentAccumulator(1, 2)

    accumulator.update(values)

    result = accumulator.compute()
    expected = torch.tensor([[[5.0, 7.0], [7.0, 10.0]]], dtype=torch.float64)
    assert result.device.type == "cpu"
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected)


def test_principal_components_are_sorted_descending():
    """The first basis vector should correspond to the largest eigenvalue."""
    moments = torch.diag_embed(torch.tensor([[2.0, 5.0, 1.0], [7.0, 3.0, 4.0]]))

    eigenvalues, basis = principal_components(moments)

    torch.testing.assert_close(eigenvalues, torch.tensor([[5.0, 2.0, 1.0], [7.0, 4.0, 3.0]]))
    reconstructed = basis @ torch.diag_embed(eigenvalues) @ basis.transpose(-1, -2)
    torch.testing.assert_close(reconstructed, moments)


def test_projector_supports_different_rank_per_head():
    """Each head should retain only its independently selected principal subspace."""
    basis = torch.eye(3).expand(2, -1, -1).clone()
    projector = HeadProjector(basis, ranks=[1, 2])
    values = torch.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]])

    compressed = projector.compress(values, head_axis=1)
    reconstructed = projector(values, head_axis=1)

    assert compressed.shape == (1, 2, 1, 2)
    torch.testing.assert_close(compressed, torch.tensor([[[[1.0, 0.0]], [[4.0, 5.0]]]]))
    torch.testing.assert_close(
        reconstructed, torch.tensor([[[[1.0, 0.0, 0.0]], [[4.0, 5.0, 0.0]]]])
    )


def test_projector_is_frozen_but_preserves_input_gradients():
    """Projection bases should be buffers while student activations remain differentiable."""
    values = torch.randn(2, 3, 4, requires_grad=True)
    projector = HeadProjector(torch.eye(4).expand(2, -1, -1).clone(), ranks=[2, 3])

    projector(values).square().sum().backward()

    assert list(projector.parameters()) == []
    assert values.grad is not None
    assert projector.state_dict().keys() == {"projection", "ranks"}


def test_activation_gradient_calibration_accumulates_local_second_moments():
    """Calibration should independently flatten positions in both forward and backward."""
    values = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]],
        requires_grad=True,
    )
    weights = torch.tensor([[[[2.0, 3.0], [4.0, 5.0]]], [[[6.0, 7.0], [8.0, 9.0]]]])
    calibration = ActivationGradientCalibration(2, 2)

    (calibration(values) * weights).sum().backward()

    flattened_values = values.detach().reshape(2, -1, 2).double()
    flattened_gradients = weights.reshape(2, -1, 2).double()
    expected_activations = (
        torch.bmm(flattened_values.transpose(1, 2), flattened_values) / flattened_values.shape[1]
    )
    expected_gradients = (
        torch.bmm(flattened_gradients.transpose(1, 2), flattened_gradients)
        / flattened_gradients.shape[1]
    )
    activation_moment, gradient_moment = calibration.compute()
    torch.testing.assert_close(activation_moment, expected_activations)
    torch.testing.assert_close(gradient_moment, expected_gradients)


def test_activation_gradient_calibration_weights_reused_forward_graph():
    """Prefix multiplicities and repeated target backwards should produce paired moments."""
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    first_gradient = torch.tensor([2.0, 3.0])
    second_gradients = torch.tensor([[5.0, 7.0], [11.0, 13.0]])
    calibration = ActivationGradientCalibration(1, 2)
    calibration.set_activation_weights(torch.tensor([[2.0, 1.0]]), normalizer=3)
    calibrated = calibration(values)

    calibration.set_gradient_normalizer(1)
    (calibrated[0, 0, 0] * first_gradient).sum().backward(retain_graph=True)
    calibration.set_gradient_normalizer(2)
    (calibrated[0, 0] * second_gradients).sum().backward()

    expected_activation = (
        2 * torch.outer(values[0, 0, 0], values[0, 0, 0])
        + torch.outer(values[0, 0, 1], values[0, 0, 1])
    ).double() / 3
    expected_gradient = (
        torch.outer(first_gradient, first_gradient)
        + torch.einsum("sd,se->de", second_gradients, second_gradients)
    ).double() / 3
    activation_moment, gradient_moment = calibration.compute()
    torch.testing.assert_close(activation_moment[0], expected_activation)
    torch.testing.assert_close(gradient_moment[0], expected_gradient)


def test_gradient_weighted_projector_is_optimal_oblique_projection():
    """The factorized projector should perform PCA in the gradient-whitened coordinates."""
    angle = torch.tensor(0.6, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((angle.cos(), -angle.sin())),
            torch.stack((angle.sin(), angle.cos())),
        )
    )
    gradient_moment = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
    gradient_inverse_root = torch.diag(torch.tensor([0.5, 1.0], dtype=torch.float64))
    weighted_moment = (
        rotation @ torch.diag(torch.tensor([5.0, 1.0], dtype=torch.float64)) @ rotation.T
    )
    activation_moment = gradient_inverse_root @ weighted_moment @ gradient_inverse_root

    eigenvalues, _, _ = gradient_weighted_components(
        activation_moment.unsqueeze(0),
        gradient_moment.unsqueeze(0),
        damping=0,
    )
    projector = HeadProjector.from_squared_loss_gap(
        activation_moment.unsqueeze(0),
        gradient_moment.unsqueeze(0),
        ranks=[1],
        damping=0,
    )
    column_projection = projector.reconstruction[0] @ projector.projection[0].T
    residual = torch.eye(2, dtype=torch.float64) - column_projection
    objective = torch.trace(gradient_moment @ residual @ activation_moment @ residual.T)
    values = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)

    torch.testing.assert_close(eigenvalues, torch.tensor([[5.0, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(column_projection @ column_projection, column_projection)
    assert not torch.allclose(column_projection, column_projection.T)
    torch.testing.assert_close(objective, eigenvalues[0, 1])
    torch.testing.assert_close(projector(values), values @ column_projection.T)
    assert projector.state_dict().keys() == {"projection", "reconstruction", "ranks"}


def test_kl_projector_uses_gradient_weighted_solver():
    """The KL constructor should interpret its gradient moment as a model Fisher."""
    activation_moment = torch.tensor([[[2.0, 0.5], [0.5, 1.0]]], dtype=torch.float64)
    fisher_moment = torch.tensor([[[1.0, 0.25], [0.25, 3.0]]], dtype=torch.float64)

    loss_gap = HeadProjector.from_squared_loss_gap(activation_moment, fisher_moment, ranks=[1])
    kl = HeadProjector.from_kl_divergence(activation_moment, fisher_moment, ranks=[1])

    torch.testing.assert_close(kl.projection, loss_gap.projection)
    torch.testing.assert_close(kl.reconstruction, loss_gap.reconstruction)


def test_gradient_weighted_solver_bounds_condition_number_and_reports_diagnostics():
    """Near-null gradient directions should be floored before the triangular solve."""
    activation_moment = torch.tensor([[[2.0, 0.25], [0.25, 1.0]]], dtype=torch.float64)
    gradient_moment = torch.diag_embed(torch.tensor([[1.0, 1e-14]], dtype=torch.float64))

    _, compression, reconstruction, diagnostics = gradient_weighted_components_with_diagnostics(
        activation_moment,
        gradient_moment,
        damping=0,
        max_condition_number=1e6,
    )

    assert diagnostics.condition_floored_eigenvalues.item() == 1
    assert diagnostics.condition_number_before_floor.item() > 1e12
    assert diagnostics.condition_number_after_floor.item() == pytest.approx(1e6)
    assert diagnostics.biorthogonality_error.item() < 1e-10
    assert diagnostics.full_rank_projection_error.item() < 1e-10
    assert torch.isfinite(compression).all()
    assert torch.isfinite(reconstruction).all()


def test_gradient_weighted_solver_clamps_only_roundoff_scale_negative_eigenvalues():
    """Small PSD violations may be rounded away, while material violations must fail."""
    gradient_moment = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    roundoff_activation = torch.diag_embed(torch.tensor([[1.0, -1e-15]], dtype=torch.float64))

    _, _, _, diagnostics = gradient_weighted_components_with_diagnostics(
        roundoff_activation, gradient_moment
    )

    assert diagnostics.activation_clamped_eigenvalues.item() == 1
    low_precision_roundoff = torch.diag_embed(torch.tensor([[1.0, -1e-6]], dtype=torch.float64))
    _, _, _, diagnostics = gradient_weighted_components_with_diagnostics(
        low_precision_roundoff,
        gradient_moment,
        activation_precision_epsilon=torch.finfo(torch.float32).eps,
    )
    assert diagnostics.activation_clamped_eigenvalues.item() == 1
    indefinite_activation = torch.diag_embed(torch.tensor([[1.0, -1e-3]], dtype=torch.float64))
    with pytest.raises(ValueError, match="not positive semidefinite"):
        gradient_weighted_components(indefinite_activation, gradient_moment)


def test_gradient_weighted_solver_rejects_nonfinite_moments_and_handles_zero_metric():
    """Invalid statistics should fail clearly and an empty metric should use a finite fallback."""
    activation_moment = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    nonfinite_gradient = activation_moment.clone()
    nonfinite_gradient[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        gradient_weighted_components(activation_moment, nonfinite_gradient)

    _, compression, reconstruction, diagnostics = gradient_weighted_components_with_diagnostics(
        activation_moment,
        torch.zeros_like(activation_moment),
        damping=0,
    )
    assert diagnostics.zero_gradient_moment.item()
    assert torch.isfinite(compression).all()
    assert torch.isfinite(reconstruction).all()
