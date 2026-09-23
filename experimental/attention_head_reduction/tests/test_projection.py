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

import torch

from experimental.attention_head_reduction import (
    HeadProjector,
    SecondMomentAccumulator,
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
