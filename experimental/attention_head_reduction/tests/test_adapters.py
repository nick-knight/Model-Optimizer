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

"""Tests for adapters around flattened multi-head projections."""

import torch
from torch import nn

from experimental.attention_head_reduction import HeadProjector, set_linear_head_transform


def test_linear_output_transform_applies_independently_per_head():
    """Flattened Hugging Face K/V projections should preserve their public shape."""
    linear = nn.Linear(2, 6, bias=False)
    linear.weight.data.copy_(torch.eye(6, 2))
    projector = HeadProjector(torch.eye(3).expand(2, -1, -1), ranks=[1, 2])
    set_linear_head_transform(linear, projector, num_heads=2, head_dim=3)

    output = linear(torch.tensor([[[1.0, 2.0]]]))

    torch.testing.assert_close(output, torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]))
    assert linear.state_dict().keys() == {
        "weight",
        "_modelopt_output_head_transform.transform.projection",
        "_modelopt_output_head_transform.transform.ranks",
    }


def test_setting_transform_twice_does_not_stack_forward_hooks():
    """Replacing calibration with projection should retain a single output hook."""
    linear = nn.Linear(4, 4, bias=False)
    linear.weight.data.copy_(torch.eye(4))
    set_linear_head_transform(linear, nn.Identity(), num_heads=2, head_dim=2)
    projector = HeadProjector(torch.eye(2).expand(2, -1, -1), ranks=[1, 1])

    set_linear_head_transform(linear, projector, num_heads=2, head_dim=2)

    assert len(linear._forward_hooks) == 1
    torch.testing.assert_close(linear(torch.ones(1, 4)), torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
