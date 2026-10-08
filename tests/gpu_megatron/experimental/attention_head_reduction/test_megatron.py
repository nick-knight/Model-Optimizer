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

"""Megatron-Core integration tests for attention head reduction."""

import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import get_forward, initialize_for_megatron
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.parallel_state import destroy_model_parallel
from torch import nn

from experimental.attention_head_reduction import HeadProjector, set_attention_key_value_transforms


def _test_te_attention_projection(rank, size):
    initialize_for_megatron(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
    )
    try:
        model = (
            get_mcore_gpt_model(
                tensor_model_parallel_size=size,
                num_layers=1,
                hidden_size=64,
                num_attention_heads=4,
                num_query_groups=2,
                vocab_size=32,
                transformer_impl="modelopt",
            )
            .cuda()
            .eval()
        )
        forward = get_forward(model)
        baseline = forward(model)
        attentions = [
            module for module in model.modules() if isinstance(module, TEDotProductAttention)
        ]
        assert len(attentions) == 1

        set_attention_key_value_transforms(
            attentions[0],
            nn.Identity(),
            nn.Identity(),
        )
        torch.testing.assert_close(forward(model), baseline)

        head_dim = 16
        local_kv_heads = 2 // size
        basis = torch.eye(head_dim, device="cuda").expand(local_kv_heads, -1, -1)
        reduced_rank = [head_dim // 2] * local_kv_heads
        set_attention_key_value_transforms(
            attentions[0],
            HeadProjector(basis, reduced_rank),
            HeadProjector(basis, reduced_rank),
        )
        assert not torch.equal(forward(model), baseline)
    finally:
        destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_te_attention_projection():
    """A real TE attention layer should accept replaceable, shape-preserving K/V projectors."""
    spawn_multiprocess_job(1, _test_te_attention_projection, backend="nccl")
