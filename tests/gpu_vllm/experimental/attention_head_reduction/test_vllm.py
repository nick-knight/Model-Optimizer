# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM attention-head reduction installation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.attention import Attention

from experimental.attention_head_reduction.plugins.vllm import (
    install_vllm_head_reduction,
)


class _Layer(nn.Module):
    def __init__(self, attention=None):
        super().__init__()
        if attention is not None:
            self.attn = attention


class _Model(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.layers = nn.ModuleList([_Layer() for _ in range(6)])
        self.layers[5] = _Layer(attention)


def _bare_attention():
    attention = object.__new__(Attention)
    nn.Module.__init__(attention)
    attention.num_kv_heads = 1
    attention.head_size = 3
    attention.head_size_v = 3
    return attention


def _runner(attention, *, prefix_caching=False):
    return SimpleNamespace(
        model=_Model(attention),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True),
            cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
            speculative_config=None,
        ),
    )


def _artifact(path):
    identity = torch.eye(3).expand(2, -1, -1).clone()
    key_basis = identity.clone()
    key_basis[1] = key_basis[1][:, [1, 0, 2]]
    torch.save(
        {
            "schema_version": 1,
            "retained_head_dim": 1,
            "layers": {
                "decoder.layers.5.self_attention.core_attention": {
                    "key_basis": key_basis,
                    "value_basis": identity,
                }
            },
        },
        path,
    )


def test_installer_selects_the_local_tp_head_and_hooks_before_cache_write(tmp_path):
    projection_path = tmp_path / "projections.pt"
    _artifact(projection_path)
    attention = _bare_attention()

    report = install_vllm_head_reduction(
        _runner(attention),
        projection_path,
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
    )

    transform = attention._modelopt_key_value_head_transform
    key, value = transform(
        torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[1.0, 2.0, 3.0]])
    )
    torch.testing.assert_close(key, torch.tensor([[0.0, 2.0, 0.0]]))
    torch.testing.assert_close(value, torch.tensor([[1.0, 0.0, 0.0]]))
    assert report.installed_layers == ("layers.5.attn",)
    assert report.kv_cache_retention == pytest.approx(1 / 3)
    assert len(attention._forward_pre_hooks) == 1


def test_installer_rejects_prefix_caching_before_mutating_the_model(tmp_path):
    projection_path = tmp_path / "projections.pt"
    _artifact(projection_path)
    attention = _bare_attention()

    with pytest.raises(NotImplementedError, match="prefix caching"):
        install_vllm_head_reduction(
            _runner(attention, prefix_caching=True),
            projection_path,
            tensor_parallel_size=2,
            tensor_parallel_rank=0,
        )

    assert not hasattr(attention, "_modelopt_key_value_head_transform")
