# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference checks for Megatron logits Jaccard distillation."""

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.distill.plugins.megatron import DistillationConfig, LogitsJaccardLoss


@pytest.fixture(scope="session", autouse=True)
def _prebuild_quant_cuda_extensions():
    """Skip unrelated quantization JIT builds for this CPU-only loss test."""


@pytest.mark.parametrize("chunk_size", [0, 2])
def test_jaccard_value_gradient_and_teacher_detach(chunk_size):
    torch.manual_seed(7)
    student = torch.randn(5, 2, 17, requires_grad=True)
    teacher = torch.randn(5, 2, 17, requires_grad=True)
    reference_student = student.detach().clone().requires_grad_(True)

    student_probability = reference_student.softmax(-1)
    teacher_probability = teacher.detach().softmax(-1)
    total_variation = 0.5 * (student_probability - teacher_probability).abs().sum(-1)
    reference = 2 * total_variation / (1 + total_variation + 1e-8)

    criterion = LogitsJaccardLoss(
        SimpleNamespace(tensor_model_parallel_size=1), chunk_size=chunk_size
    )
    loss, tensor_parallel_reduce, sequence_parallel = criterion(student, teacher)
    torch.testing.assert_close(loss, reference.T)
    loss.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(student.grad, reference_student.grad)
    assert teacher.grad is None
    assert not tensor_parallel_reduce
    assert not sequence_parallel


def test_jaccard_distillation_config_validation():
    assert DistillationConfig(logit_loss_type="jaccard").logit_loss_type == "jaccard"
    with pytest.raises(AssertionError, match="logit_loss_type"):
        DistillationConfig(logit_loss_type="unknown")
