# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional framework plugins for attention head reduction."""

from .vllm import VllmHeadReductionInstallReport, install_vllm_head_reduction

__all__ = ["VllmHeadReductionInstallReport", "install_vllm_head_reduction"]
