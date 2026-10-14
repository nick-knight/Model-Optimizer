# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker that installs frozen attention-head projection emulation."""

import os

from vllm.v1.worker.gpu_worker import Worker as BaseWorker

from experimental.attention_head_reduction.plugins.vllm import (
    install_vllm_head_reduction,
)

_PROJECTION_PATH = "MODELOPT_HEAD_REDUCTION_PROJECTION_PATH"
_RANK_ALLOCATION_PATH = "MODELOPT_HEAD_REDUCTION_RANK_ALLOCATION_PATH"
_DEFAULT_RANK = "MODELOPT_HEAD_REDUCTION_RANK"


class HeadReductionWorker(BaseWorker):
    """Install projection emulation after model loading and before profiling."""

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        projection_path = os.environ.get(_PROJECTION_PATH)
        if not projection_path:
            raise RuntimeError(f"{_PROJECTION_PATH} is required")
        allocation_path = os.environ.get(_RANK_ALLOCATION_PATH)
        rank_text = os.environ.get(_DEFAULT_RANK)
        report = install_vllm_head_reduction(
            self.model_runner,
            projection_path,
            rank_allocation_path=allocation_path,
            default_rank=None if rank_text is None else int(rank_text),
        )
        print(
            f"[ModelOpt] Attention head reduction installed on {report.installed_count} layers; "
            f"KV retention={report.kv_cache_retention:.6f}; "
            f"TP rank={report.tensor_parallel_rank}/{report.tensor_parallel_size}; "
            f"projection SHA-256={report.projection_sha256}"
        )
