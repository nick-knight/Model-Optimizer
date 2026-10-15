# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install frozen attention-head projectors into a loaded vLLM model."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from ..adapters import set_flattened_attention_key_value_transforms
from ..projection import HeadProjector

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["VllmHeadReductionInstallReport", "install_vllm_head_reduction"]

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True, slots=True)
class VllmHeadReductionInstallReport:
    """Summary of projection emulation installed into vLLM."""

    installed_layers: tuple[str, ...]
    projection_sha256: str
    kv_cache_retention: float
    tensor_parallel_size: int
    tensor_parallel_rank: int

    @property
    def installed_count(self) -> int:
        """Return the number of attention layers carrying projectors."""
        return len(self.installed_layers)


@dataclass(frozen=True, slots=True)
class _InstallPlan:
    name: str
    attention: nn.Module
    key_projector: nn.Module
    value_projector: nn.Module


def _attention_type() -> type:
    for module_name in (
        "vllm.attention.layer",
        "vllm.model_executor.layers.attention",
        "vllm.attention",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "Attention"):
            return module.Attention
    raise ImportError("No supported vLLM Attention module was found")


def _unwrapped_model(model_runner) -> nn.Module:
    model = model_runner.model
    return model.unwrap() if hasattr(model, "unwrap") else model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_projection_artifact(path: Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
        raise ValueError("unsupported projection artifact schema")
    layers = artifact.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("projection artifact contains no attention layers")
    if not isinstance(artifact.get("retained_head_dim"), int):
        raise ValueError("projection artifact has no retained_head_dim")
    for site, layer in layers.items():
        if not isinstance(site, str) or not isinstance(layer, dict):
            raise ValueError("invalid projection layer entry")
        for kind in ("key", "value"):
            basis = layer.get(f"{kind}_basis")
            if not isinstance(basis, Tensor) or basis.ndim != 3:
                raise ValueError(f"{site} has no valid {kind} basis")
            if basis.shape[-2] != basis.shape[-1] or not torch.isfinite(basis).all():
                raise ValueError(f"{site} has an invalid {kind} basis")
            reconstruction = layer.get(f"{kind}_reconstruction_basis")
            if reconstruction is not None and (
                not isinstance(reconstruction, Tensor)
                or reconstruction.shape != basis.shape
                or not torch.isfinite(reconstruction).all()
            ):
                raise ValueError(f"{site} has an invalid {kind} reconstruction basis")
    return artifact


def _load_rank_allocation(path: Path | None) -> dict[tuple[str, str, int], int]:
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = document.get("allocation") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("rank allocation must contain a non-empty allocation list")
    allocation = {}
    for entry in entries:
        try:
            target = (str(entry["site"]), str(entry["kind"]), int(entry["head"]))
            rank = int(entry["rank"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid rank allocation entry: {entry!r}") from error
        if target[1] not in ("key", "value") or rank < 1 or target in allocation:
            raise ValueError(f"invalid or duplicate rank allocation target: {target!r}")
        allocation[target] = rank
    return allocation


def _layer_index(name: str) -> int:
    matches = _LAYER_INDEX.findall(name)
    if len(matches) != 1:
        raise ValueError(
            f"attention site must contain exactly one layer index: {name!r}"
        )
    return int(matches[0])


def _optional_layer_index(name: str) -> int | None:
    matches = _LAYER_INDEX.findall(name)
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"attention site contains multiple layer indices: {name!r}")
    return int(matches[0])


def _kv_head_indices(
    num_global_heads: int,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    num_local_heads: int,
) -> list[int]:
    if num_global_heads < tensor_parallel_size:
        if tensor_parallel_size % num_global_heads or num_local_heads != 1:
            raise ValueError("unsupported replicated GQA partition")
        return [tensor_parallel_rank // (tensor_parallel_size // num_global_heads)]
    if num_global_heads % tensor_parallel_size:
        raise ValueError("global K/V heads must divide TP size or vice versa")
    expected = num_global_heads // tensor_parallel_size
    if num_local_heads != expected:
        raise ValueError(
            f"expected {expected} local K/V heads, found {num_local_heads}"
        )
    start = tensor_parallel_rank * expected
    return list(range(start, start + expected))


def _ranks_for_site(
    site: str,
    kind: str,
    num_heads: int,
    dimension: int,
    default_rank: int,
    allocation: Mapping[tuple[str, str, int], int],
) -> list[int]:
    ranks = [
        allocation.get((site, kind, head), default_rank) for head in range(num_heads)
    ]
    if any(rank < 1 or rank > dimension for rank in ranks):
        raise ValueError(f"rank outside [1, {dimension}] for {site} {kind}: {ranks}")
    return ranks


def _projector(
    layer: Mapping[str, Any],
    kind: str,
    local_heads: list[int],
    local_ranks: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    basis = layer[f"{kind}_basis"]
    dimension = basis.shape[-1]
    if all(rank == dimension for rank in local_ranks):
        return nn.Identity()
    reconstruction = layer.get(f"{kind}_reconstruction_basis")
    projector = HeadProjector(
        basis[local_heads].to(device=device, dtype=dtype),
        local_ranks,
        reconstruction_basis=(
            None
            if reconstruction is None
            else reconstruction[local_heads].to(device=device, dtype=dtype)
        ),
    )
    return projector.requires_grad_(False)


def _validate_runtime(model_runner) -> None:
    config = getattr(model_runner, "vllm_config", None)
    model_config = getattr(config, "model_config", None)
    cache_config = getattr(config, "cache_config", None)
    if model_config is None or cache_config is None:
        raise RuntimeError("vLLM model and cache configs are required")
    if getattr(cache_config, "enable_prefix_caching", False):
        raise NotImplementedError(
            "head-reduction emulation has not validated prefix caching"
        )
    if getattr(config, "speculative_config", None) is not None:
        raise NotImplementedError(
            "head-reduction emulation has not validated speculative decoding"
        )


def install_vllm_head_reduction(
    model_runner,
    projection_path: str | Path,
    *,
    rank_allocation_path: str | Path | None = None,
    default_rank: int | None = None,
    tensor_parallel_size: int | None = None,
    tensor_parallel_rank: int | None = None,
) -> VllmHeadReductionInstallReport:
    """Install same-shaped K/V projection emulation before vLLM cache writes."""
    _validate_runtime(model_runner)
    projection_path = Path(projection_path)
    artifact = _load_projection_artifact(projection_path)
    allocation = _load_rank_allocation(
        None if rank_allocation_path is None else Path(rank_allocation_path)
    )
    if default_rank is None:
        default_rank = artifact["retained_head_dim"]
    if default_rank < 1:
        raise ValueError("default rank must be positive")
    if tensor_parallel_size is None:
        tensor_parallel_size = get_tensor_model_parallel_world_size()
    if tensor_parallel_rank is None:
        tensor_parallel_rank = get_tensor_model_parallel_rank()
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if tensor_parallel_rank < 0 or tensor_parallel_rank >= tensor_parallel_size:
        raise ValueError("tensor_parallel_rank must be in [0, tensor_parallel_size)")

    model = _unwrapped_model(model_runner)
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("vLLM model contains no parameters")
    dtype = (
        parameter.dtype
        if parameter.dtype in (torch.float16, torch.bfloat16)
        else torch.float32
    )
    attention_type = _attention_type()
    by_layer: dict[int, list[tuple[str, nn.Module]]] = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, attention_type):
            layer_index = _optional_layer_index(name)
            if layer_index is not None:
                by_layer[layer_index].append((name, module))

    plans = []
    expected_targets = set()
    retained_dimensions = 0
    baseline_dimensions = 0
    for site, layer in artifact["layers"].items():
        layer_index = _layer_index(site)
        matches = by_layer.get(layer_index, [])
        if len(matches) != 1:
            raise ValueError(
                f"expected one vLLM attention module for {site}, found {[name for name, _ in matches]}"
            )
        name, attention = matches[0]
        num_local_heads = getattr(attention, "num_kv_heads", None)
        head_dim = getattr(attention, "head_size", None)
        value_head_dim = getattr(attention, "head_size_v", head_dim)
        if not isinstance(num_local_heads, int) or not isinstance(head_dim, int):
            raise ValueError(f"{name} does not expose vLLM K/V head metadata")

        projectors = []
        for kind, runtime_dimension in (("key", head_dim), ("value", value_head_dim)):
            basis = layer[f"{kind}_basis"]
            num_global_heads, dimension, _ = basis.shape
            if dimension != runtime_dimension:
                raise ValueError(
                    f"{site} {kind} dimension {dimension} does not match {name} dimension {runtime_dimension}"
                )
            global_ranks = _ranks_for_site(
                site,
                kind,
                num_global_heads,
                dimension,
                default_rank,
                allocation,
            )
            local_heads = _kv_head_indices(
                num_global_heads,
                tensor_parallel_size,
                tensor_parallel_rank,
                num_local_heads,
            )
            local_ranks = [global_ranks[head] for head in local_heads]
            projectors.append(
                _projector(
                    layer,
                    kind,
                    local_heads,
                    local_ranks,
                    device=parameter.device,
                    dtype=dtype,
                )
            )
            expected_targets.update(
                (site, kind, head) for head in range(num_global_heads)
            )
            retained_dimensions += sum(global_ranks)
            baseline_dimensions += num_global_heads * dimension
        plans.append(_InstallPlan(name, attention, *projectors))

    if allocation:
        unknown = set(allocation) - expected_targets
        missing = expected_targets - set(allocation)
        if unknown:
            raise ValueError(
                f"rank allocation contains unknown targets: {sorted(unknown)!r}"
            )
        if missing:
            raise ValueError(f"rank allocation omits targets: {sorted(missing)!r}")

    for plan in plans:
        set_flattened_attention_key_value_transforms(
            plan.attention,
            plan.key_projector,
            plan.value_projector,
            num_heads=plan.attention.num_kv_heads,
            head_dim=plan.attention.head_size,
            value_head_dim=getattr(
                plan.attention, "head_size_v", plan.attention.head_size
            ),
        )
    return VllmHeadReductionInstallReport(
        installed_layers=tuple(plan.name for plan in plans),
        projection_sha256=_sha256(projection_path),
        kv_cache_retention=retained_dimensions / baseline_dimensions,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
    )
