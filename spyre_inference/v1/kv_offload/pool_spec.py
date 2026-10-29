# Copyright 2026 The Spyre-Inference Authors.
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

"""Pure model geometry used by the shared-pool delegator."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

from spyre_inference.v1.attention.ops.layout import kv_stick_geometry

_LAYOUTS = frozenset(("token-major", "head-major"))


def planned_page_bytes(
    dtype: torch.dtype,
    block_size: int,
    local_kv_heads: int,
    head_size: int,
    *,
    layout_kind: str = "token-major",
    layout_version: int = 1,
) -> int:
    """Physical bytes occupied by one K or V page after stick padding."""
    if layout_kind not in _LAYOUTS:
        raise ValueError(f"unsupported layout_kind {layout_kind!r}")
    if layout_version != 1:
        raise ValueError(f"unsupported layout_version {layout_version}")
    if block_size <= 0 or local_kv_heads <= 0:
        raise ValueError(
            f"block_size and local_kv_heads must be positive, got {block_size} and {local_kv_heads}"
        )
    sticks, elements_per_stick = kv_stick_geometry(dtype, head_size)
    element_size = torch.empty((), dtype=dtype).element_size()
    return block_size * local_kv_heads * sticks * elements_per_stick * element_size


def _parallel_size(parallel_config, name: str, fallback: str | None = None) -> int:
    value = getattr(parallel_config, name, None)
    if value is None and fallback is not None:
        value = getattr(parallel_config, fallback, 1)
    return int(1 if value is None else value)


def _specs_by_layer(kv_cache_config) -> dict[str, object]:
    specs: dict[str, object] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        per_layer = (
            group_spec.kv_cache_specs if isinstance(group_spec, UniformTypeKVCacheSpecs) else {}
        )
        for layer_name in group.layer_names:
            specs[layer_name] = per_layer.get(layer_name, group_spec)
    return specs


def _allocated_layer_order(kv_cache_config, group_layers: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in tensor.layers:
            if layer_name not in seen:
                ordered.append(layer_name)
                seen.add(layer_name)
    for layer_name in group_layers:
        if layer_name not in seen:
            ordered.append(layer_name)
            seen.add(layer_name)
    return ordered


def _layout_for_layer(layer_name: str, static_context) -> tuple[str, int, str | None]:
    layer = static_context.get(layer_name)
    impl = getattr(layer, "impl", None)
    if impl is None:
        raise ValueError(f"layer {layer_name!r} has no static KV layout information")

    kind = getattr(impl, "kv_layout_kind", None)
    version = getattr(impl, "kv_layout_version", 1)
    if kind is None:
        impl_name = type(impl).__name__
        if impl_name == "SpyreHeadMajorAttentionImpl":
            kind = "head-major"
        elif impl_name == "SpyreAttentionImpl":
            kind = "token-major"
        else:
            raise ValueError(
                f"layer {layer_name!r} has unsupported attention implementation "
                f"{impl_name!r}; cannot determine KV layout"
            )
    if kind not in _LAYOUTS or version != 1:
        raise ValueError(f"layer {layer_name!r} has unsupported KV layout {kind!r} v{version}")
    alias = getattr(layer, "kv_sharing_target_layer_name", None)
    if alias is None:
        alias = getattr(impl, "kv_sharing_target_layer_name", None)
    return kind, int(version), alias


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.uint8:
        return "float8_e4m3fn"
    return str(dtype).removeprefix("torch.")


def build_model_pool_spec(
    vllm_config,
    kv_cache_config,
    pool_name: str,
    cpu_bytes_to_use: int,
) -> dict[str, object]:
    """Build delegator protobuf JSON without allocating any device tensors."""
    if not pool_name:
        raise ValueError("pool_name must not be empty")
    if cpu_bytes_to_use <= 0:
        raise ValueError("cpu_bytes_to_use must be positive")
    if len(kv_cache_config.kv_cache_groups) != 1:
        raise ValueError("shared host offload requires exactly one KV cache group")

    parallel = vllm_config.parallel_config
    tp_size = _parallel_size(parallel, "tensor_parallel_size", "tp_size")
    if tp_size <= 0:
        raise ValueError("tensor parallel size must be positive")
    if _parallel_size(parallel, "pipeline_parallel_size", "pp_size") != 1:
        raise ValueError("pipeline parallel plans are not supported")
    if _parallel_size(parallel, "data_parallel_size", "dp_size") != 1:
        raise ValueError("data parallel plans are not supported")
    if _parallel_size(parallel, "decode_context_parallel_size") != 1:
        raise ValueError("context parallel plans are not supported")

    group = kv_cache_config.kv_cache_groups[0]
    specs = _specs_by_layer(kv_cache_config)
    static_context = vllm_config.compilation_config.static_forward_context
    cache_rows: list[dict[str, object]] = []
    dtype: torch.dtype | None = None
    block_size: int | None = None
    cache_index = 0
    for layer_name in _allocated_layer_order(kv_cache_config, group.layer_names):
        spec = specs.get(layer_name)
        if not isinstance(spec, FullAttentionSpec):
            raise ValueError(
                f"layer {layer_name!r} has unsupported attention spec {type(spec).__name__}"
            )
        layout_kind, layout_version, alias = _layout_for_layer(layer_name, static_context)
        if alias is not None:
            continue
        if dtype is None:
            dtype = spec.dtype
            block_size = int(spec.block_size)
        elif spec.dtype != dtype or int(spec.block_size) != block_size:
            raise ValueError(
                "all physical caches in one pool must use the same KV dtype and block size"
            )
        page_bytes = planned_page_bytes(
            spec.dtype,
            int(spec.block_size),
            int(spec.num_kv_heads),
            int(spec.head_size),
            layout_kind=layout_kind,
            layout_version=layout_version,
        )
        cache_rows.append(
            {
                "cacheIndex": cache_index,
                "physicalPageBytes": str(page_bytes),
                "layoutKind": layout_kind,
                "layoutVersion": layout_version,
                "localKvHeads": int(spec.num_kv_heads),
                "headSize": int(spec.head_size),
            }
        )
        cache_index += 1

    if not cache_rows or dtype is None or block_size is None:
        raise ValueError("no physical attention cache was found")

    rank_rows = [{"tpRank": tp_rank, **row} for tp_rank in range(tp_size) for row in cache_rows]
    model_config = vllm_config.model_config
    return {
        "planningVersion": 1,
        "name": pool_name,
        "model": str(model_config.model),
        "modelRevision": str(getattr(model_config, "revision", None) or ""),
        "kvDtype": _dtype_name(dtype),
        "blockSize": str(block_size),
        "tensorParallelSize": tp_size,
        "cpuBytesToUse": str(cpu_bytes_to_use),
        "rankCaches": rank_rows,
    }
