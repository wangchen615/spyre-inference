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

"""Strict immutable view of a delegator-created model pool plan."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mmap
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity


@dataclass(frozen=True)
class PlannedPartition:
    partition_id: int
    arena_name: str
    partition_name: str
    tp_rank: int
    cache_index: int
    offset_bytes: int
    num_slots: int
    slot_bytes: int
    physical_page_bytes: int


@dataclass(frozen=True)
class PlannedComponent:
    global_component_id: int
    tp_rank: int
    cache_index: int
    role: str
    arena_name: str
    partition_name: str
    partition_id: int
    partition_offset_bytes: int
    partition_slot_bytes: int
    physical_page_bytes: int


@dataclass(frozen=True)
class PlannedRankCache:
    tp_rank: int
    cache_index: int
    physical_page_bytes: int
    layout_kind: str
    layout_version: int
    local_kv_heads: int
    head_size: int


@dataclass(frozen=True)
class PlannedArena:
    name: str
    backend: str
    model: str
    kv_dtype: str
    block_size: int
    tensor_parallel_size: int
    partitions: tuple[PlannedPartition, ...]
    num_slots: int = 0
    slot_bytes: int = 0
    layout_signature: str = ""


@dataclass(frozen=True)
class PlannedPool:
    identity: SharedPoolIdentity
    planning_version: int
    model: str
    model_revision: str
    kv_dtype: str
    block_size: int
    tensor_parallel_size: int
    cpu_bytes_to_use: int
    bytes_per_complete_block: int
    logical_block_capacity: int
    allocated_bytes: int
    unused_bytes: int
    arena: PlannedArena
    partitions: tuple[PlannedPartition, ...]
    components: tuple[PlannedComponent, ...]
    rank_caches: tuple[PlannedRankCache, ...] = ()
    metadata_max_chunks: int = 1

    def partitions_for_rank(self, tp_rank: int) -> tuple[PlannedPartition, ...]:
        if not 0 <= tp_rank < self.tensor_parallel_size:
            raise ValueError(
                f"tensor-parallel rank {tp_rank} is outside [0, {self.tensor_parallel_size})"
            )
        return tuple(item for item in self.partitions if item.tp_rank == tp_rank)

    def components_for_rank(self, tp_rank: int) -> tuple[PlannedComponent, ...]:
        self.partitions_for_rank(tp_rank)
        return tuple(item for item in self.components if item.tp_rank == tp_rank)

    def caches_for_rank(self, tp_rank: int) -> tuple[PlannedRankCache, ...]:
        self.partitions_for_rank(tp_rank)
        return tuple(item for item in self.rank_caches if item.tp_rank == tp_rank)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _uint(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an unsigned integer")
    if isinstance(value, str):
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f"{field} must be an unsigned integer")
        result = int(value)
    elif isinstance(value, int):
        result = value
    else:
        raise ValueError(f"{field} must be an unsigned integer")
    if result < 0 or (positive and result == 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{field} must be a {qualifier}unsigned integer")
    if result > (1 << 64) - 1:
        raise ValueError(f"{field} must fit uint64")
    return result


def _checked_add(left: int, right: int, field: str) -> int:
    result = left + right
    if result > (1 << 64) - 1:
        raise ValueError(f"{field} overflows uint64")
    return result


def _checked_multiply(left: int, right: int, field: str) -> int:
    result = left * right
    if result > (1 << 64) - 1:
        raise ValueError(f"{field} overflows uint64")
    return result


def _align_up(value: int, alignment: int, field: str) -> int:
    remainder = value % alignment
    return value if remainder == 0 else _checked_add(value, alignment - remainder, field)


def _generation(value: object) -> bytes:
    encoded = _text(value, "poolDescriptor.generation")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("poolDescriptor.generation must be valid base64") from error
    if len(decoded) != 16:
        raise ValueError("poolDescriptor.generation must encode exactly 16 bytes")
    return decoded


def _parse_partition(value: object, field: str) -> PlannedPartition:
    row = _mapping(value, field)
    return PlannedPartition(
        partition_id=_uint(row.get("partitionId"), f"{field}.partitionId"),
        arena_name=_text(row.get("arenaName"), f"{field}.arenaName"),
        partition_name=_text(row.get("partitionName"), f"{field}.partitionName"),
        tp_rank=_uint(row.get("tpRank", 0), f"{field}.tpRank"),
        cache_index=_uint(row.get("cacheIndex", 0), f"{field}.cacheIndex"),
        offset_bytes=_uint(row.get("offsetBytes", 0), f"{field}.offsetBytes"),
        num_slots=_uint(row.get("numSlots"), f"{field}.numSlots", positive=True),
        slot_bytes=_uint(row.get("slotBytes"), f"{field}.slotBytes", positive=True),
        physical_page_bytes=_uint(
            row.get("physicalPageBytes"),
            f"{field}.physicalPageBytes",
            positive=True,
        ),
    )


def _parse_component(value: object, field: str) -> PlannedComponent:
    row = _mapping(value, field)
    role_value = _text(row.get("role"), f"{field}.role")
    roles = {
        "COMPONENT_ROLE_K": "k",
        "COMPONENT_ROLE_V": "v",
        "K": "k",
        "V": "v",
    }
    if role_value not in roles:
        raise ValueError(f"{field}.role must identify K or V")
    return PlannedComponent(
        global_component_id=_uint(row.get("globalComponentId"), f"{field}.globalComponentId"),
        tp_rank=_uint(row.get("tpRank", 0), f"{field}.tpRank"),
        cache_index=_uint(row.get("cacheIndex", 0), f"{field}.cacheIndex"),
        role=roles[role_value],
        arena_name=_text(row.get("arenaName"), f"{field}.arenaName"),
        partition_name=_text(row.get("partitionName"), f"{field}.partitionName"),
        partition_id=_uint(row.get("partitionId"), f"{field}.partitionId"),
        partition_offset_bytes=_uint(
            row.get("partitionOffsetBytes", 0), f"{field}.partitionOffsetBytes"
        ),
        partition_slot_bytes=_uint(
            row.get("partitionSlotBytes"),
            f"{field}.partitionSlotBytes",
            positive=True,
        ),
        physical_page_bytes=_uint(
            row.get("physicalPageBytes"),
            f"{field}.physicalPageBytes",
            positive=True,
        ),
    )


def _configuration_partition(row: object, field: str) -> tuple[object, ...]:
    value = _mapping(row, field)
    return (
        _text(value.get("name"), f"{field}.name"),
        _uint(value.get("tpRank", 0), f"{field}.tpRank"),
        _uint(value.get("cacheIndex", 0), f"{field}.cacheIndex"),
        _uint(value.get("offsetBytes", 0), f"{field}.offsetBytes"),
        _uint(value.get("numSlots"), f"{field}.numSlots", positive=True),
        _uint(value.get("slotBytes"), f"{field}.slotBytes", positive=True),
    )


def parse_planned_pool(document: Mapping[str, object]) -> PlannedPool:
    """Parse protobuf JSON and independently recheck every planning invariant."""
    root = _mapping(document, "planned pool document")
    if root.get("code") not in {"OK", "ALREADY_EXISTS"}:
        raise ValueError(f"planned pool result is not attachable: {root.get('code')!r}")
    descriptor = _mapping(root.get("poolDescriptor"), "poolDescriptor")
    if descriptor.get("state") != "READY":
        raise ValueError("poolDescriptor must be READY before workers attach")
    plan = _mapping(root.get("plan"), "plan")
    spec = _mapping(plan.get("normalizedSpec"), "plan.normalizedSpec")
    version = _uint(spec.get("planningVersion"), "planning version", positive=True)
    if version != 1:
        raise ValueError(f"unsupported planning version {version}")

    pool_name = _text(spec.get("name"), "plan.normalizedSpec.name")
    descriptor_name = _text(descriptor.get("name"), "poolDescriptor.name")
    plan_config = _mapping(plan.get("config"), "plan.config")
    config_name = _text(plan_config.get("name"), "plan.config.name")
    effective = _mapping(descriptor.get("effectiveConfig"), "effective config")
    effective_name = _text(effective.get("name"), "effectiveConfig.name")
    if len({pool_name, descriptor_name, config_name, effective_name}) != 1:
        raise ValueError("descriptor, specification, and plan config name mismatch")
    plan_metadata = _mapping(plan_config.get("metadata"), "plan.config.metadata")
    effective_metadata = _mapping(effective.get("metadata"), "effectiveConfig.metadata")
    metadata_max_chunks = _uint(
        plan_metadata.get("maxChunks"), "plan.config.metadata.maxChunks", positive=True
    )
    if (
        _uint(
            effective_metadata.get("maxChunks"),
            "effectiveConfig.metadata.maxChunks",
            positive=True,
        )
        != metadata_max_chunks
    ):
        raise ValueError("descriptor effective metadata differs from the planned config")

    tp_size = _uint(
        spec.get("tensorParallelSize"),
        "plan.normalizedSpec.tensorParallelSize",
        positive=True,
    )
    model = _text(spec.get("model"), "plan.normalizedSpec.model")
    revision = _text(
        spec.get("modelRevision", ""),
        "plan.normalizedSpec.modelRevision",
        allow_empty=True,
    )
    kv_dtype = _text(spec.get("kvDtype"), "plan.normalizedSpec.kvDtype")
    block_size = _uint(spec.get("blockSize"), "plan.normalizedSpec.blockSize", positive=True)
    budget = _uint(
        spec.get("cpuBytesToUse"),
        "plan.normalizedSpec.cpuBytesToUse",
        positive=True,
    )

    arenas = _sequence(plan_config.get("dataPools"), "plan.config.dataPools")
    effective_arenas = _sequence(effective.get("dataPools"), "effectiveConfig.dataPools")
    if len(arenas) != 1 or len(effective_arenas) != 1:
        raise ValueError("a planned shared pool must contain exactly one data arena")
    arena_row = _mapping(arenas[0], "plan.config.dataPools[0]")
    effective_arena_row = _mapping(effective_arenas[0], "effectiveConfig.dataPools[0]")
    arena_name = _text(arena_row.get("name"), "planned arena name")
    if _text(effective_arena_row.get("name"), "effective arena name") != arena_name:
        raise ValueError("descriptor effective arena differs from the planned config")
    backend = _text(arena_row.get("backend"), "planned arena backend")
    if (
        backend != "HOST_DRAM"
        or _text(effective_arena_row.get("backend"), "effective arena backend") != backend
    ):
        raise ValueError(f"unsupported planned arena backend {backend!r}")
    compatibility = _mapping(arena_row.get("compatibility"), "arena compatibility")
    effective_compatibility = _mapping(
        effective_arena_row.get("compatibility"), "effective arena compatibility"
    )
    expected_model = model if not revision else f"{model}@{revision}"
    if _text(compatibility.get("model"), "compatibility.model") != expected_model:
        raise ValueError("arena model compatibility does not match the normalized spec")
    if _text(compatibility.get("kvDtype"), "compatibility.kvDtype") != kv_dtype:
        raise ValueError("arena KV dtype compatibility does not match the normalized spec")
    if _uint(compatibility.get("blockSize"), "compatibility.blockSize") != block_size:
        raise ValueError("arena block size compatibility does not match the normalized spec")
    if (
        _uint(
            compatibility.get("tensorParallelSize"),
            "compatibility.tensorParallelSize",
        )
        != tp_size
    ):
        raise ValueError("arena tensor parallel compatibility does not match the plan")
    layout_signature = _text(compatibility.get("layoutSignature"), "arena layout signature")
    if len(layout_signature) != 40 or any(
        character not in "0123456789abcdef" for character in layout_signature
    ):
        raise ValueError("arena layout signature must be 40 lowercase hexadecimal digits")
    compatibility_fields = (
        "model",
        "kvDtype",
        "blockSize",
        "tensorParallelSize",
        "layoutSignature",
    )
    if any(
        effective_compatibility.get(field) != compatibility.get(field)
        for field in compatibility_fields
    ):
        raise ValueError("descriptor effective compatibility differs from the planned config")

    partitions = tuple(
        _parse_partition(value, f"plan.partitions[{index}]")
        for index, value in enumerate(_sequence(plan.get("partitions"), "plan.partitions"))
    )
    partition_ids = [item.partition_id for item in partitions]
    partition_names = [item.partition_name for item in partitions]
    partition_keys = [(item.tp_rank, item.cache_index) for item in partitions]
    if (
        len(set(partition_ids)) != len(partitions)
        or len(set(partition_names)) != len(partitions)
        or len(set(partition_keys)) != len(partitions)
    ):
        raise ValueError("duplicate planned partition identity")
    if partition_ids != list(range(len(partitions))):
        raise ValueError("planned partition IDs must be contiguous and ordered")
    if partition_keys != sorted(partition_keys):
        raise ValueError("planned partitions must be sorted by rank and cache index")
    if {item.tp_rank for item in partitions} != set(range(tp_size)):
        raise ValueError("planned partitions must include every tensor-parallel rank")
    for rank in range(tp_size):
        cache_indices = [item.cache_index for item in partitions if item.tp_rank == rank]
        if cache_indices != list(range(len(cache_indices))):
            raise ValueError(f"planned cache indices for rank {rank} are not contiguous")

    expected_offset = 0
    for item in partitions:
        expected_name = f"host-r{item.tp_rank}-c{item.cache_index}"
        if item.partition_name != expected_name:
            raise ValueError(
                f"planned partition name {item.partition_name!r} does not equal "
                f"the deterministic name {expected_name!r}"
            )
        if item.arena_name != arena_name:
            raise ValueError(f"partition {item.partition_name!r} refers to a foreign arena")
        if item.offset_bytes % mmap.PAGESIZE or item.slot_bytes % mmap.PAGESIZE:
            raise ValueError("planned partition offsets and slot strides must be page aligned")
        if item.slot_bytes != _align_up(
            item.physical_page_bytes, mmap.PAGESIZE, "partition slot alignment"
        ):
            raise ValueError(
                f"partition {item.partition_name!r} slot stride does not match its physical page"
            )
        if item.offset_bytes != expected_offset:
            raise ValueError(
                f"partition {item.partition_name!r} offset {item.offset_bytes} "
                f"does not equal exact prefix {expected_offset}"
            )
        expected_offset = _checked_add(
            expected_offset,
            _checked_multiply(item.num_slots, item.slot_bytes, "partition byte extent"),
            "arena byte extent",
        )

    config_partitions = tuple(
        _configuration_partition(value, f"arena.partitions[{index}]")
        for index, value in enumerate(_sequence(arena_row.get("partitions"), "arena.partitions"))
    )
    plan_partition_geometry = tuple(
        (
            item.partition_name,
            item.tp_rank,
            item.cache_index,
            item.offset_bytes,
            item.num_slots,
            item.slot_bytes,
        )
        for item in partitions
    )
    if config_partitions != plan_partition_geometry:
        raise ValueError("arena partition geometry differs from the partition plan")
    effective_config_partitions = tuple(
        _configuration_partition(value, f"effectiveArena.partitions[{index}]")
        for index, value in enumerate(
            _sequence(effective_arena_row.get("partitions"), "effectiveArena.partitions")
        )
    )
    if effective_config_partitions != plan_partition_geometry:
        raise ValueError("descriptor effective config partition geometry differs from the plan")

    raw_rank_caches = _sequence(spec.get("rankCaches"), "plan.normalizedSpec.rankCaches")
    rank_caches = []
    for index, raw_cache in enumerate(raw_rank_caches):
        cache = _mapping(raw_cache, f"rankCaches[{index}]")
        rank_caches.append(
            PlannedRankCache(
                tp_rank=_uint(cache.get("tpRank", 0), f"rankCaches[{index}].tpRank"),
                cache_index=_uint(cache.get("cacheIndex", 0), f"rankCaches[{index}].cacheIndex"),
                physical_page_bytes=_uint(
                    cache.get("physicalPageBytes"),
                    f"rankCaches[{index}].physicalPageBytes",
                    positive=True,
                ),
                layout_kind=_text(cache.get("layoutKind"), f"rankCaches[{index}].layoutKind"),
                layout_version=_uint(
                    cache.get("layoutVersion"),
                    f"rankCaches[{index}].layoutVersion",
                    positive=True,
                ),
                local_kv_heads=_uint(
                    cache.get("localKvHeads"),
                    f"rankCaches[{index}].localKvHeads",
                    positive=True,
                ),
                head_size=_uint(
                    cache.get("headSize"),
                    f"rankCaches[{index}].headSize",
                    positive=True,
                ),
            )
        )
    cache_keys = [(item.tp_rank, item.cache_index) for item in rank_caches]
    if len(set(cache_keys)) != len(cache_keys):
        raise ValueError("duplicate normalized rank cache identity")
    if cache_keys != partition_keys:
        raise ValueError("normalized rank caches do not match planned partitions")
    if any(
        cache.physical_page_bytes != partition.physical_page_bytes
        for cache, partition in zip(rank_caches, partitions, strict=True)
    ):
        raise ValueError("partition physical page geometry differs from normalized cache rows")
    if any(
        cache.layout_kind not in {"token-major", "head-major"} or cache.layout_version != 1
        for cache in rank_caches
    ):
        raise ValueError("normalized cache layout kind or version is unsupported")
    canonical_layout = "".join(
        f"{cache.tp_rank}:{cache.cache_index}:{cache.physical_page_bytes}:"
        f"{cache.layout_kind}:{cache.layout_version}:{cache.local_kv_heads}:"
        f"{cache.head_size};"
        for cache in rank_caches
    )
    if hashlib.sha1(canonical_layout.encode()).hexdigest() != layout_signature:
        raise ValueError("arena layout signature does not match normalized cache geometry")

    capacity = _uint(plan.get("logicalBlockCapacity"), "logicalBlockCapacity", positive=True)
    slots_per_partition = _checked_multiply(2, capacity, "partition slot count")
    if any(item.num_slots != slots_per_partition for item in partitions):
        raise ValueError("partition geometry does not match logicalBlockCapacity")
    bytes_per_block = _checked_multiply(
        2,
        sum(item.slot_bytes for item in partitions),
        "bytes per complete block",
    )
    if _uint(plan.get("bytesPerCompleteBlock"), "bytesPerCompleteBlock") != bytes_per_block:
        raise ValueError("bytesPerCompleteBlock does not match partition geometry")
    allocated = _checked_multiply(bytes_per_block, capacity, "allocated bytes")
    if _uint(plan.get("allocatedBytes"), "allocatedBytes") != allocated:
        raise ValueError("allocatedBytes does not match capacity arithmetic")
    if expected_offset != allocated:
        raise ValueError("partition extents do not match allocatedBytes")
    if budget < allocated:
        raise ValueError("planned allocation exceeds cpuBytesToUse")
    unused = budget - allocated
    if _uint(plan.get("unusedBytes"), "unusedBytes") != unused:
        raise ValueError("unusedBytes does not match budget arithmetic")
    if unused >= bytes_per_block:
        raise ValueError("logicalBlockCapacity is not maximal for cpuBytesToUse")
    effective_arena_slots = _uint(
        effective_arena_row.get("numSlots"),
        "effective arena numSlots",
        positive=True,
    )
    effective_arena_stride = _uint(
        effective_arena_row.get("slotBytes"),
        "effective arena slotBytes",
        positive=True,
    )
    if (
        effective_arena_stride != mmap.PAGESIZE
        or effective_arena_slots != allocated // mmap.PAGESIZE
    ):
        raise ValueError("descriptor effective arena summary is not canonical")
    planned_arena_slots = _uint(arena_row.get("numSlots", 0), "planned arena numSlots")
    planned_arena_stride = _uint(arena_row.get("slotBytes", 0), "planned arena slotBytes")
    if (planned_arena_slots, planned_arena_stride) not in {
        (0, 0),
        (effective_arena_slots, effective_arena_stride),
    }:
        raise ValueError("planned arena summary is inconsistent with effective geometry")

    components = tuple(
        _parse_component(value, f"plan.components[{index}]")
        for index, value in enumerate(_sequence(plan.get("components"), "plan.components"))
    )
    component_ids = [item.global_component_id for item in components]
    if len(set(component_ids)) != len(components):
        raise ValueError("duplicate global component ID")
    if component_ids != list(range(len(components))):
        raise ValueError("global component IDs must be contiguous and ordered")
    partitions_by_id = {item.partition_id: item for item in partitions}
    roles_by_partition: dict[int, set[str]] = {item.partition_id: set() for item in partitions}
    for component in components:
        partition = partitions_by_id.get(component.partition_id)
        if partition is None or (
            component.partition_name != partition.partition_name
            or component.arena_name != partition.arena_name
            or component.tp_rank != partition.tp_rank
            or component.cache_index != partition.cache_index
            or component.partition_offset_bytes != partition.offset_bytes
            or component.partition_slot_bytes != partition.slot_bytes
            or component.physical_page_bytes != partition.physical_page_bytes
        ):
            raise ValueError(
                f"component {component.global_component_id} has a foreign partition reference"
            )
        if component.role in roles_by_partition[partition.partition_id]:
            raise ValueError("duplicate component role for one partition")
        roles_by_partition[partition.partition_id].add(component.role)
    if any(roles != {"k", "v"} for roles in roles_by_partition.values()):
        raise ValueError("every planned partition requires one complete K/V component pair")
    expected_component_order = [
        (partition.tp_rank, partition.cache_index, role)
        for partition in partitions
        for role in ("k", "v")
    ]
    actual_component_order = [
        (component.tp_rank, component.cache_index, component.role) for component in components
    ]
    if actual_component_order != expected_component_order:
        raise ValueError("planned components must use canonical rank/cache/K-then-V order")

    arena = PlannedArena(
        name=arena_name,
        backend=backend,
        model=expected_model,
        kv_dtype=kv_dtype,
        block_size=block_size,
        tensor_parallel_size=tp_size,
        partitions=partitions,
        num_slots=effective_arena_slots,
        slot_bytes=effective_arena_stride,
        layout_signature=layout_signature,
    )
    return PlannedPool(
        identity=SharedPoolIdentity(
            pool_name, _generation(descriptor.get("generation")), arena_name
        ),
        planning_version=version,
        model=model,
        model_revision=revision,
        kv_dtype=kv_dtype,
        block_size=block_size,
        tensor_parallel_size=tp_size,
        cpu_bytes_to_use=budget,
        bytes_per_complete_block=bytes_per_block,
        logical_block_capacity=capacity,
        allocated_bytes=allocated,
        unused_bytes=unused,
        arena=arena,
        partitions=partitions,
        components=components,
        rank_caches=tuple(rank_caches),
        metadata_max_chunks=metadata_max_chunks,
    )


def _decode_inline(value: object, field: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return _mapping(value, field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a JSON object or encoded JSON string")
    try:
        return _mapping(json.loads(value), field)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON: {error.msg}") from error


def load_planned_pool(
    extra_config: Mapping[str, object], environ: Mapping[str, str] = os.environ
) -> PlannedPool:
    """Load explicit inline/path configuration before environment fallbacks."""
    if "pool_plan" in extra_config:
        return parse_planned_pool(_decode_inline(extra_config["pool_plan"], "pool_plan"))
    path_value = extra_config.get("pool_plan_path")
    if path_value is None:
        if "SPYRE_KV_POOL_PLAN" in environ:
            return parse_planned_pool(
                _decode_inline(environ["SPYRE_KV_POOL_PLAN"], "SPYRE_KV_POOL_PLAN")
            )
        path_value = environ.get("SPYRE_KV_POOL_PLAN_PATH")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(
            "pool_plan or pool_plan_path must be specified in "
            "kv_connector_extra_config or the Spyre KV pool environment"
        )
    path = Path(path_value)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load pool_plan_path {path}: {error}") from error
    return parse_planned_pool(_mapping(value, "pool_plan_path document"))
