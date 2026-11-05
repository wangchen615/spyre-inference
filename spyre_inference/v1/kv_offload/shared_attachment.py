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

"""Worker-side ownership of one persistent shared-pool attachment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from spyre_inference.v1.kv_offload.pool_plan import PlannedPartition, PlannedPool
from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
from spyre_inference.v1.kv_offload.shared_runtime import load_shared_runtime


def _code_name(code: object) -> str:
    return str(getattr(code, "name", code))


def _attach_error(pool_name: str, result: object) -> str:
    code = _code_name(getattr(result, "code", "UNKNOWN"))
    detail = str(getattr(result, "detail", "") or "")
    message = f"cannot attach shared pool {pool_name!r}: {code}"
    if detail:
        message += f": {detail}"
    return message


@dataclass(frozen=True)
class ResolvedPartition:
    partition: PlannedPartition
    registered: Any
    data_pool: Any


def _expected_native_config(runtime: Any, plan: PlannedPool) -> Any:
    arenas = []
    for arena in plan.arenas:
        compatibility = runtime.PoolCompatibility(
            arena.model,
            arena.kv_dtype,
            arena.block_size,
            arena.tensor_parallel_size,
            arena.layout_signature,
        )
        partitions = [
            runtime.SharedDataPartitionConfig(
                partition.partition_name,
                partition.tp_rank,
                partition.cache_index,
                partition.num_slots,
                partition.slot_bytes,
                partition.offset_bytes,
            )
            for partition in arena.partitions
        ]
        arenas.append(
            runtime.SharedDataPoolConfig(
                arena.name,
                runtime.PoolBackend.HOST_DRAM,
                arena.num_slots,
                arena.slot_bytes,
                partitions,
                compatibility,
            )
        )
    return runtime.PoolConfig(
        plan.identity.pool_name,
        runtime.MetadataConfig(plan.metadata_max_chunks),
        arenas,
    )


class SharedMetadataAttachment:
    """Own one eager rank-scoped pool attachment and its global directory."""

    def __init__(
        self,
        runtime: Any,
        native_attachment: Any,
        plan: PlannedPool | None,
        registered: dict,
        attached_data: dict,
        local_partitions: tuple[ResolvedPartition, ...],
    ) -> None:
        self._runtime = runtime
        self._native_attachment = native_attachment
        self.metadata = native_attachment.metadata
        self.plan = plan
        self.registered = MappingProxyType(dict(registered))
        self._attached_data = MappingProxyType(dict(attached_data))
        self.local_partitions = local_partitions

    @classmethod
    def attach(
        cls,
        plan: PlannedPool,
        tp_rank: int,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> SharedMetadataAttachment:
        expected = {partition.partition_name: partition for partition in plan.partitions}
        return cls._attach_identity(plan.identity, expected, plan, tp_rank, runtime_loader)

    @classmethod
    def _attach_identity(
        cls,
        identity: SharedPoolIdentity,
        expected: dict[str, PlannedPartition | None],
        plan: PlannedPool | None,
        tp_rank: int,
        runtime_loader: Callable[[], Any],
    ) -> SharedMetadataAttachment:
        runtime = runtime_loader()
        if plan is None:
            result = runtime.SharedPool.attach_pool(
                identity.pool_name,
                identity.generation,
                tp_rank,
            )
        else:
            result = runtime.SharedPool.attach_pool(
                identity.pool_name,
                identity.generation,
                _expected_native_config(runtime, plan),
                tp_rank,
            )
        native_attachment = result.attachment
        if result.code is not runtime.PoolResultCode.OK or native_attachment is None:
            message = _attach_error(identity.pool_name, result)
            if result.code is runtime.PoolResultCode.OK and native_attachment is None:
                message += ": no attachment returned"
            raise RuntimeError(message)

        try:
            attached_rank = int(native_attachment.tp_rank)
            if attached_rank != tp_rank:
                raise RuntimeError(
                    f"shared pool {identity.pool_name!r} returned tensor-parallel "
                    f"rank {attached_rank}, expected {tp_rank}"
                )
            metadata = native_attachment.metadata
            if metadata is None:
                raise RuntimeError(f"shared pool {identity.pool_name!r} attachment has no metadata")
            attached_data = {}
            for item in native_attachment.data_pools:
                name = str(item.registration.name)
                if name in attached_data:
                    raise RuntimeError(
                        f"shared pool {identity.pool_name!r} returned duplicate eager "
                        f"partition {name!r}"
                    )
                attached_data[name] = item

            if plan is None:
                local_partitions = ()
                expected_local_names = set(expected)
            else:
                local_plan = plan.partitions_for_rank(tp_rank)
                expected_local_names = {partition.partition_name for partition in local_plan}
                local_partitions = tuple(
                    ResolvedPartition(
                        partition,
                        attached_data[partition.partition_name].registration,
                        attached_data[partition.partition_name].data_pool,
                    )
                    for partition in local_plan
                    if partition.partition_name in attached_data
                )

            attached_names = set(attached_data)
            missing = expected_local_names - attached_names
            unexpected = attached_names - expected_local_names
            if missing:
                if plan is None:
                    name = next(iter(missing))
                    raise RuntimeError(
                        f"data pool {name!r} was not eagerly attached in shared "
                        f"pool {identity.pool_name!r}"
                    )
                raise RuntimeError(
                    "missing eager rank-local partitions: " + ", ".join(sorted(missing))
                )
            if unexpected and plan is not None:
                raise RuntimeError(
                    "unexpected eager rank-local partitions: " + ", ".join(sorted(unexpected))
                )

            registered = {}
            for partition_name, partition in expected.items():
                item = attached_data.get(partition_name)
                row = item.registration if item is not None else metadata.find_pool(partition_name)
                if row is None:
                    if partition is None:
                        raise RuntimeError(
                            f"data pool {partition_name!r} is not configured "
                            f"in shared pool {identity.pool_name!r}"
                        )
                    raise RuntimeError(
                        f"planned partition {partition_name!r} is not "
                        f"registered in shared pool {identity.pool_name!r}"
                    )
                if partition is not None and (
                    int(row.slot_count) != partition.num_slots
                    or int(row.slot_bytes) != partition.slot_bytes
                ):
                    raise RuntimeError(
                        f"registered partition {partition_name!r} geometry "
                        "does not match the immutable pool plan"
                    )
                registered[partition_name] = row
            return cls(
                runtime,
                native_attachment,
                plan,
                registered,
                attached_data,
                local_partitions,
            )
        except BaseException:
            runtime.SharedPool.detach_pool(native_attachment)
            raise

    def close(self) -> None:
        native_attachment = getattr(self, "_native_attachment", None)
        if native_attachment is None:
            return
        runtime = self._runtime
        self.local_partitions = ()
        self._attached_data = MappingProxyType({})
        self.registered = MappingProxyType({})
        self.metadata = None
        self._runtime = None
        self._native_attachment = None
        runtime.SharedPool.detach_pool(native_attachment)

    def __del__(self) -> None:
        try:  # noqa: SIM105 - globals may be cleared during interpreter teardown
            self.close()
        except BaseException:
            pass


class SharedPoolAttachment:
    """TP1 compatibility wrapper owning metadata and one resolved child."""

    def __init__(
        self,
        metadata_attachment: SharedMetadataAttachment,
        registered: Any,
        data_pool: Any,
    ) -> None:
        self._metadata_attachment = metadata_attachment
        self.metadata = metadata_attachment.metadata
        self.registered = registered
        self.data_pool = data_pool

    @classmethod
    def attach(
        cls,
        identity: SharedPoolIdentity,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> SharedPoolAttachment:
        metadata_attachment = SharedMetadataAttachment._attach_identity(
            identity,
            {identity.data_pool_name: None},
            None,
            0,
            runtime_loader,
        )
        try:
            attached = metadata_attachment._attached_data[identity.data_pool_name]
            registered = attached.registration
            data_pool = attached.data_pool
            return cls(metadata_attachment, registered, data_pool)
        except BaseException:
            metadata_attachment.close()
            raise

    def close(self) -> None:
        """Release child handles, then detach the root exactly once."""

        metadata = getattr(self, "metadata", None)
        if metadata is None:
            return
        metadata_attachment = self._metadata_attachment
        self.data_pool = None
        self.registered = None
        self.metadata = None
        self._metadata_attachment = None
        metadata_attachment.close()

    def __del__(self) -> None:
        try:  # noqa: SIM105 - globals may be cleared during interpreter teardown
            self.close()
        except BaseException:
            pass
