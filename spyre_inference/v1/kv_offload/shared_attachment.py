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


class SharedMetadataAttachment:
    """Own one root lock lease while leaving data arena views unresolved."""

    def __init__(
        self,
        runtime: Any,
        metadata: Any,
        plan: PlannedPool | None,
        registered: dict,
    ) -> None:
        self._runtime = runtime
        self.metadata = metadata
        self.plan = plan
        self.registered = MappingProxyType(dict(registered))

    @classmethod
    def attach(
        cls,
        plan: PlannedPool,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> SharedMetadataAttachment:
        expected = {partition.partition_name: partition for partition in plan.partitions}
        return cls._attach_identity(plan.identity, expected, plan, runtime_loader)

    @classmethod
    def _attach_identity(
        cls,
        identity: SharedPoolIdentity,
        expected: dict[str, PlannedPartition | None],
        plan: PlannedPool | None,
        runtime_loader: Callable[[], Any],
    ) -> SharedMetadataAttachment:
        runtime = runtime_loader()
        result = runtime.SharedPool.attach_pool(identity.pool_name, identity.generation)
        if result.code is not runtime.PoolResultCode.OK or result.metadata is None:
            message = _attach_error(identity.pool_name, result)
            if result.code is runtime.PoolResultCode.OK and result.metadata is None:
                message += ": no metadata returned"
            raise RuntimeError(message)

        metadata = result.metadata
        try:
            registered = {}
            for partition_name, partition in expected.items():
                row = metadata.find_pool(partition_name)
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
            return cls(runtime, metadata, plan, registered)
        except BaseException:
            runtime.SharedPool.detach_pool(metadata)
            raise

    def resolve_rank(self, tp_rank: int) -> tuple[ResolvedPartition, ...]:
        if self.plan is None:
            raise RuntimeError("rank resolution requires a complete planned pool")
        resolved = []
        for partition in self.plan.partitions_for_rank(tp_rank):
            registered = self.registered[partition.partition_name]
            data_pool = self.metadata.resolve_pool(registered.pool_ref)
            if data_pool is None:
                raise RuntimeError(
                    f"planned partition {partition.partition_name!r} cannot be resolved"
                )
            resolved.append(ResolvedPartition(partition, registered, data_pool))
        return tuple(resolved)

    def close(self) -> None:
        metadata = getattr(self, "metadata", None)
        if metadata is None:
            return
        runtime = self._runtime
        self.registered = MappingProxyType({})
        self.metadata = None
        self._runtime = None
        runtime.SharedPool.detach_pool(metadata)

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
            runtime_loader,
        )
        metadata = metadata_attachment.metadata
        try:
            registered = metadata_attachment.registered[identity.data_pool_name]
            data_pool = metadata.resolve_pool(registered.pool_ref)
            if data_pool is None:
                raise RuntimeError(
                    f"data pool {identity.data_pool_name!r} cannot be resolved "
                    f"in shared pool {identity.pool_name!r}"
                )
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
