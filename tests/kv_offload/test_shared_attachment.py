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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace

import pytest

from spyre_inference.v1.kv_offload.pool_plan import parse_planned_pool
from spyre_inference.v1.kv_offload.shared_attachment import (
    SharedMetadataAttachment,
    SharedPoolAttachment,
)
from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
from tests.kv_offload.test_pool_plan import valid_planned_pool_document

GENERATION = bytes.fromhex("11" * 16)
IDENTITY = SharedPoolIdentity("root", GENERATION, "host-0")
POOL_REF = object()


class _Code(Enum):
    OK = auto()
    NOT_FOUND = auto()
    INITIALIZING = auto()
    DESTROYING = auto()
    STALE_GENERATION = auto()
    CONFIGURATION_MISMATCH = auto()
    RESOURCE_ERROR = auto()


class _Backend(Enum):
    HOST_DRAM = auto()


@dataclass(frozen=True)
class _Compatibility:
    model: str
    kv_dtype: str
    block_size: int
    tensor_parallel_size: int
    layout_signature: str


@dataclass(frozen=True)
class _PartitionConfig:
    name: str
    tp_rank: int
    cache_index: int
    num_slots: int
    slot_bytes: int
    offset_bytes: int


@dataclass(frozen=True)
class _ArenaConfig:
    name: str
    backend: _Backend
    num_slots: int
    slot_bytes: int
    partitions: list[_PartitionConfig]
    compatibility: _Compatibility


@dataclass(frozen=True)
class _MetadataConfig:
    max_chunks: int


@dataclass(frozen=True)
class _PoolConfig:
    name: str
    metadata: _MetadataConfig
    data_pools: list[_ArenaConfig]


@pytest.fixture
def fake_runtime():
    calls = []
    metadata = SimpleNamespace()
    registered = SimpleNamespace(
        name="host-0",
        pool_ref=POOL_REF,
        slot_count=8,
        slot_bytes=4096,
    )
    data_pool = object()
    native_attachment = SimpleNamespace(
        tp_rank=0,
        metadata=metadata,
        data_pools=[SimpleNamespace(registration=registered, data_pool=data_pool)],
    )
    state = SimpleNamespace(
        attach_code=_Code.OK,
        attach_attachment=native_attachment,
        detail="",
        find_result=registered,
        find_error=None,
    )

    def find_pool(name):
        calls.append(("find", name))
        if state.find_error is not None:
            raise state.find_error
        return state.find_result

    metadata.find_pool = find_pool

    class FakeSharedPool:
        @staticmethod
        def attach_pool(name, generation, expected_config_or_rank, tp_rank=None):
            call = ("attach", name, generation)
            if tp_rank is None:
                call += (expected_config_or_rank,)
            else:
                call += (expected_config_or_rank, tp_rank)
            calls.append(call)
            return SimpleNamespace(
                code=state.attach_code,
                attachment=state.attach_attachment,
                detail=state.detail,
            )

        @staticmethod
        def detach_pool(value):
            calls.append(("detach", value))

        @staticmethod
        def create_pool(*_args):
            raise AssertionError("worker attachment must not create pools")

        @staticmethod
        def open_pool(*_args):
            raise AssertionError("worker attachment must not open pools")

        @staticmethod
        def destroy_pool(*_args):
            raise AssertionError("worker attachment must not destroy pools")

    return SimpleNamespace(
        SharedPool=FakeSharedPool,
        PoolResultCode=_Code,
        calls=calls,
        state=state,
        metadata=metadata,
        native_attachment=native_attachment,
        registered=registered,
        data_pool=data_pool,
        PoolBackend=_Backend,
        PoolCompatibility=_Compatibility,
        SharedDataPartitionConfig=_PartitionConfig,
        SharedDataPoolConfig=_ArenaConfig,
        MetadataConfig=_MetadataConfig,
        PoolConfig=_PoolConfig,
    )


def test_legacy_attach_eagerly_returns_rank_zero_child_and_close_detaches_once(
    fake_runtime,
):
    attachment = SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)

    assert fake_runtime.calls == [("attach", "root", GENERATION, 0)]
    assert attachment.metadata is fake_runtime.metadata
    assert attachment.registered is fake_runtime.registered
    assert attachment.data_pool is fake_runtime.data_pool

    attachment.close()
    attachment.close()

    assert fake_runtime.calls.count(("detach", fake_runtime.native_attachment)) == 1
    assert attachment.metadata is None
    assert attachment.registered is None
    assert attachment.data_pool is None


def test_legacy_attach_selects_named_child_from_all_eager_rank_zero_children(
    fake_runtime,
):
    extra = SimpleNamespace(
        registration=SimpleNamespace(name="host-extra"),
        data_pool=object(),
    )
    fake_runtime.native_attachment.data_pools.append(extra)

    attachment = SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)

    assert attachment.registered is fake_runtime.registered
    assert attachment.data_pool is fake_runtime.data_pool
    attachment.close()
    assert fake_runtime.calls.count(("detach", fake_runtime.native_attachment)) == 1


@pytest.mark.parametrize(
    "code",
    [
        "NOT_FOUND",
        "STALE_GENERATION",
        "INITIALIZING",
        "DESTROYING",
        "CONFIGURATION_MISMATCH",
        "RESOURCE_ERROR",
    ],
)
def test_structured_attach_failure_names_result(code, fake_runtime):
    fake_runtime.state.attach_code = getattr(_Code, code)
    fake_runtime.state.attach_attachment = None
    fake_runtime.state.detail = "runtime detail"

    with pytest.raises(RuntimeError, match=rf"{code}.*runtime detail"):
        SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)

    assert fake_runtime.calls == [("attach", "root", GENERATION, 0)]


def test_ok_without_attachment_is_rejected(fake_runtime):
    fake_runtime.state.attach_attachment = None

    with pytest.raises(RuntimeError, match="OK.*no attachment"):
        SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)


def test_missing_eager_child_after_attach_detaches_once(fake_runtime):
    fake_runtime.native_attachment.data_pools = []

    with pytest.raises(RuntimeError, match="was not eagerly attached"):
        SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)

    assert fake_runtime.calls.count(("detach", fake_runtime.native_attachment)) == 1


def test_attachment_path_never_invokes_management_calls(fake_runtime):
    attachment = SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)
    attachment.close()

    operation_names = [call[0] for call in fake_runtime.calls]
    assert "create" not in operation_names
    assert "open" not in operation_names
    assert "destroy" not in operation_names


def test_finalizer_stays_no_throw_during_module_teardown(fake_runtime, monkeypatch):
    attachment = SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)
    monkeypatch.setattr(attachment, "close", lambda: (_ for _ in ()).throw(RuntimeError("late")))

    attachment.__del__()

    assert fake_runtime.calls.count(("detach", fake_runtime.native_attachment)) == 0


def _configure_planned_rank(fake_runtime, plan, tp_rank):
    registrations = {
        partition.partition_name: SimpleNamespace(
            name=partition.partition_name,
            pool_ref=object(),
            slot_count=partition.num_slots,
            slot_bytes=partition.slot_bytes,
        )
        for partition in plan.partitions
    }
    fake_runtime.metadata.find_pool = lambda name: (
        fake_runtime.calls.append(("find", name)) or registrations.get(name)
    )
    local_entries = [
        SimpleNamespace(
            registration=registrations[partition.partition_name],
            data_pool=f"view:{partition.partition_name}",
        )
        for partition in plan.partitions_for_rank(tp_rank)
    ]
    native_attachment = SimpleNamespace(
        tp_rank=tp_rank,
        metadata=fake_runtime.metadata,
        data_pools=local_entries,
    )
    fake_runtime.state.attach_attachment = native_attachment
    return registrations, native_attachment


def test_planned_attach_registers_global_directory_and_returns_eager_rank_entries(
    fake_runtime,
):
    plan = parse_planned_pool(valid_planned_pool_document())
    registrations, native_attachment = _configure_planned_rank(fake_runtime, plan, 1)

    attachment = SharedMetadataAttachment.attach(plan, 1, lambda: fake_runtime)

    assert set(attachment.registered) == set(registrations)
    expected_config = fake_runtime.calls[0][3]
    assert fake_runtime.calls[0] == (
        "attach",
        plan.identity.pool_name,
        plan.identity.generation,
        expected_config,
        1,
    )
    assert expected_config.name == plan.identity.pool_name
    assert len(expected_config.data_pools) == plan.tensor_parallel_size
    assert expected_config.data_pools[0].num_slots == plan.arena_for_rank(0).num_slots
    assert expected_config.data_pools[0].partitions[0].offset_bytes == 0
    assert [item.partition.tp_rank for item in attachment.local_partitions] == [1]
    assert [item.registered for item in attachment.local_partitions] == [
        registrations["host-r1-c0"]
    ]
    assert [item.data_pool for item in attachment.local_partitions] == ["view:host-r1-c0"]
    attachment.close()
    assert fake_runtime.calls.count(("detach", native_attachment)) == 1


def test_planned_attach_rejects_native_attachment_for_another_rank(fake_runtime):
    plan = parse_planned_pool(valid_planned_pool_document())
    _, native_attachment = _configure_planned_rank(fake_runtime, plan, 0)

    with pytest.raises(RuntimeError, match="returned tensor-parallel rank 0, expected 1"):
        SharedMetadataAttachment.attach(plan, 1, lambda: fake_runtime)

    assert fake_runtime.calls.count(("detach", native_attachment)) == 1


def test_planned_attach_rejects_missing_eager_local_partition(fake_runtime):
    plan = parse_planned_pool(valid_planned_pool_document())
    _, native_attachment = _configure_planned_rank(fake_runtime, plan, 1)
    native_attachment.data_pools = []

    with pytest.raises(RuntimeError, match="missing eager rank-local partitions"):
        SharedMetadataAttachment.attach(plan, 1, lambda: fake_runtime)

    assert fake_runtime.calls.count(("detach", native_attachment)) == 1
