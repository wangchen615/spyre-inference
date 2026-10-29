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


@pytest.fixture
def fake_runtime():
    calls = []
    metadata = SimpleNamespace()
    registered = SimpleNamespace(pool_ref=POOL_REF, slot_count=8, slot_bytes=4096)
    data_pool = object()
    state = SimpleNamespace(
        attach_code=_Code.OK,
        attach_metadata=metadata,
        detail="",
        find_result=registered,
        resolve_result=data_pool,
        find_error=None,
        resolve_error=None,
    )

    def find_pool(name):
        calls.append(("find", name))
        if state.find_error is not None:
            raise state.find_error
        return state.find_result

    def resolve_pool(pool_ref):
        calls.append(("resolve", pool_ref))
        if state.resolve_error is not None:
            raise state.resolve_error
        return state.resolve_result

    metadata.find_pool = find_pool
    metadata.resolve_pool = resolve_pool

    class FakeSharedPool:
        @staticmethod
        def attach_pool(name, generation):
            calls.append(("attach", name, generation))
            return SimpleNamespace(
                code=state.attach_code,
                metadata=state.attach_metadata,
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
        registered=registered,
        data_pool=data_pool,
    )


def test_attach_resolves_precreated_child_and_close_detaches_once(fake_runtime):
    attachment = SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)

    assert fake_runtime.calls == [
        ("attach", "root", GENERATION),
        ("find", "host-0"),
        ("resolve", POOL_REF),
    ]
    assert attachment.metadata is fake_runtime.metadata
    assert attachment.registered is fake_runtime.registered
    assert attachment.data_pool is fake_runtime.data_pool

    attachment.close()
    attachment.close()

    assert fake_runtime.calls.count(("detach", fake_runtime.metadata)) == 1
    assert attachment.metadata is None
    assert attachment.registered is None
    assert attachment.data_pool is None


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
    fake_runtime.state.attach_metadata = None
    fake_runtime.state.detail = "runtime detail"

    with pytest.raises(RuntimeError, match=rf"{code}.*runtime detail"):
        SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)

    assert fake_runtime.calls == [("attach", "root", GENERATION)]


def test_ok_without_metadata_is_rejected(fake_runtime):
    fake_runtime.state.attach_metadata = None

    with pytest.raises(RuntimeError, match="OK.*no metadata"):
        SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)


@pytest.mark.parametrize(
    ("failure_stage", "message"),
    [("find_missing", "is not configured"), ("resolve_missing", "cannot be resolved")],
)
def test_resolution_failure_after_attach_detaches_once(fake_runtime, failure_stage, message):
    if failure_stage == "find_missing":
        fake_runtime.state.find_result = None
    else:
        fake_runtime.state.resolve_result = None

    with pytest.raises(RuntimeError, match=message):
        SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)

    assert fake_runtime.calls.count(("detach", fake_runtime.metadata)) == 1


@pytest.mark.parametrize("failure_stage", ["find", "resolve"])
def test_resolution_exception_after_attach_detaches_once(fake_runtime, failure_stage):
    setattr(fake_runtime.state, f"{failure_stage}_error", RuntimeError("resolution failed"))

    with pytest.raises(RuntimeError, match="resolution failed"):
        SharedPoolAttachment.attach(IDENTITY, lambda: fake_runtime)

    assert fake_runtime.calls.count(("detach", fake_runtime.metadata)) == 1


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

    assert fake_runtime.calls.count(("detach", fake_runtime.metadata)) == 0


def test_metadata_attachment_registers_every_partition_without_mapping(fake_runtime):
    plan = parse_planned_pool(valid_planned_pool_document())
    registrations = {
        partition.partition_name: SimpleNamespace(
            pool_ref=object(),
            slot_count=partition.num_slots,
            slot_bytes=partition.slot_bytes,
        )
        for partition in plan.partitions
    }
    fake_runtime.metadata.find_pool = lambda name: (
        fake_runtime.calls.append(("find", name)) or registrations.get(name)
    )

    attachment = SharedMetadataAttachment.attach(plan, lambda: fake_runtime)

    assert set(attachment.registered) == set(registrations)
    assert all(call[0] != "resolve" for call in fake_runtime.calls)
    attachment.close()


def test_rank_resolver_maps_only_local_partitions(fake_runtime):
    plan = parse_planned_pool(valid_planned_pool_document())
    registrations = {
        partition.partition_name: SimpleNamespace(
            pool_ref=partition.partition_name,
            slot_count=partition.num_slots,
            slot_bytes=partition.slot_bytes,
        )
        for partition in plan.partitions
    }
    fake_runtime.metadata.find_pool = registrations.get
    fake_runtime.metadata.resolve_pool = lambda pool_ref: (
        fake_runtime.calls.append(("resolve", pool_ref)) or f"view:{pool_ref}"
    )
    attachment = SharedMetadataAttachment.attach(plan, lambda: fake_runtime)

    resolved = attachment.resolve_rank(1)

    assert [item.partition.tp_rank for item in resolved] == [1]
    assert [call for call in fake_runtime.calls if call[0] == "resolve"] == [
        ("resolve", "host-r1-c0")
    ]
    attachment.close()
