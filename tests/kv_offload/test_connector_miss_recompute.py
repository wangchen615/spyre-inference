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

"""A shared-directory miss leaves the block for ordinary model execution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import ReqContext, make_offload_key

from spyre_inference.v1.kv_offload.connector import SpyreOffloadingConnector
from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
from spyre_inference.v1.kv_offload.shared_manager import (
    SpyreSharedOffloadingManager,
)

OFFLOAD_KEY = make_offload_key(bytes.fromhex("11" * 32), 0)
GENERATION = bytes.fromhex("44" * 16)


class EmptyDirectory:
    def __init__(self):
        self.pin_calls = []

    def find_pool(self, name):
        return self.registered if name == self.registered.name else None

    def resolve_pool(self, pool_ref):
        return self.data_pool if pool_ref is self.registered.pool_ref else None

    def lookup(self, key):
        return None

    def pin_read(self, entry):
        self.pin_calls.append(entry)
        return object()


def _manager_with_empty_directory():
    compatibility = SimpleNamespace(metadata_version=1, compatibility_id=7)
    registered = SimpleNamespace(
        name="shared.data",
        pool_ref=SimpleNamespace(pool_id=10, pool_version=1),
        compatibility=compatibility,
        slot_count=4,
    )
    directory = EmptyDirectory()
    directory.registered = registered
    directory.data_pool = object()
    ok = object()
    runtime = SimpleNamespace(
        SharedPool=SimpleNamespace(
            attach_pool=lambda _name, _generation: SimpleNamespace(
                code=ok, metadata=directory, detail=""
            ),
            detach_pool=lambda _metadata: None,
        ),
        PoolResultCode=SimpleNamespace(OK=ok),
        CompatibleBlockKey=lambda compatibility, block_hash: SimpleNamespace(
            compatibility=compatibility, block_hash=block_hash
        ),
    )
    manager = SpyreSharedOffloadingManager(
        identity=SharedPoolIdentity("shared-root", GENERATION, "shared.data"),
        component_count=2,
        max_pool_slots=4,
        cache_policy="lru",
        cache_policy_module_path=None,
        enable_events=False,
        store_threshold=0,
        max_tracker_size=64_000,
        runtime_loader=lambda: runtime,
    )
    return manager, directory


def test_shared_directory_miss_reports_zero_external_tokens():
    manager, directory = _manager_with_empty_directory()
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = manager
    scheduler._events_tracker = SimpleNamespace(record_lookup=lambda *args: None)

    matched = scheduler._maximal_prefix_lookup(
        [OFFLOAD_KEY], ReqContext("request-1"), MagicMock(), MagicMock(), 0
    )

    assert matched == 0
    assert directory.pin_calls == []


def test_zero_external_tokens_do_not_create_an_h2d_job():
    class NoLoadManager:
        def prepare_load(self, *args):
            raise AssertionError("prepare_load must not run for a cache miss")

    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = NoLoadManager()
    scheduler._current_batch_load_jobs = {}
    scheduler._jobs = {}

    result = scheduler.update_state_after_alloc(
        SimpleNamespace(request_id="request-1"), MagicMock(), num_external_tokens=0
    )

    assert result is None
    assert scheduler._current_batch_load_jobs == {}
    assert scheduler._jobs == {}


def test_external_launcher_scheduler_counts_one_completion_per_engine(monkeypatch):
    class SchedulerConfig(NamedTuple):
        num_workers: int

    scheduler = SimpleNamespace(config=SchedulerConfig(num_workers=4))

    def fake_init(connector, *_args):
        connector.connector_scheduler = scheduler
        connector.connector_worker = None

    monkeypatch.setattr(OffloadingConnector, "__init__", fake_init)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend="external_launcher")
    )

    SpyreOffloadingConnector(config, object(), object())

    assert scheduler.config.num_workers == 1
