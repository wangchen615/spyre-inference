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
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_offload.base import ReqContext, make_offload_key

from spyre_inference.v1.kv_offload.shared_manager import (
    SpyreSharedOffloadingManager,
)
from spyre_inference.v1.kv_offload.shared_types import SharedPoolFamily

OFFLOAD_KEY = make_offload_key(bytes.fromhex("11" * 32), 0)


class EmptyDirectory:
    def __init__(self):
        self.pin_calls = []

    def lookup(self, key):
        return None

    def pin_read(self, entry):
        self.pin_calls.append(entry)
        return object()


def _manager_with_empty_directory():
    compatibility = SimpleNamespace(metadata_version=1, compatibility_id=7)
    anchor = SimpleNamespace(
        name="alpha.c0.k",
        pool_ref=SimpleNamespace(pool_id=10),
        compatibility=compatibility,
    )
    directory = EmptyDirectory()
    runtime = SimpleNamespace(
        CompatibleBlockKey=lambda compatibility, block_hash: SimpleNamespace(
            compatibility=compatibility, block_hash=block_hash
        )
    )
    manager = SpyreSharedOffloadingManager(
        metadata_name="shared-meta",
        families=(SharedPoolFamily("alpha", 2),),
        max_components=2,
        num_blocks=2,
        cache_policy="lru",
        cache_policy_module_path=None,
        enable_events=False,
        store_threshold=0,
        max_tracker_size=64_000,
        runtime_loader=lambda: runtime,
    )
    manager._runtime = runtime
    manager._directory = directory
    manager._anchors = (anchor,)
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
