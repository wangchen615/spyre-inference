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

"""SpyreOffloadingSpec registration and configuration.

Runs on CPU: nothing here allocates on the Spyre device, so these tests act as a
cheap gate on the upstream offloading API before spending card time.
"""

import pytest

from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

import spyre_inference
from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec

KV_BYTES_PER_BLOCK = 1_048_576


def _make_config(
    cpu_bytes_to_use: int = 2_000_000_000,
    worker_kv_bytes_per_block: int = KV_BYTES_PER_BLOCK,
    world_size: int = 1,
    blocks_per_chunk: int = 1,
    extra: dict | None = None,
) -> OffloadingConfig:
    extra_config = {"cpu_bytes_to_use": cpu_bytes_to_use} if cpu_bytes_to_use else {}
    extra_config.update(extra or {})
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(tokens_per_block=128, layer_names=("layer0",)),),
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config=extra_config,
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=128, blocks_per_chunk=blocks_per_chunk),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=True,
        ),
    )


@pytest.mark.spyre
def test_spec_is_registered_with_the_factory():
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    spyre_inference.register_offloading_specs()
    assert "SpyreOffloadingSpec" in OffloadingSpecFactory._registry

    # The factory stores a lazy loader; resolving it must yield our class.
    spec_cls = OffloadingSpecFactory._registry["SpyreOffloadingSpec"]()
    assert spec_cls is SpyreOffloadingSpec


@pytest.mark.spyre
def test_registration_is_idempotent():
    """`register_spec` raises on a duplicate name, so re-registering must be a no-op."""
    spyre_inference.register_offloading_specs()
    spyre_inference.register_offloading_specs()


@pytest.mark.spyre
def test_num_blocks_divides_cpu_budget_by_block_size():
    spec = SpyreOffloadingSpec(_make_config())
    assert spec.num_blocks == 2_000_000_000 // KV_BYTES_PER_BLOCK


@pytest.mark.spyre
def test_num_blocks_scales_with_world_size():
    """`worker_kv_bytes_per_block` is a per-worker share, so an offloaded block
    costs `world_size` times as much host memory."""
    spec = SpyreOffloadingSpec(_make_config(world_size=2))
    assert spec.num_blocks == 2_000_000_000 // (KV_BYTES_PER_BLOCK * 2)


@pytest.mark.spyre
def test_num_blocks_is_zero_when_kv_size_unknown():
    """A profiling run reports no KV bytes; must not divide by zero."""
    spec = SpyreOffloadingSpec(_make_config(worker_kv_bytes_per_block=0))
    assert spec.num_blocks == 0


@pytest.mark.spyre
def test_missing_cpu_bytes_to_use_is_rejected():
    with pytest.raises(ValueError, match="cpu_bytes_to_use"):
        SpyreOffloadingSpec(_make_config(cpu_bytes_to_use=0))


@pytest.mark.spyre
def test_multi_block_chunks_are_rejected():
    """Only one offloaded block per device block is supported."""
    with pytest.raises(ValueError, match="blocks_per_chunk"):
        SpyreOffloadingSpec(_make_config(blocks_per_chunk=2))


@pytest.mark.spyre
def test_get_manager_returns_a_cached_cpu_manager():
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    spec = SpyreOffloadingSpec(_make_config())
    manager = spec.get_manager()
    assert isinstance(manager, CPUOffloadingManager)
    assert spec.get_manager() is manager


@pytest.mark.spyre
def test_eviction_policy_is_configurable():
    spec = SpyreOffloadingSpec(_make_config(extra={"eviction_policy": "lru"}))
    assert spec.eviction_policy == "lru"


@pytest.mark.spyre
def test_get_worker_before_priming_raises():
    """The model-runner hook must prime the raw paged caches first."""
    spec = SpyreOffloadingSpec(_make_config())
    with pytest.raises(RuntimeError, match="prime_kv_caches"):
        spec.get_worker(None)
