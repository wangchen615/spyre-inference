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

"""`SpyreOffloadingSpec` sizing, manager reuse, and misconfiguration guards.

No device is needed: the spec is scheduler-side bookkeeping plus a worker factory,
and `get_worker` is only reached here through its refusal paths.

Several of these pin *misconfigurations that would otherwise be silent*. A
`num_blocks` of 0 means every lookup misses and nothing is ever offloaded -- a
benchmark showing no benefit rather than an error. `blocks_per_chunk != 1` means a
host block spans several device blocks, which a one-page-per-slot copy cannot
represent. Both raise.
"""

from __future__ import annotations

import pytest
from vllm.v1.kv_offload.base import OffloadingSpec
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec

# One K page of the geometry this work is benchmarked at (128 tokens x 8 heads x
# 128 dims, fp16); K+V is twice that.
KV_BYTES_PER_BLOCK = 2 * 262144


def _config(
    *,
    extra: dict | None = None,
    blocks_per_chunk: int = 1,
    world_size: int = 1,
    rank: int = 0,
    worker_kv_bytes_per_block: int = KV_BYTES_PER_BLOCK,
    enable_kv_cache_events: bool = False,
) -> OffloadingConfig:
    extra_config = {"cpu_bytes_to_use": 32 * KV_BYTES_PER_BLOCK}
    if extra is not None:
        extra_config = {**extra_config, **extra}
        # a None value means "remove this key", so a test can drop a required one
        extra_config = {k: v for k, v in extra_config.items() if v is not None}
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(tokens_per_block=128, layer_names=("layer.0",)),
        ),
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config=extra_config,
        engine_id="eng0",
        model=OffloadingModelConfig(name="micro-g3.3-8b", dtype="float16"),
        cache=OffloadingCacheConfig(
            tokens_per_hash=128, blocks_per_chunk=blocks_per_chunk
        ),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )


def test_subclasses_offloading_spec_not_the_cpu_spec():
    """Must derive from the base spec, not upstream's CPU one.

    `CPUOffloadingSpec` builds a worker around pinned host tensors and
    accelerator device indexing, neither of which exists on Spyre. Inheriting it
    to reuse `get_manager` would drag that in. This pins the choice so an
    upstream refactor that makes CPUOffloadingSpec look reusable does not quietly
    become the base class.
    """
    assert issubclass(SpyreOffloadingSpec, OffloadingSpec)
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    assert not issubclass(SpyreOffloadingSpec, CPUOffloadingSpec)


def test_num_blocks_divides_budget_by_kv_bytes_per_block():
    spec = SpyreOffloadingSpec(_config())
    assert spec.num_blocks == 32


def test_num_blocks_accounts_for_world_size():
    """The budget is shared across workers, so more workers means fewer blocks."""
    spec = SpyreOffloadingSpec(_config(world_size=4))
    assert spec.num_blocks == 8


def test_zero_blocks_raises_rather_than_offloading_nothing():
    """A budget too small for one block must fail loudly.

    Left to itself this configuration offloads nothing and every lookup misses,
    so a benchmark would show no benefit and no error.
    """
    with pytest.raises(ValueError, match="too small"):
        SpyreOffloadingSpec(_config(extra={"cpu_bytes_to_use": KV_BYTES_PER_BLOCK - 1}))


def test_missing_cpu_bytes_to_use_raises():
    with pytest.raises(ValueError, match="cpu_bytes_to_use"):
        SpyreOffloadingSpec(_config(extra={"cpu_bytes_to_use": None}))


@pytest.mark.parametrize("blocks_per_chunk", [2, 4])
def test_chunking_is_rejected(blocks_per_chunk):
    """Chunking has no representation in a one-page-per-slot copy.

    Asserted on the *derived* value, which upstream computes from either the
    `blocks_per_chunk` or the `block_size` extra-config key, so this covers both
    spellings.
    """
    with pytest.raises(ValueError, match="blocks_per_chunk"):
        SpyreOffloadingSpec(_config(blocks_per_chunk=blocks_per_chunk))


def test_get_manager_is_the_upstream_cpu_manager_and_memoized():
    """The scheduler side needs no Spyre code, so upstream's manager is reused.

    Memoization matters: the manager holds the block-id bookkeeping, and a second
    instance would hand out ids that disagree with the first.
    """
    spec = SpyreOffloadingSpec(_config())
    manager = spec.get_manager()
    assert isinstance(manager, CPUOffloadingManager)
    assert spec.get_manager() is manager


def test_pool_prefix_is_unique_per_engine_and_rank():
    """Two ranks on one host must not collide on a POSIX shared-memory name."""
    prefixes = {
        SpyreOffloadingSpec(_config(rank=r, world_size=2))._pool_prefix
        for r in (0, 1)
    }
    assert len(prefixes) == 2
    assert all("eng0" in p for p in prefixes)


def test_get_worker_before_binding_raises_naming_the_method():
    """The canonical caches cannot drive DMA, so binding is not optional.

    The error has to name `bind_physical_caches`: reaching this point means the
    connector was not the Spyre one, and the message is the only pointer to why.
    """
    spec = SpyreOffloadingSpec(_config())
    with pytest.raises(RuntimeError, match="bind_physical_caches"):
        spec.get_worker(kv_caches=None)


def test_offload_prompt_only_default_is_upstreams():
    """Not overridden here; pinned so a change is deliberate."""
    assert SpyreOffloadingSpec(_config()).offload_prompt_only is True
    spec = SpyreOffloadingSpec(_config(extra={"offload_prompt_only": False}))
    assert spec.offload_prompt_only is False


def test_resolvable_through_the_upstream_spec_factory():
    """`vllm serve` reaches this class by module path, not by registry entry.

    Goes through `create_spec`, which is the actual serve entry point, so this
    covers both the out-of-tree import and construction from the same
    `kv_connector_extra_config` dict a real config file would carry. If this
    stops working the connector cannot be configured out-of-tree at all.
    """
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    config = _config(
        extra={
            "spec_name": "SpyreOffloadingSpec",
            "spec_module_path": "spyre_inference.v1.kv_offload.spec",
        }
    )
    assert (
        OffloadingSpecFactory.get_spec_cls(config.extra_config) is SpyreOffloadingSpec
    )
    spec = OffloadingSpecFactory.create_spec(config)
    assert isinstance(spec, SpyreOffloadingSpec)
    assert spec.num_blocks == 32


def test_declares_every_metric_the_reused_manager_emits() -> None:
    """The spec must declare what `CPUOffloadingManager` reports.

    Prometheus builds its metric set from the spec class but the values come from
    the manager, and upstream asserts the key was declared
    (`offloading/metrics.py`, `assert key in self._offloading_metric_defs`). With
    the base class's empty default this crashed the output handler on the *first
    served request* and shut the server down -- invisible to every other test
    here, because none of them touch the Prometheus path.

    Rather than hand-listing the keys, read them out of the manager's own source:
    a metric added to the manager upstream then fails here instead of at serve
    time. `STORES_SKIPPED` is conditional on `store_threshold >= 2` in both the
    manager and the definitions, so it is checked under a config that enables it.
    """
    import inspect
    import re

    from vllm.v1.kv_offload.cpu import manager as manager_mod
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingMetrics

    # The manager names members (`CPUOffloadingMetrics.STORES_SKIPPED`) but the
    # definitions dict is keyed by the metric-name *string* those members hold
    # ("vllm:kv_offload_stores_skipped"), which is also what `observe()` looks up.
    # Resolve through the class so the two sides are compared in one namespace.
    members = set(
        re.findall(r"CPUOffloadingMetrics\.([A-Z_]+)", inspect.getsource(manager_mod))
    )
    assert members, "found no CPUOffloadingMetrics references; upstream changed shape"
    emitted = {getattr(CPUOffloadingMetrics, name) for name in members}

    config = _config(extra={"store_threshold": 2})
    declared = set(SpyreOffloadingSpec.build_metric_definitions(config.extra_config))

    assert emitted <= declared, (
        f"manager emits metrics the spec never declares: {sorted(emitted - declared)}"
    )


def test_stores_skipped_declared_only_when_filtering_is_on() -> None:
    """Mirror the manager: it only emits STORES_SKIPPED when the threshold is set."""
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingMetrics

    off = SpyreOffloadingSpec.build_metric_definitions(_config().extra_config)
    on = SpyreOffloadingSpec.build_metric_definitions(
        _config(extra={"store_threshold": 2}).extra_config
    )
    assert set(on) - set(off) == {CPUOffloadingMetrics.STORES_SKIPPED}
