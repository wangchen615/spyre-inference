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

"""Configuration, compatibility, and lazy loading for shared KV offload."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from spyre_inference.v1.kv_offload.connector import (
    TOKEN_MAJOR,
    SpyrePhysicalCaches,
)
from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec
from spyre_inference.v1.worker.spyre_kv_offload import PageSignature

KV_BYTES_PER_BLOCK = 2 * 262144


def _shared_spec_module():
    from spyre_inference.v1.kv_offload import shared_spec

    return shared_spec


def _config(
    *,
    extra: dict | None = None,
    num_blocks: int = 8,
    world_size: int = 1,
    group_count: int = 1,
    dtype: str = "float16",
) -> OffloadingConfig:
    extra_config = {
        "cpu_bytes_to_use": num_blocks * KV_BYTES_PER_BLOCK * world_size,
        "shared_metadata_name": "run-42",
        "shared_pool_families": ["pool-a", "pool-b", "pool-c"],
    }
    if extra is not None:
        extra_config.update(extra)
    groups = tuple(
        OffloadingGroupConfig(
            tokens_per_block=128,
            layer_names=(f"layer.{index}",),
        )
        for index in range(group_count)
    )
    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=KV_BYTES_PER_BLOCK,
        enable_kv_cache_events=False,
        extra_config=extra_config,
        engine_id="eng0",
        model=OffloadingModelConfig(name="micro-g3.3-8b", dtype=dtype),
        cache=OffloadingCacheConfig(tokens_per_hash=128, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
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


def _vllm_config(
    *,
    backend: str = "uni",
    model: str = "model-a",
    revision: str | None = "rev-a",
    hash_algorithm: str = "sha256",
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend=backend),
        model_config=SimpleNamespace(model=model, revision=revision),
        cache_config=SimpleNamespace(prefix_caching_hash_algo=hash_algorithm),
    )


def _physical(layout_kind: str = TOKEN_MAJOR) -> SpyrePhysicalCaches:
    return SpyrePhysicalCaches(
        caches=cast(
            tuple[tuple[torch.Tensor, torch.Tensor], ...],
            ((object(), object()),),
        ),
        layout_kinds=(layout_kind,),
        tensor_idx_to_cache={0: 0, 1: 0},
        num_blocks=4,
    )


def _signature(
    *,
    layout_kind: str = TOKEN_MAJOR,
    device_dtype: str = "torch.float16",
    block_size: int = 128,
    local_kv_heads: int = 8,
    head_size: int = 128,
    page_bytes: int = 262144,
    layout_version: int = 1,
) -> PageSignature:
    return PageSignature(
        layout_kind=layout_kind,
        device_dtype=device_dtype,
        block_size=block_size,
        local_kv_heads=local_kv_heads,
        head_size=head_size,
        page_bytes=page_bytes,
        layout_version=layout_version,
    )


def test_shared_configuration_splits_aggregate_capacity_in_order():
    spec = _shared_spec_module().SpyreSharedOffloadingSpec(_config())

    assert [(family.name, family.slot_count) for family in spec.families] == [
        ("pool-a", 3),
        ("pool-b", 3),
        ("pool-c", 2),
    ]
    assert spec.metadata_name == "run-42"
    assert spec.max_components == 2


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"shared_metadata_name": " "}, "shared_metadata_name"),
        ({"shared_pool_families": []}, "family"),
        ({"shared_pool_families": ["same", "same"]}, "unique"),
    ],
)
def test_rejects_invalid_shared_configuration(extra, match):
    with pytest.raises(ValueError, match=match):
        _shared_spec_module().SpyreSharedOffloadingSpec(_config(extra=extra))


def test_rejects_fewer_slots_than_pool_families():
    with pytest.raises(ValueError, match="fewer slots"):
        _shared_spec_module().SpyreSharedOffloadingSpec(_config(num_blocks=2))


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (_config(world_size=2), "world_size"),
        (_config(group_count=2), "one KV cache group"),
        (_config(dtype="bfloat16"), "float16"),
    ],
)
def test_rejects_unsupported_shared_execution_shapes(config, match):
    with pytest.raises(ValueError, match=match):
        _shared_spec_module().SpyreSharedOffloadingSpec(config)


def test_rejects_non_uni_executor_after_vllm_resolution():
    spec = _shared_spec_module().SpyreSharedOffloadingSpec(_config())

    with pytest.raises(ValueError, match="uni"):
        spec.bind_vllm_config(_vllm_config(backend="mp"))


def test_compatibility_payload_covers_every_interpretation_field(
    monkeypatch,
):
    module = _shared_spec_module()
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(module, "page_signature", lambda *_: _signature())
    spec = module.SpyreSharedOffloadingSpec(_config())
    spec.bind_vllm_config(_vllm_config())

    assert spec.compatibility_payload(_physical()) == {
        "format": 1,
        "model": "model-a",
        "revision": "rev-a",
        "hash_algorithm": "sha256",
        "hash_seed": "0",
        "tokens_per_hash": 128,
        "tokens_per_block": (128,),
        "dtype": "float16",
        "tp_size": 1,
        "components": (
            {
                "cache_index": 0,
                "role": "k",
                "layout_kind": "token-major",
                "layout_version": 1,
                "block_size": 128,
                "local_kv_heads": 8,
                "head_size": 128,
                "page_bytes": 262144,
            },
            {
                "cache_index": 0,
                "role": "v",
                "layout_kind": "token-major",
                "layout_version": 1,
                "block_size": 128,
                "local_kv_heads": 8,
                "head_size": 128,
                "page_bytes": 262144,
            },
        ),
    }


def test_every_compatibility_field_changes_the_digest():
    module = _shared_spec_module()
    base: dict[str, Any] = {
        "format": 1,
        "model": "model-a",
        "revision": "rev-a",
        "hash_algorithm": "sha256",
        "hash_seed": "0",
        "tokens_per_hash": 128,
        "tokens_per_block": (128,),
        "dtype": "float16",
        "tp_size": 1,
        "components": (
            {
                "cache_index": 0,
                "role": "k",
                "layout_kind": "token-major",
                "layout_version": 1,
                "block_size": 128,
                "local_kv_heads": 8,
                "head_size": 128,
                "page_bytes": 262144,
            },
        ),
    }
    expected = module.compatibility_digest(base)
    mutations = {
        "format": 2,
        "model": "model-b",
        "revision": "rev-b",
        "hash_algorithm": "sha256_cbor_64bit",
        "hash_seed": "1",
        "tokens_per_hash": 64,
        "tokens_per_block": (64,),
        "dtype": "bfloat16",
        "tp_size": 2,
    }

    for field, value in mutations.items():
        changed = copy.deepcopy(base)
        changed[field] = value
        assert module.compatibility_digest(changed) != expected, field

    for field, value in {
        "cache_index": 1,
        "role": "v",
        "layout_kind": "head-major",
        "layout_version": 2,
        "block_size": 64,
        "local_kv_heads": 4,
        "head_size": 64,
        "page_bytes": 131072,
    }.items():
        changed = copy.deepcopy(base)
        changed["components"][0][field] = value
        assert module.compatibility_digest(changed) != expected, field


def test_compatibility_digest_is_stable_across_processes():
    module = _shared_spec_module()
    payload = {
        "format": 1,
        "model": "model-a",
        "revision": "rev-a",
        "hash_algorithm": "sha256",
        "hash_seed": "0",
        "tokens_per_hash": 128,
        "tokens_per_block": (128,),
        "dtype": "float16",
        "tp_size": 1,
        "components": (),
    }
    expected = module.compatibility_digest(payload).hex()
    assert expected == "a3567edf4796a801e7681081e5a6e0d299be4ca03fb5445b9af681f28eaf9883"
    code = (
        "from spyre_inference.v1.kv_offload.shared_spec import "
        "compatibility_digest; "
        f"print(compatibility_digest({payload!r}).hex())"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )

    assert completed.stdout.strip().splitlines()[-1] == expected


def test_compatibility_requires_an_explicit_python_hash_seed(monkeypatch):
    module = _shared_spec_module()
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(module, "page_signature", lambda *_: _signature())
    spec = module.SpyreSharedOffloadingSpec(_config())
    spec.bind_vllm_config(_vllm_config())

    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        spec.compatibility_payload(_physical())


def test_shared_factories_receive_validated_configuration(monkeypatch):
    module = _shared_spec_module()
    from spyre_inference.v1.kv_offload import shared_manager, shared_worker

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(module, "page_signature", lambda *_: _signature())
    manager_kwargs = {}
    worker_kwargs = {}

    class RecordingManager:
        def __init__(self, **kwargs):
            manager_kwargs.update(kwargs)

    class RecordingWorker:
        _bytes_per_block = KV_BYTES_PER_BLOCK

        def __init__(self, **kwargs):
            worker_kwargs.update(kwargs)

    monkeypatch.setattr(shared_manager, "SpyreSharedOffloadingManager", RecordingManager)
    monkeypatch.setattr(shared_worker, "SpyreSharedOffloadingWorker", RecordingWorker)
    spec = module.SpyreSharedOffloadingSpec(_config())
    spec.bind_vllm_config(_vllm_config())
    physical = _physical()
    spec.bind_physical_caches(physical)

    assert spec.get_manager().__class__ is RecordingManager
    spec._create_worker(physical)

    assert manager_kwargs["metadata_name"] == "run-42"
    assert manager_kwargs["families"] == spec.families
    assert manager_kwargs["num_blocks"] == 8
    assert worker_kwargs["physical"] is physical
    assert worker_kwargs["families"] == spec.families
    assert len(worker_kwargs["compatibility_digest"]) == 32


def test_shared_spec_factory_registration_is_lazy():
    code = """
import sys
import spyre_inference
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

module = "spyre_inference.v1.kv_offload.shared_spec"
assert module not in sys.modules
cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "SpyreSharedOffloadingSpec"})
assert cls.__name__ == "SpyreSharedOffloadingSpec"
assert module in sys.modules
print("lazy-ok")
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )

    assert completed.stdout.strip().splitlines()[-1] == "lazy-ok"


def test_missing_m2_runtime_does_not_break_plugin_or_m1():
    code = """
import sys
import types

torch_spyre = types.ModuleType("torch_spyre")
torch_spyre.__path__ = []
torch_spyre._C = types.ModuleType("torch_spyre._C")
sys.modules["torch_spyre"] = torch_spyre
sys.modules["torch_spyre._C"] = torch_spyre._C

import spyre_inference
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

m1 = OffloadingSpecFactory.get_spec_cls({
    "spec_name": "SpyreOffloadingSpec",
    "spec_module_path": "spyre_inference.v1.kv_offload.spec",
})
assert m1.__name__ == "SpyreOffloadingSpec"
m2 = OffloadingSpecFactory.get_spec_cls({"spec_name": "SpyreSharedOffloadingSpec"})

from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

config = OffloadingConfig(
    groups=(OffloadingGroupConfig(tokens_per_block=128, layer_names=("layer.0",)),),
    worker_kv_bytes_per_block=524288,
    enable_kv_cache_events=False,
    extra_config={
        "cpu_bytes_to_use": 4194304,
        "shared_metadata_name": "run-42",
        "shared_pool_families": ["pool-a"],
    },
    engine_id="eng0",
    model=OffloadingModelConfig(name="model-a", dtype="float16"),
    cache=OffloadingCacheConfig(tokens_per_hash=128, blocks_per_chunk=1),
    parallel=OffloadingParallelConfig(
        rank=0,
        world_size=1,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        data_parallel_index=0,
        data_parallel_size=1,
        data_parallel_rank_local=None,
        is_parallelism_agnostic=False,
    ),
)
try:
    m2(config).get_manager().directory
except RuntimeError as error:
    message = str(error)
    assert "torch-spyre M2" in message
    assert "SharedMetadata" in message
else:
    raise AssertionError("missing M2 runtime unexpectedly succeeded")
print("missing-runtime-ok")
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONHASHSEED": "0",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
    )

    assert completed.stdout.strip().splitlines()[-1] == "missing-runtime-ok"


def test_m1_factory_still_builds_the_upstream_manager():
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    config = _config(
        extra={
            "spec_name": "SpyreOffloadingSpec",
            "spec_module_path": "spyre_inference.v1.kv_offload.spec",
        }
    )
    spec = OffloadingSpecFactory.create_spec(config)

    assert type(spec) is SpyreOffloadingSpec
    assert type(spec.get_manager()) is CPUOffloadingManager
