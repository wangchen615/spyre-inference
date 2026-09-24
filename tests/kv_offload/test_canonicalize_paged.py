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

"""Canonicalization of Spyre paged KV caches into vLLM's canonical form.

Runs on CPU tensors: `spyre_paged_to_canonical` only reshapes metadata and
records identities, so no device is needed. The DMA path that *does* need a
device is covered by `test_worker_hw.py`.

What matters here is the *pairing*: the worker's slot arithmetic is an identity
mapping only because K and V of one physical cache map to the same cache index,
and because layers sharing one allocation collapse to one cache rather than being
offloaded twice.
"""

from __future__ import annotations

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from spyre_inference.v1.kv_offload.connector import (
    HEAD_MAJOR,
    TOKEN_MAJOR,
    spyre_paged_to_canonical,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
NUM_KV_HEADS = 2
HEAD_SIZE = 16
DTYPE = torch.float16

# K and V each carry half of upstream's page, which spans both.
HALF_PAGE = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2


def _pages() -> torch.Tensor:
    return torch.zeros(
        (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE
    )


def _cache() -> tuple[torch.Tensor, torch.Tensor]:
    """A SpyrePagedKVCache-shaped 2-tuple; the connector duck-types it."""
    return (_pages(), _pages())


def _spec(head_size: int = HEAD_SIZE) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=head_size,
        dtype=DTYPE,
    )


def _kv_cache_config(layer_names: list[str], spec=None) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=layer_names, kv_cache_spec=spec or _spec()
            )
        ],
    )


def _canonicalize(kv_caches: dict, layouts: dict | None = None):
    names = list(kv_caches)
    return spyre_paged_to_canonical(
        kv_caches,
        _kv_cache_config(names),
        layouts or dict.fromkeys(names, TOKEN_MAJOR),
    )


def test_k_and_v_map_to_the_same_cache_index():
    """The worker's identity slot mapping depends on this.

    Each cache contributes two canonical tensors (K then V) that must resolve to
    one cache index, because there is one `SharedHostPool` per *cache* and the
    offloader pairs K/V internally as slots 2h and 2h+1.
    """
    kv_caches = {f"layer.{i}": _cache() for i in range(3)}
    canonical, physical = _canonicalize(kv_caches)

    assert len(canonical.tensors) == 6
    assert len(physical.caches) == 3
    for c in range(3):
        assert physical.tensor_idx_to_cache[2 * c] == c
        assert physical.tensor_idx_to_cache[2 * c + 1] == c


def test_layers_sharing_one_allocation_collapse_to_one_cache():
    """Aliased layers must not offload the same page twice.

    The model runner binds one `SpyrePagedKVCache` to every layer in `shared_by`,
    so without dedup on `id(cache)` an aliased layer would get its own pool and
    its own DMA for bytes already moved.
    """
    shared = _cache()
    canonical, physical = _canonicalize(
        {"layer.0": shared, "layer.1": shared, "layer.2": _cache()}
    )

    assert len(physical.caches) == 2
    assert len(canonical.tensors) == 4
    # The two aliased layers point at the same pair of canonical tensors.
    refs = canonical.group_data_refs[0]
    assert [r.tensor_idx for r in refs] == [0, 1, 0, 1, 2, 3]


def test_canonical_tensors_are_flat_views_sharing_storage():
    """Metadata-only: the view must not copy, or vLLM would bookkeep a shadow."""
    k, v = _cache()
    canonical, _ = _canonicalize({"layer.0": (k, v)})

    flat_k = canonical.tensors[0].tensor
    assert flat_k.shape == (NUM_BLOCKS, BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE)
    assert flat_k.data_ptr() == k.data_ptr()
    assert canonical.tensors[0].page_size_bytes == HALF_PAGE


def test_head_major_layout_is_recorded():
    """The layout is not recoverable from the tensor, so it must be carried.

    Both layouts fold logical dim 1 into device dim 0, so shape alone cannot tell
    them apart; guessing would transpose block_size against num_kv_heads.
    """
    _, physical = _canonicalize(
        {"layer.0": _cache()}, layouts={"layer.0": HEAD_MAJOR}
    )
    assert physical.layout_kinds == (HEAD_MAJOR,)


def test_layout_disagreement_within_one_cache_raises():
    """Two layers on one allocation cannot have been allocated two ways.

    Disagreement means the impl lookup is wrong. Picking either value would
    transpose the page geometry for one of them, so refuse.
    """
    shared = _cache()
    with pytest.raises(ValueError, match="disagree on layout"):
        _canonicalize(
            {"layer.0": shared, "layer.1": shared},
            layouts={"layer.0": TOKEN_MAJOR, "layer.1": HEAD_MAJOR},
        )


def test_unresolved_layout_raises():
    with pytest.raises(ValueError, match="unresolved layout kind"):
        _canonicalize({"layer.0": _cache()}, layouts={"layer.0": "sideways"})


def test_page_size_disagreement_with_the_spec_raises():
    """A spec/device mismatch would size every pool wrong."""
    kv_caches = {"layer.0": _cache()}
    with pytest.raises(ValueError, match="device page is"):
        spyre_paged_to_canonical(
            kv_caches,
            _kv_cache_config(["layer.0"], spec=_spec(head_size=HEAD_SIZE * 2)),
            {"layer.0": TOKEN_MAJOR},
        )


def test_non_tuple_cache_is_rejected():
    """Upstream asserts `isinstance(cache, torch.Tensor)`; we assert the 2-tuple."""
    with pytest.raises(TypeError, match="2-tuple"):
        _canonicalize({"layer.0": _pages()})


def test_mismatched_k_and_v_pages_are_rejected():
    with pytest.raises(ValueError, match="k/v pages disagree"):
        _canonicalize(
            {"layer.0": (_pages(), torch.zeros((NUM_BLOCKS, 1), dtype=DTYPE))}
        )


def test_num_blocks_disagreement_across_layers_raises():
    """One block id must mean one page in every layer."""
    small = (
        torch.zeros((2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE),
        torch.zeros((2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE), dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="num_blocks"):
        _canonicalize({"layer.0": _cache(), "layer.1": small})


def test_num_blocks_is_carried_for_pool_sizing():
    _, physical = _canonicalize({"layer.0": _cache()})
    assert physical.num_blocks == NUM_BLOCKS
