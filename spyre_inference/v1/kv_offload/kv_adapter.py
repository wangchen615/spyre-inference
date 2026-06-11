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

"""Bridge the Spyre paged-list KV layout to the offloading handler.

vLLM's ``OffloadingConnectorWorker.register_kv_caches`` assumes each layer's KV
cache is a single ``torch.Tensor`` whose storage can be reinterpreted as a flat
``(num_blocks, page_size_bytes)`` int8 buffer via
``untyped_storage().set_().view(...)``. Neither assumption holds on Spyre:

- ``TorchSpyreModelRunner.initialize_kv_cache_tensors`` binds each layer to a
  ``SpyrePagedKVCache(k_pages, v_pages)`` — two Python *lists* of per-block
  tensors of shape ``[num_kv_heads, block_size, head_size]`` (fp16), not one
  tensor.
- storage/pointer reinterpretation is not viable on Spyre device tensors, and
  on-device slicing corrupts memory (see ``spyre_attn.py``); the only supported
  device<->host path is a whole-tensor ``.copy_()`` / ``.to("cpu")``.

This module provides a plugin-side adapter (no vLLM patch — see the RFC tracking
doc) that walks the bound ``kv_caches`` dict and produces, per layer, a flat
``KVCacheLayerView`` exposing each block as its own device tensor plus the matching
host staging tensor. The handler copies block-by-block through ``SpyreKvDmaCopier``
against these views, never touching tensor storage.

The view treats K and V as two separate "sub-caches" of the same layer because the
Spyre cache stores them in independent page lists. Each block id therefore maps to
two device page tensors (k and v); the host side mirrors that with two host page
lists. Page sizes match exactly (same shape/dtype), so no int8 re-view is needed.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class KVCacheLayerView:
    """Per-layer, per-block handle onto a Spyre paged KV cache and its host mirror.

    ``device_k_pages`` / ``device_v_pages`` are the live lists owned by the model
    runner's ``SpyrePagedKVCache`` (we hold references, we do not copy them).
    ``host_k_pages`` / ``host_v_pages`` are host staging tensors this adapter
    allocates, one per device block, with identical shape and dtype. Block id ``b``
    addresses ``device_*_pages[b]`` and ``host_*_pages[b]``.
    """

    layer_name: str
    device_k_pages: list[torch.Tensor]
    device_v_pages: list[torch.Tensor]
    host_k_pages: list[torch.Tensor] = field(default_factory=list)
    host_v_pages: list[torch.Tensor] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return len(self.device_k_pages)

    @property
    def page_shape(self) -> torch.Size:
        return self.device_k_pages[0].shape

    @property
    def page_dtype(self) -> torch.dtype:
        return self.device_k_pages[0].dtype

    def page_size_bytes(self) -> int:
        page = self.device_k_pages[0]
        return page.numel() * page.element_size()


def _is_paged_kv_cache(value: object) -> bool:
    """Duck-type a SpyrePagedKVCache without importing the attention backend.

    Importing ``spyre_attn`` here would pull the attention stack into the offload
    import path; the cache is a 2-tuple of (k_pages, v_pages), so we match on shape.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], list)
        and isinstance(value[1], list)
        and len(value[0]) == len(value[1])
        and (len(value[0]) == 0 or isinstance(value[0][0], torch.Tensor))
    )


def build_layer_views(
    kv_caches: dict[str, object],
    num_cpu_blocks: int,
) -> list[KVCacheLayerView]:
    """Build one KVCacheLayerView per unique layer cache, allocating host pages.

    Shared layers (multiple names pointing at the same ``SpyrePagedKVCache``
    instance) are de-duplicated by object identity so host pages are allocated
    once per physical cache.

    Args:
        kv_caches: the bound ``{layer_name: SpyrePagedKVCache}`` dict from the
            model runner. Entries that are not paged caches are skipped.
        num_cpu_blocks: number of host blocks to stage per layer. Host pages are
            allocated up to ``min(num_cpu_blocks, num_device_blocks)`` — the host
            tier never needs more blocks than exist on device for a single layer.

    Returns:
        A list of views, one per unique physical cache, in first-seen order.
    """
    views: list[KVCacheLayerView] = []
    seen_ids: dict[int, KVCacheLayerView] = {}

    for layer_name, cache in kv_caches.items():
        if not _is_paged_kv_cache(cache):
            logger.debug(
                "kv_adapter: skipping layer %r (not a paged KV cache)", layer_name
            )
            continue

        cache_id = id(cache)
        if cache_id in seen_ids:
            continue

        k_pages, v_pages = cache  # type: ignore[misc]
        if len(k_pages) == 0:
            logger.debug("kv_adapter: skipping layer %r (empty cache)", layer_name)
            continue

        view = KVCacheLayerView(
            layer_name=layer_name,
            device_k_pages=k_pages,
            device_v_pages=v_pages,
        )

        n_host = min(num_cpu_blocks, view.num_blocks)
        view.host_k_pages = _alloc_host_pages(view, n_host)
        view.host_v_pages = _alloc_host_pages(view, n_host)

        seen_ids[cache_id] = view
        views.append(view)

    return views


def _alloc_host_pages(view: KVCacheLayerView, count: int) -> list[torch.Tensor]:
    """Allocate ``count`` host staging pages matching a layer's page shape/dtype.

    Uses plain ``torch.empty`` on CPU. There is no Spyre equivalent of CUDA host
    pinning (``cudaHostRegister``), so pages are unpinned — consistent with the
    rest of the plugin's device<->host transfers.
    """
    return [
        torch.empty(view.page_shape, dtype=view.page_dtype, device="cpu")
        for _ in range(count)
    ]


def iter_block_pages(
    views: Iterable[KVCacheLayerView],
) -> Iterable[tuple[KVCacheLayerView, int]]:
    """Yield ``(view, block_id)`` for every device block across all layer views.

    A small helper for callers that want to walk every (layer, block) pair, e.g.
    tests that round-trip the whole cache.
    """
    for view in views:
        for block_id in range(view.num_blocks):
            yield view, block_id
