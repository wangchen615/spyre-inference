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

"""Spyre variant of vLLM's OffloadingConnector.

Upstream `OffloadingConnectorWorker.register_kv_caches` canonicalizes each
layer's cache by asserting it is one `torch.Tensor` and reinterpreting its
storage (`untyped_storage()` + `.set_()` + `as_strided`). Spyre fails both: a
layer is bound to a `SpyrePagedKVCache(k_pages, v_pages)` 2-tuple, and storage
reinterpretation is unsupported on Spyre device tensors.

Canonicalization runs before `spec.get_worker()`, so no spec-level hook can fix
it -- hence this subclass, which overrides exactly that one method.

The canonical tensors it emits are **metadata only**. They are flattened
`(num_blocks, -1)` views, and `copy_kv_page_raw` rejects those outright
("expected a rank-4 KV cache [N,X,Y,D], got rank 2"): it needs the original
rank-4 allocation from `allocate_pages` so it can validate the KV-page contract
and derive the physical byte range itself. So canonicalization returns *two*
halves -- the canonical view for vLLM's block bookkeeping, and
`SpyrePhysicalCaches` for the DMA path -- and the worker is handed the latter
via `SpyreOffloadingSpec.bind_physical_caches`.
"""

import dataclasses
from collections.abc import Iterable, Mapping

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)

from spyre_inference.v1.kv_offload.upstream_compat import OffloadingConnectorWorker

logger = init_logger(__name__)

TOKEN_MAJOR = "token-major"
HEAD_MAJOR = "head-major"


@dataclasses.dataclass(frozen=True)
class SpyrePhysicalCaches:
    """The original rank-4 caches, deduped, as the DMA path needs them.

    `copy_kv_page_raw` takes a whole `k_pages`/`v_pages` allocation plus a
    `block_id`; it cannot take the flattened canonical view. This carries those
    originals alongside the layout kind each was allocated with, which is not
    recoverable from the tensor (see `_layout_kind_by_layer`).
    """

    # Deduped by id(); the index into this tuple is the "cache index" used to
    # name one SharedHostPool per cache.
    caches: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    # Parallel to `caches`: TOKEN_MAJOR or HEAD_MAJOR.
    layout_kinds: tuple[str, ...]
    # Canonical tensor index -> cache index. K and V of one cache both map to
    # the same cache index, which is what makes the worker's slot arithmetic a
    # plain identity (see worker.py).
    tensor_idx_to_cache: Mapping[int, int]
    num_blocks: int

    def __post_init__(self) -> None:
        if len(self.caches) != len(self.layout_kinds):
            raise ValueError(
                f"caches/layout_kinds length mismatch: "
                f"{len(self.caches)} vs {len(self.layout_kinds)}"
            )


def spyre_paged_to_canonical(
    kv_caches: Mapping[str, object],
    kv_cache_config: KVCacheConfig,
    layout_kind_by_layer: Mapping[str, str],
) -> tuple[CanonicalKVCaches, SpyrePhysicalCaches]:
    """Canonicalize Spyre paged KV caches without touching tensor storage.

    Each layer is bound to a `SpyrePagedKVCache(k_pages, v_pages)`, two dense
    `[num_blocks, block_size, num_kv_heads, head_size]` device tensors. Upstream
    wants `(num_blocks, page_size_bytes)` views of one storage; we instead emit
    one `CanonicalKVCacheTensor` per *pages* tensor, flattened on the trailing
    dims with `.view()`, which is metadata-only and device-safe.

    K and V therefore become two canonical tensors per physical cache, each
    carrying half of `AttentionSpec.page_size_bytes` (upstream's page size spans
    both). Layers sharing one `SpyrePagedKVCache` share both tensor indices.

    Returns the canonical view *and* the physical caches, because the DMA path
    cannot use the former. Kept as a single function so upstream drift has one
    blast radius.
    """
    specs = _specs_by_layer(kv_cache_config)

    tensors: list[CanonicalKVCacheTensor] = []
    # id(SpyrePagedKVCache) -> the (k_idx, v_idx) it contributed to `tensors`,
    # so layers sharing a physical cache reuse the same entries.
    indices_by_cache: dict[int, list[int]] = {}
    refs_by_layer: dict[str, list[CanonicalKVCacheRef]] = {}

    # Parallel accumulators for the physical half, in first-seen order.
    physical: list[tuple[torch.Tensor, torch.Tensor]] = []
    layout_kinds: list[str] = []
    # id(cache) -> cache index, and the layer that first claimed it (for errors).
    cache_index: dict[int, int] = {}
    first_layer: dict[int, str] = {}
    tensor_idx_to_cache: dict[int, int] = {}
    num_blocks: int | None = None

    for layer_name, spec in specs.items():
        if not isinstance(spec, AttentionSpec):
            raise NotImplementedError(
                f"Spyre KV offloading supports AttentionSpec layers only; "
                f"layer {layer_name!r} has {type(spec).__name__}"
            )

        cache = kv_caches[layer_name]
        k_pages, v_pages = _unpack_paged_cache(layer_name, cache)

        # Upstream's page spans K and V together; ours are separate tensors.
        if spec.page_size_bytes % 2:
            raise ValueError(
                f"layer {layer_name!r}: odd page_size_bytes "
                f"{spec.page_size_bytes} cannot be split across K and V"
            )
        half_page = spec.page_size_bytes // 2
        _check_page_size(layer_name, k_pages, half_page)

        layout_kind = layout_kind_by_layer.get(layer_name)
        if layout_kind not in (TOKEN_MAJOR, HEAD_MAJOR):
            raise ValueError(
                f"layer {layer_name!r}: unresolved layout kind {layout_kind!r}; "
                f"expected {TOKEN_MAJOR!r} or {HEAD_MAJOR!r}"
            )

        if num_blocks is None:
            num_blocks = int(k_pages.shape[0])
        elif int(k_pages.shape[0]) != num_blocks:
            raise ValueError(
                f"layer {layer_name!r}: num_blocks {k_pages.shape[0]} disagrees "
                f"with {num_blocks} seen on earlier layers"
            )

        cache_id = id(cache)
        if cache_id not in indices_by_cache:
            indices_by_cache[cache_id] = [
                _append_flat_tensor(tensors, pages, half_page)
                for pages in (k_pages, v_pages)
            ]
            cache_index[cache_id] = len(physical)
            first_layer[cache_id] = layer_name
            physical.append((k_pages, v_pages))
            layout_kinds.append(layout_kind)
            for idx in indices_by_cache[cache_id]:
                tensor_idx_to_cache[idx] = cache_index[cache_id]
        elif layout_kinds[cache_index[cache_id]] != layout_kind:
            # Layers sharing one allocation must have been allocated by one
            # backend; disagreement means the impl lookup is wrong, and
            # guessing would transpose block_size/num_kv_heads.
            raise ValueError(
                f"layers {first_layer[cache_id]!r} and {layer_name!r} share one "
                f"KV allocation but disagree on layout: "
                f"{layout_kinds[cache_index[cache_id]]!r} vs {layout_kind!r}"
            )

        # mapping=None: the byte layout is device-private, so it is not
        # certified as parallelism-agnostic and stays worker-local.
        refs_by_layer[layer_name] = [
            CanonicalKVCacheRef(tensor_idx=idx, page_size_bytes=half_page, mapping=None)
            for idx in indices_by_cache[cache_id]
        ]

    if num_blocks is None:
        raise ValueError("no AttentionSpec layers found to offload")

    group_data_refs = [
        [ref for layer_name in group.layer_names for ref in refs_by_layer[layer_name]]
        for group in kv_cache_config.kv_cache_groups
    ]
    canonical = CanonicalKVCaches(tensors=tensors, group_data_refs=group_data_refs)
    physical_caches = SpyrePhysicalCaches(
        caches=tuple(physical),
        layout_kinds=tuple(layout_kinds),
        tensor_idx_to_cache=tensor_idx_to_cache,
        num_blocks=num_blocks,
    )
    return canonical, physical_caches


def _layout_kind_by_layer(
    vllm_config: VllmConfig, layer_names: Iterable[str]
) -> dict[str, str]:
    """Resolve each layer's KV layout from its attention impl.

    The layout is *not* inferrable from the tensor: both layouts fold logical
    dim 1 into device dim 0, so `device_size[0] // num_blocks == size(1)` either
    way (see `spyre_kv_offload.page_signature`). Only the allocating backend
    knows, so read the same `static_forward_context[layer].impl` the model runner
    uses to pick `allocate_pages`.
    """
    static_ctx = vllm_config.compilation_config.static_forward_context
    kinds: dict[str, str] = {}
    for layer_name in layer_names:
        layer = static_ctx.get(layer_name)
        impl = getattr(layer, "impl", None)
        if impl is None:
            raise ValueError(
                f"layer {layer_name!r}: no attention impl in "
                f"static_forward_context; cannot determine KV layout"
            )
        kinds[layer_name] = (
            HEAD_MAJOR
            if type(impl).__name__ == "SpyreHeadMajorAttentionImpl"
            else TOKEN_MAJOR
        )
    return kinds


def _specs_by_layer(kv_cache_config: KVCacheConfig) -> dict[str, object]:
    specs: dict[str, object] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        per_layer = (
            group_spec.kv_cache_specs
            if isinstance(group_spec, UniformTypeKVCacheSpecs)
            else {}
        )
        for layer_name in group.layer_names:
            specs[layer_name] = per_layer.get(layer_name, group_spec)
    return specs


def _unpack_paged_cache(
    layer_name: str, cache: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Duck-type SpyrePagedKVCache without importing the attention stack."""
    if not (isinstance(cache, tuple) and len(cache) == 2):
        raise TypeError(
            f"layer {layer_name!r}: expected a SpyrePagedKVCache 2-tuple, got "
            f"{type(cache).__name__}"
        )
    k_pages, v_pages = cache
    if not isinstance(k_pages, torch.Tensor) or not isinstance(v_pages, torch.Tensor):
        raise TypeError(
            f"layer {layer_name!r}: expected Tensor k/v pages, got "
            f"{type(k_pages).__name__}/{type(v_pages).__name__}"
        )
    if k_pages.shape != v_pages.shape or k_pages.dtype != v_pages.dtype:
        raise ValueError(
            f"layer {layer_name!r}: k/v pages disagree -- "
            f"{tuple(k_pages.shape)}/{k_pages.dtype} vs "
            f"{tuple(v_pages.shape)}/{v_pages.dtype}"
        )
    return k_pages, v_pages


def _check_page_size(layer_name: str, pages: torch.Tensor, expected_bytes: int) -> None:
    actual = pages[0].numel() * pages.element_size()
    if actual != expected_bytes:
        raise ValueError(
            f"layer {layer_name!r}: device page is {actual} bytes but the spec "
            f"implies {expected_bytes} per K/V half"
        )


def _append_flat_tensor(
    tensors: list[CanonicalKVCacheTensor],
    pages: torch.Tensor,
    page_size_bytes: int,
) -> int:
    """Append `pages` flattened to (num_blocks, -1) and return its index."""
    if not pages.is_contiguous():
        raise ValueError("Spyre KV pages must be contiguous to canonicalize")
    tensors.append(
        CanonicalKVCacheTensor(
            tensor=pages.view(pages.shape[0], -1),
            page_size_bytes=page_size_bytes,
        )
    )
    return len(tensors) - 1


class SpyreOffloadingConnectorWorker(OffloadingConnectorWorker):
    """Worker overriding only `register_kv_caches` for the paged-list layout."""

    def __init__(self, *args, **kwargs):
        # Upstream has already changed this signature once (it gained
        # `vllm_config` as a 2nd positional arg); forward blindly.
        super().__init__(*args, **kwargs)
        # `upstream_compat` guards the module move but not the attribute
        # contract this override depends on. Fail here, naming the attribute,
        # rather than at registration with a bare AttributeError.
        for attr in ("spec", "vllm_config", "kv_cache_config", "_init_worker"):
            if not hasattr(self, attr):
                raise AttributeError(
                    f"upstream OffloadingConnectorWorker has no {attr!r}; "
                    f"vLLM changed the contract this subclass relies on"
                )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # Imported here so a module-level import cycle cannot form and so the
        # spec module (which imports torch_spyre lazily) stays off the import
        # path until registration actually happens.
        from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec

        if not isinstance(self.spec, SpyreOffloadingSpec):
            raise TypeError(
                f"SpyreOffloadingConnector requires SpyreOffloadingSpec, got "
                f"{type(self.spec).__name__}. Set spec_name=SpyreOffloadingSpec "
                f"and spec_module_path=spyre_inference.v1.kv_offload.spec in "
                f"kv_connector_extra_config."
            )

        canonical, physical = spyre_paged_to_canonical(
            kv_caches,
            self.kv_cache_config,
            _layout_kind_by_layer(self.vllm_config, kv_caches.keys()),
        )
        logger.info(
            "Spyre KV offloading: canonicalized %d layer(s) into %d tensor(s) "
            "over %d physical cache(s), layouts=%s",
            len(kv_caches),
            len(canonical.tensors),
            len(physical.caches),
            ",".join(physical.layout_kinds),
        )
        self.spec.bind_physical_caches(physical)
        self._init_worker(canonical)


class SpyreOffloadingConnector(OffloadingConnector):
    """OffloadingConnector wired to the Spyre worker. Scheduler side untouched.

    Registered via `KVTransferConfig.kv_connector_module_path`; no upstream patch.
    """

    # Upstream reads this only from `use_uniform_kv_cache`, which Spyre's
    # model-runner override never reaches -- so the cross-layer slab path is
    # already dead here. Pin it False anyway: if upstream starts consulting it
    # on a path Spyre does take, a slab layout would break the per-cache
    # rank-4 assumption the DMA path depends on.
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_worker is not None:
            self.connector_worker = SpyreOffloadingConnectorWorker(
                self.connector_worker.spec,
                vllm_config,
                kv_cache_config,
            )
