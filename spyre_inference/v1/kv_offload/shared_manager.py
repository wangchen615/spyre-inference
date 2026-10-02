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

"""Upstream cache policy backed by one cross-instance Spyre data pool."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import override
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from spyre_inference.v1.kv_offload.shared_runtime import load_shared_runtime
from spyre_inference.v1.kv_offload.shared_types import (
    SharedBlockTransfer,
    SharedLoadStoreSpec,
    SharedPageLocation,
    SharedPageTransfer,
    shared_page_hash,
)


@dataclass
class _PinnedPage:
    component_id: int
    entry: object
    pin: object


@dataclass(eq=False)
class _SharedRequestState:
    pending_pins: dict[OffloadKey, tuple[_PinnedPage, ...]] = field(default_factory=dict)
    active_pins: dict[OffloadKey, tuple[_PinnedPage, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingStore:
    pages: tuple[SharedPageTransfer, ...]


class SpyreSharedOffloadingManager(OffloadingManager):
    """Apply one logical cache policy to complete shared page bundles."""

    def __init__(
        self,
        *,
        metadata_name: str,
        pool_name: str,
        component_count: int,
        max_pool_slots: int,
        cache_policy: str,
        cache_policy_module_path: str | None,
        enable_events: bool,
        store_threshold: int,
        max_tracker_size: int,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> None:
        self._metadata_name = metadata_name
        self._pool_name = pool_name
        self._component_count = component_count
        self._max_pool_slots = max_pool_slots
        self._cache_policy = cache_policy
        self._cache_policy_module_path = cache_policy_module_path
        self._enable_events = enable_events
        self._store_threshold = store_threshold
        self._max_tracker_size = max_tracker_size
        self._runtime_loader = runtime_loader
        self._runtime = None
        self._directory = None
        self._registered = None
        self._local_manager: CPUOffloadingManager | None = None
        self._local_keys: set[OffloadKey] = set()
        self._owned_entries: dict[OffloadKey, tuple[object, ...]] = {}
        self._pending_stores: dict[OffloadKey, _PendingStore] = {}
        self._request_states: list[_SharedRequestState] = []

    @property
    def directory(self):
        directory, _, _, _ = self._ensure_directory()
        return directory

    def _ensure_directory(self):
        runtime = self._runtime
        if runtime is None:
            runtime = self._runtime_loader()
            self._runtime = runtime

        directory = self._directory
        if directory is None:
            capacity = runtime.SharedMetadataCapacity(1, self._max_pool_slots, 1)
            config = runtime.SharedMetadataConfig(1, [], capacity)
            directory = runtime.SharedMetadata.create_or_attach(self._metadata_name, config)
            self._directory = directory

        registered = self._registered
        if registered is None:
            registered = directory.find_pool(self._pool_name)
            if registered is None:
                raise RuntimeError(
                    f"shared pool {self._pool_name!r} is not registered; "
                    "worker KV-cache registration must finish before lookup"
                )
            slot_count = int(registered.slot_count)
            if (
                slot_count <= 0
                or slot_count % self._component_count
                or slot_count > self._max_pool_slots
            ):
                raise RuntimeError(
                    f"registered pool slot_count {slot_count} is incompatible with "
                    f"component_count={self._component_count} and "
                    f"max_pool_slots={self._max_pool_slots}"
                )
            self._registered = registered
            self._local_manager = CPUOffloadingManager(
                num_blocks=slot_count // self._component_count,
                cache_policy=self._cache_policy,
                cache_policy_module_path=self._cache_policy_module_path,
                enable_events=self._enable_events,
                store_threshold=self._store_threshold,
                max_tracker_size=self._max_tracker_size,
            )

        assert self._local_manager is not None
        return directory, registered, runtime, self._local_manager

    @staticmethod
    def _same_pool(left: object, right: object) -> bool:
        return all(
            getattr(left, field_name, None) == getattr(right, field_name, None)
            for field_name in ("metadata_version", "pool_id", "pool_version")
        )

    @classmethod
    def _same_slot(cls, left: object, right: object) -> bool:
        return (
            cls._same_pool(left.pool, right.pool)
            and left.slot_id == right.slot_id
            and left.slot_version == right.slot_version
        )

    def _valid_slot(self, slot: object, registered: object) -> bool:
        return self._same_pool(slot.pool, registered.pool_ref) and 0 <= int(slot.slot_id) < int(
            registered.slot_count
        )

    def _page_key(self, runtime: object, registered: object, key: OffloadKey, component_id: int):
        return runtime.CompatibleBlockKey(
            registered.compatibility,
            shared_page_hash(key, component_id),
        )

    def _request_state(self, req_context: ReqContext) -> _SharedRequestState:
        state = req_context.get_state(_SharedRequestState)
        if state is None:
            state = _SharedRequestState()
            req_context.set_state(state)
            self._request_states.append(state)
        return state

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        _, _, _, local_manager = self._ensure_directory()
        return local_manager.on_new_request(req_context)

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        directory, registered, runtime, local_manager = self._ensure_directory()
        local_result = local_manager.lookup(key, req_context)
        if local_result is LookupResult.HIT_PENDING:
            return local_result

        state = self._request_state(req_context)
        if key in state.pending_pins or key in state.active_pins:
            return LookupResult.HIT

        pinned: list[_PinnedPage] = []
        for component_id in range(self._component_count):
            page_key = self._page_key(runtime, registered, key, component_id)
            entry = directory.lookup(page_key)
            if entry is None or not self._valid_slot(entry.slot, registered):
                pinned.clear()
                return LookupResult.MISS
            pin = directory.pin_read(entry)
            if pin is None:
                pinned.clear()
                return LookupResult.MISS
            pinned.append(_PinnedPage(component_id, entry, pin))

        state.pending_pins[key] = tuple(pinned)
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        _, _, _, local_manager = self._ensure_directory()
        state = self._request_state(req_context)
        requested = set(keys)
        for key in tuple(state.pending_pins):
            if key not in requested:
                del state.pending_pins[key]

        local_keys = [key for key in keys if key in self._local_keys]
        if local_keys:
            local_manager.prepare_load(local_keys, req_context)

        transfers = []
        for key in keys:
            pinned = state.pending_pins.pop(key, None)
            if pinned is None:
                raise RuntimeError(f"shared block {key!r} was not pinned by lookup")
            state.active_pins[key] = pinned
            transfers.append(
                SharedBlockTransfer(
                    key,
                    tuple(
                        SharedPageTransfer(
                            item.component_id,
                            SharedPageLocation(
                                item.entry.slot.pool.pool_id,
                                item.entry.slot.slot_id,
                            ),
                        )
                        for item in pinned
                    ),
                )
            )
        return SharedLoadStoreSpec(transfers)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        _, _, _, local_manager = self._ensure_directory()
        local_manager.touch(keys, req_context)

    @override
    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        _, _, _, local_manager = self._ensure_directory()
        state = self._request_state(req_context)
        local_keys = [key for key in keys if key in self._local_keys]
        try:
            if local_keys:
                local_manager.complete_load(local_keys, req_context)
        finally:
            for key in keys:
                state.active_pins.pop(key, None)
            if not state.pending_pins and not state.active_pins and state in self._request_states:
                self._request_states.remove(state)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        _, _, _, local_manager = self._ensure_directory()
        local_manager.on_request_finished(req_context)
        state = req_context.get_state(_SharedRequestState)
        if state is not None:
            state.pending_pins.clear()
            if not state.active_pins and state in self._request_states:
                self._request_states.remove(state)

    def _abort_pages(self, directory: object, pages: Collection[SharedPageTransfer]) -> None:
        for page in pages:
            directory.abort(page.reservation)

    def _release_attempt_pages(
        self,
        directory: object,
        pages: Collection[SharedPageTransfer],
    ) -> None:
        for page in pages:
            reservation = page.reservation
            entry = directory.lookup(reservation.key)
            if entry is None:
                directory.abort(reservation)
            elif self._same_slot(entry.slot, reservation.slot):
                directory.evict(entry)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        directory, registered, runtime, local_manager = self._ensure_directory()
        upstream = local_manager.prepare_store(keys, req_context)
        if upstream is None:
            return None

        for key in upstream.evicted_keys:
            for entry in self._owned_entries.pop(key, ()):
                directory.evict(entry)
            self._local_keys.discard(key)

        transfers: list[SharedBlockTransfer] = []
        prepared_keys: list[OffloadKey] = []
        ready_without_transfer: list[OffloadKey] = []
        current_pages: list[SharedPageTransfer] = []
        try:
            for key in upstream.keys_to_store:
                current_pages = []
                complete = True
                for component_id in range(self._component_count):
                    page_key = self._page_key(runtime, registered, key, component_id)
                    result = directory.claim(registered.pool_ref, page_key)
                    if isinstance(result, runtime.Reservation):
                        if not self._valid_slot(result.slot, registered):
                            raise RuntimeError("reservation returned a foreign pool slot")
                        current_pages.append(
                            SharedPageTransfer(
                                component_id,
                                SharedPageLocation(result.slot.pool.pool_id, result.slot.slot_id),
                                result,
                            )
                        )
                    elif isinstance(result, runtime.ExistingClaim):
                        if not self._valid_slot(result.slot, registered):
                            raise RuntimeError("existing claim returned a foreign pool slot")
                        if not result.valid or directory.lookup(page_key) is None:
                            complete = False
                            break
                    elif isinstance(result, runtime.NoSpace):
                        complete = False
                        break
                    elif isinstance(result, runtime.Unavailable):
                        raise RuntimeError("shared metadata is unavailable while claiming a page")
                    else:
                        raise RuntimeError(
                            f"shared metadata returned unknown claim result {type(result).__name__}"
                        )

                if not complete:
                    self._abort_pages(directory, current_pages)
                    current_pages = []
                    local_manager.complete_store([key], req_context, success=False)
                    continue
                if current_pages:
                    block = SharedBlockTransfer(key, tuple(current_pages))
                    transfers.append(block)
                    prepared_keys.append(key)
                    self._pending_stores[key] = _PendingStore(block.pages)
                    current_pages = []
                else:
                    ready_without_transfer.append(key)

            if ready_without_transfer:
                local_manager.complete_store(ready_without_transfer, req_context, success=True)
                self._local_keys.update(ready_without_transfer)
        except Exception:
            self._abort_pages(directory, current_pages)
            for block in transfers:
                self._abort_pages(directory, block.pages)
                self._pending_stores.pop(block.key, None)
            local_manager.complete_store(upstream.keys_to_store, req_context, success=False)
            raise

        return PrepareStoreOutput(
            keys_to_store=prepared_keys,
            store_spec=SharedLoadStoreSpec(transfers),
            evicted_keys=upstream.evicted_keys,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        directory, registered, runtime, local_manager = self._ensure_directory()
        keys = tuple(keys)
        if not success:
            for key in keys:
                self._pending_stores.pop(key, None)
            local_manager.complete_store(keys, req_context, success=False)
            return

        ownership: dict[OffloadKey, tuple[object, ...]] = {}
        try:
            for key in keys:
                pending = self._pending_stores[key]
                reservations = {page.component_id: page.reservation for page in pending.pages}
                owned = []
                for component_id in range(self._component_count):
                    page_key = self._page_key(runtime, registered, key, component_id)
                    entry = directory.lookup(page_key)
                    if entry is None or not self._valid_slot(entry.slot, registered):
                        raise RuntimeError(
                            f"shared block {key!r} does not have a complete page set"
                        )
                    reservation = reservations.get(component_id)
                    if reservation is not None:
                        if not self._same_slot(entry.slot, reservation.slot):
                            raise RuntimeError(
                                f"shared block {key!r} publication changed its page slot"
                            )
                        owned.append(entry)
                ownership[key] = tuple(owned)
        except Exception:
            for key in keys:
                pending = self._pending_stores.pop(key, None)
                if pending is None:
                    continue
                self._release_attempt_pages(directory, pending.pages)
            local_manager.complete_store(keys, req_context, success=False)
            raise

        local_manager.complete_store(keys, req_context, success=True)
        for key in keys:
            self._owned_entries[key] = ownership[key]
            self._local_keys.add(key)
            self._pending_stores.pop(key, None)

    @override
    def reset_cache(self) -> None:
        for state in self._request_states:
            state.pending_pins.clear()
            state.active_pins.clear()
        self._request_states.clear()

        if self._directory is not None:
            for entries in self._owned_entries.values():
                for entry in entries:
                    self._directory.evict(entry)
            for pending in self._pending_stores.values():
                self._release_attempt_pages(self._directory, pending.pages)
        self._owned_entries.clear()
        self._pending_stores.clear()
        self._local_keys.clear()
        if self._local_manager is not None:
            self._local_manager.reset_cache()

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        _, _, _, local_manager = self._ensure_directory()
        return local_manager.take_events()

    @override
    def get_stats(self):
        _, _, _, local_manager = self._ensure_directory()
        return local_manager.get_stats()
