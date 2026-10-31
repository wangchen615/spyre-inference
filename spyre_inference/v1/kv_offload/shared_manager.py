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

"""One global cache policy backed by partitioned cross-instance metadata."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import override
from vllm.v1.kv_offload.base import (
    CacheHitSource,
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

from spyre_inference.v1.kv_offload.pool_plan import PlannedPool
from spyre_inference.v1.kv_offload.shared_attachment import (
    SharedMetadataAttachment,
    SharedPoolAttachment,
)
from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
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


@dataclass(frozen=True)
class _ComponentRoute:
    component_id: int
    registered: object


class SpyreSharedOffloadingManager(OffloadingManager):
    """Apply one logical cache policy to complete shared page bundles."""

    def __init__(
        self,
        *,
        cache_policy: str,
        cache_policy_module_path: str | None,
        enable_events: bool,
        store_threshold: int,
        max_tracker_size: int,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
        plan: PlannedPool | None = None,
        identity: SharedPoolIdentity | None = None,
        component_count: int | None = None,
        max_pool_slots: int | None = None,
        tp_rank: int | None = None,
    ) -> None:
        if plan is None:
            if identity is None or component_count is None or max_pool_slots is None:
                raise ValueError(
                    "a complete plan or legacy identity/component geometry is required"
                )
            if component_count <= 0 or max_pool_slots <= 0:
                raise ValueError("legacy component geometry must be positive")
        self._plan = plan
        if plan is not None:
            if tp_rank is None:
                raise ValueError("planned shared manager requires a tensor-parallel rank")
            if not 0 <= tp_rank < plan.tensor_parallel_size:
                raise ValueError(
                    f"tensor-parallel rank {tp_rank} is outside [0, {plan.tensor_parallel_size})"
                )
        self._tp_rank = tp_rank
        self._identity = plan.identity if plan is not None else identity
        if plan is not None:
            self._component_count = len(plan.components)
        else:
            assert component_count is not None
            self._component_count = component_count
        self._max_pool_slots = max_pool_slots
        self._cache_policy = cache_policy
        self._cache_policy_module_path = cache_policy_module_path
        self._enable_events = enable_events
        self._store_threshold = store_threshold
        self._max_tracker_size = max_tracker_size
        self._runtime_loader = runtime_loader
        self._runtime = None
        self._attachment: SharedMetadataAttachment | SharedPoolAttachment | None = None
        self._directory = None
        self._registered = None
        self._component_routes: tuple[_ComponentRoute, ...] = ()
        self._store_routes: tuple[_ComponentRoute, ...] = ()
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
        if self._attachment is None:
            runtime = self._runtime_loader()
            if self._plan is None:
                assert self._identity is not None
                attachment = SharedPoolAttachment.attach(self._identity, lambda: runtime)
                registered = attachment.registered
                routes = tuple(
                    _ComponentRoute(component_id, registered)
                    for component_id in range(self._component_count)
                )
                store_routes = routes
                slot_count = int(registered.slot_count)
                if (
                    slot_count <= 0
                    or slot_count % self._component_count
                    or self._max_pool_slots is None
                    or slot_count > self._max_pool_slots
                ):
                    attachment.close()
                    raise RuntimeError(
                        f"registered pool slot_count {slot_count} is incompatible with "
                        f"component_count={self._component_count} and "
                        f"max_pool_slots={self._max_pool_slots}"
                    )
                policy_capacity = slot_count // self._component_count
            else:
                assert self._tp_rank is not None
                attachment = SharedMetadataAttachment.attach(
                    self._plan,
                    self._tp_rank,
                    lambda: runtime,
                )
                routes = tuple(
                    _ComponentRoute(
                        component.global_component_id,
                        attachment.registered[component.partition_name],
                    )
                    for component in self._plan.components
                )
                store_routes = tuple(
                    route
                    for route in routes
                    if self._plan.components[route.component_id].tp_rank == self._tp_rank
                )
                registered = routes[0].registered
                policy_capacity = self._plan.logical_block_capacity
            directory = attachment.metadata
            try:
                local_manager = CPUOffloadingManager(
                    num_chunks=policy_capacity,
                    cache_policy=self._cache_policy,
                    cache_policy_module_path=self._cache_policy_module_path,
                    enable_events=self._enable_events,
                    store_threshold=self._store_threshold,
                    max_tracker_size=self._max_tracker_size,
                )
            except BaseException:
                attachment.close()
                raise

            self._runtime = runtime
            self._attachment = attachment
            self._directory = directory
            self._registered = registered
            self._component_routes = routes
            self._store_routes = store_routes
            self._local_manager = local_manager

        directory = self._directory
        registered = self._registered
        runtime = self._runtime
        if directory is None or registered is None or runtime is None:
            raise RuntimeError("shared pool attachment was not initialized completely")

        assert self._local_manager is not None
        return directory, self._component_routes, runtime, self._local_manager

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
        if not any(tracked is state for tracked in self._request_states):
            self._request_states.append(state)
        return state

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        _, _, _, local_manager = self._ensure_directory()
        return local_manager.on_new_request(req_context)

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        directory, routes, runtime, local_manager = self._ensure_directory()
        local_result = local_manager.lookup(key, req_context)
        if local_result is LookupResult.HIT_PENDING:
            return local_result

        state = self._request_state(req_context)
        if key in state.pending_pins or key in state.active_pins:
            return LookupResult.HIT

        pinned: list[_PinnedPage] = []
        for route in routes:
            component_id = route.component_id
            registered = route.registered
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
    def get_load_source(self, key: OffloadKey, req_context: ReqContext) -> CacheHitSource:
        return CacheHitSource.HOST

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
                # The worker may already have aborted this exact token.
                with suppress(RuntimeError):
                    directory.abort(reservation)
            elif self._same_slot(entry.slot, reservation.slot):
                directory.evict(entry)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        directory, _, runtime, local_manager = self._ensure_directory()
        routes = self._store_routes
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
                for route in routes:
                    component_id = route.component_id
                    registered = route.registered
                    page_key = self._page_key(runtime, registered, key, component_id)
                    result = directory.claim(registered.pool_ref, page_key)
                    if isinstance(result, runtime.Reservation):
                        if not self._valid_slot(result.slot, registered):
                            raise RuntimeError("reservation returned a foreign pool slot")
                        token = result.token
                        current_pages.append(
                            SharedPageTransfer(
                                component_id,
                                SharedPageLocation(token.slot.pool.pool_id, token.slot.slot_id),
                                token,
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
        directory, _, runtime, local_manager = self._ensure_directory()
        routes = self._store_routes
        keys = tuple(keys)
        if not success:
            for key in keys:
                pending = self._pending_stores.pop(key, None)
                if pending is not None:
                    self._release_attempt_pages(directory, pending.pages)
            local_manager.complete_store(keys, req_context, success=False)
            return

        ownership: dict[OffloadKey, tuple[object, ...]] = {}
        try:
            for key in keys:
                pending = self._pending_stores[key]
                reservations = {page.component_id: page.reservation for page in pending.pages}
                owned = []
                for route in routes:
                    component_id = route.component_id
                    registered = route.registered
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

    @override
    def shutdown(self) -> None:
        self.reset_cache()
        local_manager = self._local_manager
        if local_manager is not None:
            local_manager.shutdown()
        attachment = self._attachment
        if attachment is not None:
            attachment.close()
        self._attachment = None
        self._directory = None
        self._registered = None
        self._component_routes = ()
        self._store_routes = ()
        self._local_manager = None
        self._runtime = None
