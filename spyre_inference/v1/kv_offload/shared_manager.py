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

"""Upstream CPU-cache policy backed by the cross-instance Spyre directory."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import override
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from spyre_inference.v1.kv_offload.shared_runtime import load_shared_runtime
from spyre_inference.v1.kv_offload.shared_types import (
    SharedLoadStoreSpec,
    SharedLocation,
    SharedPoolFamily,
    SharedTransfer,
    shared_block_hash,
)


@dataclass
class _PinnedEntry:
    entry: object
    pin: object


@dataclass(eq=False)
class _SharedRequestState:
    pending_pins: dict[OffloadKey, _PinnedEntry] = field(default_factory=dict)
    active_pins: dict[OffloadKey, _PinnedEntry] = field(default_factory=dict)


class SpyreSharedOffloadingManager(CPUOffloadingManager):
    """Retain upstream policy while resolving bytes through SharedMetadata."""

    def __init__(
        self,
        *,
        metadata_name: str,
        families: tuple[SharedPoolFamily, ...],
        max_components: int,
        num_blocks: int,
        cache_policy: str,
        cache_policy_module_path: str | None,
        enable_events: bool,
        store_threshold: int,
        max_tracker_size: int,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> None:
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,
            cache_policy_module_path=cache_policy_module_path,
            enable_events=enable_events,
            store_threshold=store_threshold,
            max_tracker_size=max_tracker_size,
        )
        self._metadata_name = metadata_name
        self._families = families
        self._max_components = max_components
        self._runtime_loader = runtime_loader
        self._runtime = None
        self._directory = None
        self._anchors = None
        self._owned_entries: dict[OffloadKey, object] = {}
        self._pending_reservations: dict[OffloadKey, object] = {}
        self._request_states: list[_SharedRequestState] = []

    @property
    def directory(self):
        directory, _, _ = self._ensure_directory()
        return directory

    def _ensure_directory(self):
        runtime = self._runtime
        if runtime is None:
            runtime = self._runtime_loader()
            self._runtime = runtime
        directory = self._directory
        if directory is None:
            capacity = runtime.SharedMetadataCapacity(
                len(self._families) * self._max_components,
                max(family.slot_count for family in self._families),
                1,
            )
            config = runtime.SharedMetadataConfig(self._max_components, [], capacity)
            directory = runtime.SharedMetadata.create_or_attach(self._metadata_name, config)
            self._directory = directory

        anchors = self._anchors
        if anchors is None:
            resolved = []
            for family in self._families:
                name = f"{family.name}.c0.k"
                anchor = directory.find_pool(name)
                if anchor is None:
                    raise RuntimeError(
                        f"shared pool anchor {name!r} is not registered; "
                        "worker KV-cache registration must finish before lookup"
                    )
                resolved.append(anchor)
            compatibility_ids = {
                (
                    anchor.compatibility.metadata_version,
                    anchor.compatibility.compatibility_id,
                )
                for anchor in resolved
            }
            if len(compatibility_ids) != 1:
                raise RuntimeError("shared pool family anchors have incompatible descriptors")
            anchors = tuple(resolved)
            self._anchors = anchors
        return directory, anchors, runtime

    def _request_state(self, req_context: ReqContext) -> _SharedRequestState:
        state = req_context.get_state(_SharedRequestState)
        if state is None:
            state = _SharedRequestState()
            req_context.set_state(state)
            self._request_states.append(state)
        return state

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        local_result = super().lookup(key, req_context)
        if local_result is LookupResult.HIT_PENDING:
            return local_result

        state = self._request_state(req_context)
        if key in state.pending_pins or key in state.active_pins:
            return LookupResult.HIT

        directory, anchors, runtime = self._ensure_directory()
        compatible_key = runtime.CompatibleBlockKey(
            anchors[0].compatibility, shared_block_hash(key)
        )
        entry = directory.lookup(compatible_key)
        if entry is None:
            return LookupResult.MISS
        pin = directory.pin_read(entry)
        if pin is None:
            return LookupResult.MISS
        state.pending_pins[key] = _PinnedEntry(entry, pin)
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        state = self._request_state(req_context)
        requested = set(keys)
        for key in tuple(state.pending_pins):
            if key not in requested:
                del state.pending_pins[key]

        local_keys = [key for key in keys if key in self._owned_entries]
        if local_keys:
            super().prepare_load(local_keys, req_context)

        transfers = []
        for key in keys:
            pinned = state.pending_pins.pop(key, None)
            if pinned is None:
                raise RuntimeError(f"shared block {key!r} was not pinned by lookup")
            state.active_pins[key] = pinned
            slot = pinned.entry.slot
            transfers.append(
                SharedTransfer(
                    key,
                    SharedLocation(slot.pool.pool_id, slot.slot_id),
                )
            )
        return SharedLoadStoreSpec(transfers)

    @override
    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        state = self._request_state(req_context)
        local_keys = [key for key in keys if key in self._owned_entries]
        try:
            if local_keys:
                super().complete_load(local_keys, req_context)
        finally:
            for key in keys:
                state.active_pins.pop(key, None)
            if not state.pending_pins and not state.active_pins and state in self._request_states:
                self._request_states.remove(state)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        state = req_context.get_state(_SharedRequestState)
        if state is not None:
            state.pending_pins.clear()
            if not state.active_pins and state in self._request_states:
                self._request_states.remove(state)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        directory, anchors, runtime = self._ensure_directory()
        upstream = super().prepare_store(keys, req_context)
        if upstream is None:
            return None

        for key in upstream.evicted_keys:
            entry = self._owned_entries.pop(key)
            directory.evict(entry)

        transfers: list[SharedTransfer] = []
        try:
            for key in upstream.keys_to_store:
                block_hash = shared_block_hash(key)
                start = block_hash % len(anchors)
                reservation = None
                for offset in range(len(anchors)):
                    anchor = anchors[(start + offset) % len(anchors)]
                    compatible_key = runtime.CompatibleBlockKey(anchor.compatibility, block_hash)
                    result = directory.claim(anchor.pool_ref, compatible_key)
                    if isinstance(result, runtime.Reservation):
                        reservation = result
                        break
                    if isinstance(result, runtime.NoSpace):
                        continue
                    if isinstance(result, runtime.ExistingClaim):
                        break
                    if isinstance(result, runtime.Unavailable):
                        raise RuntimeError("shared metadata is unavailable while claiming a slot")
                    raise RuntimeError(
                        f"shared metadata returned unknown claim result {type(result).__name__}"
                    )

                if reservation is None:
                    super().complete_store([key], req_context, success=False)
                    continue

                self._pending_reservations[key] = reservation
                slot = reservation.slot
                transfers.append(
                    SharedTransfer(
                        key,
                        SharedLocation(slot.pool.pool_id, slot.slot_id),
                        reservation,
                    )
                )
        except Exception:
            for transfer in transfers:
                directory.abort(transfer.reservation)
                self._pending_reservations.pop(transfer.key, None)
            super().complete_store(upstream.keys_to_store, req_context, success=False)
            raise

        return PrepareStoreOutput(
            keys_to_store=[transfer.key for transfer in transfers],
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
        if not success:
            for key in keys:
                self._pending_reservations.pop(key, None)
            super().complete_store(keys, req_context, success=False)
            return

        directory, _, _ = self._ensure_directory()
        entries = {}
        try:
            for key in keys:
                reservation = self._pending_reservations[key]
                entry = directory.lookup(reservation.key)
                if entry is None:
                    raise RuntimeError(
                        f"shared block {key!r} was not published after store completion"
                    )
                entries[key] = entry
        except Exception:
            super().complete_store(keys, req_context, success=False)
            for key in keys:
                self._pending_reservations.pop(key, None)
            raise

        super().complete_store(keys, req_context, success=True)
        for key, entry in entries.items():
            self._owned_entries[key] = entry
            self._pending_reservations.pop(key, None)

    @override
    def reset_cache(self) -> None:
        for state in self._request_states:
            state.pending_pins.clear()
            state.active_pins.clear()
        self._request_states.clear()
        if self._directory is not None:
            for entry in self._owned_entries.values():
                self._directory.evict(entry)
            for reservation in self._pending_reservations.values():
                self._directory.abort(reservation)
        self._owned_entries.clear()
        self._pending_reservations.clear()
        super().reset_cache()
