#!/usr/bin/env python3
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

"""Remove persistent shared-memory objects created by the two-instance demo."""

from __future__ import annotations

import argparse
import os
from math import ceil
from pathlib import Path
from typing import Any

DEFAULT_METADATA_NAME = "spyre_manual_4096"
DEFAULT_COMPONENT_COUNT = 8
DEFAULT_MAX_POOL_SLOTS = 2048
DEFAULT_LEGACY_CACHE_COUNT = 4
DEFAULT_LEGACY_TOTAL_HOST_BLOCKS = 256
MOCK_RUNTIME_ENV = {
    "VLLM_PLUGINS": "",
    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    "FLEX_DEVICE": "MOCK1p0",
    "FLEX_COMPUTE": "NULL",
    "AIU_WORLD_SIZE": "1",
    "LOCAL_RANK": "0",
}


def _live_vllm_processes(
    metadata_name: str,
    metadata_path: Path,
    proc_root: Path,
) -> list[tuple[int, str]]:
    matches = []
    for process_dir in proc_root.glob("[0-9]*"):
        if not process_dir.name.isdigit():
            continue
        try:
            argv = [
                argument.decode()
                for argument in (process_dir / "cmdline").read_bytes().split(b"\0")
                if argument
            ]
        except (OSError, UnicodeDecodeError):
            argv = []
        command = " ".join(argv)
        executable_launch = any(Path(argument).name == "vllm" for argument in argv)
        module_launch = any(
            argument == "-m"
            and index + 1 < len(argv)
            and argv[index + 1].split(".", 1)[0] == "vllm"
            for index, argument in enumerate(argv)
        )
        configured_server = (
            (executable_launch or module_launch) and "serve" in argv and metadata_name in command
        )

        metadata_target = str(metadata_path)
        mapped_metadata = False
        try:
            for line in (process_dir / "maps").read_text().splitlines():
                fields = line.removesuffix(" (deleted)").split()
                if fields and fields[-1] == metadata_target:
                    mapped_metadata = True
                    break
        except (OSError, UnicodeDecodeError):
            pass
        if not mapped_metadata:
            try:
                for descriptor in (process_dir / "fd").iterdir():
                    try:
                        target = os.readlink(descriptor).removesuffix(" (deleted)")
                    except OSError:
                        continue
                    if target == metadata_target:
                        mapped_metadata = True
                        break
            except OSError:
                pass

        if configured_server or mapped_metadata:
            matches.append((int(process_dir.name), command.strip()))
    return sorted(matches)


def _backing_name(pool_ref: Any) -> str:
    return (
        f"/flex_kv_{pool_ref.metadata_version:016x}_{pool_ref.pool_id}_{pool_ref.pool_version:016x}"
    )


def _cleanup_with_runtime(
    runtime: Any,
    *,
    metadata_name: str,
    pool_names: tuple[str, ...] | None,
    max_pool_slots: int,
    legacy_cache_count: int,
    legacy_total_host_blocks: int,
    shm_root: Path,
) -> list[str]:
    current_config = runtime.SharedMetadataConfig(
        1,
        [],
        runtime.SharedMetadataCapacity(1, max_pool_slots, 1),
    )
    legacy_components = 2 * legacy_cache_count
    legacy_config = runtime.SharedMetadataConfig(
        legacy_components,
        [],
        runtime.SharedMetadataCapacity(
            2 * legacy_components,
            ceil(legacy_total_host_blocks / 2),
            1,
        ),
    )
    try:
        metadata = runtime.SharedMetadata.create_or_attach(metadata_name, current_config)
    except RuntimeError as error:
        if "fixed capacity configuration mismatch" not in str(error):
            raise
        metadata = runtime.SharedMetadata.create_or_attach(metadata_name, legacy_config)

    candidates = [f"{metadata_name}.data"]
    candidates.extend(
        f"{metadata_name}.{family}.c{cache_index}.{role}"
        for family in ("a", "b")
        for cache_index in range(legacy_cache_count)
        for role in ("k", "v")
    )
    if pool_names is not None:
        candidates.extend(pool_names)
    candidates = list(dict.fromkeys(candidates))
    registered = [pool for name in candidates if (pool := metadata.find_pool(name)) is not None]
    if metadata.pool_count() != len(registered):
        raise RuntimeError(
            f"metadata {metadata_name!r} contains pools outside the known demo layout; "
            "refusing partial cleanup"
        )

    backing_names = [_backing_name(pool.pool_ref) for pool in registered]
    for backing_name in backing_names:
        runtime.SharedHostPool.unlink_by_name(backing_name)

    remaining_backings = []
    for backing_name in backing_names:
        backing_path = shm_root / backing_name.lstrip("/")
        for path in (backing_path, backing_path.with_name(f"{backing_path.name}.ctl")):
            if path.exists():
                remaining_backings.append(str(path))
    if remaining_backings:
        raise RuntimeError("failed to unlink shared data pools: " + ", ".join(remaining_backings))

    runtime.SharedMetadata.unlink_by_name(metadata_name)
    if (shm_root / metadata_name).exists():
        raise RuntimeError(f"failed to unlink shared metadata: {shm_root / metadata_name}")
    return backing_names


def cleanup_shared_kv_demo(
    *,
    metadata_name: str = DEFAULT_METADATA_NAME,
    pool_names: tuple[str, ...] | None = None,
    component_count: int = DEFAULT_COMPONENT_COUNT,
    max_pool_slots: int = DEFAULT_MAX_POOL_SLOTS,
    legacy_cache_count: int = DEFAULT_LEGACY_CACHE_COUNT,
    legacy_total_host_blocks: int = DEFAULT_LEGACY_TOTAL_HOST_BLOCKS,
    proc_root: Path = Path("/proc"),
    shm_root: Path = Path("/dev/shm"),
    runtime: Any | None = None,
) -> list[str]:
    if not metadata_name or "/" in metadata_name:
        raise ValueError("metadata_name must be a non-empty POSIX shared-memory name")
    if (
        component_count <= 0
        or max_pool_slots <= 0
        or legacy_cache_count <= 0
        or legacy_total_host_blocks <= 0
    ):
        raise ValueError("pool geometry values must be positive")
    for pool_name in pool_names or ():
        if not pool_name or pool_name == "/" or "/" in pool_name.lstrip("/"):
            raise ValueError("pool_names must be non-empty POSIX shared-memory names")

    metadata_path = shm_root / metadata_name
    live = _live_vllm_processes(metadata_name, metadata_path, proc_root)
    if live:
        processes = ", ".join(f"PID {pid}" for pid, _ in live)
        raise RuntimeError(
            f"refusing to unlink {metadata_name!r} while it is used by {processes}; "
            "stop both demo servers first"
        )

    if not metadata_path.exists():
        return []

    cleanup_args = {
        "metadata_name": metadata_name,
        "pool_names": pool_names,
        "max_pool_slots": max_pool_slots,
        "legacy_cache_count": legacy_cache_count,
        "legacy_total_host_blocks": legacy_total_host_blocks,
        "shm_root": shm_root,
    }
    if runtime is not None:
        return _cleanup_with_runtime(runtime, **cleanup_args)

    previous_env = {name: os.environ.get(name) for name in MOCK_RUNTIME_ENV}
    os.environ.update(MOCK_RUNTIME_ENV)
    try:
        from spyre_inference.v1.kv_offload.shared_runtime import (
            load_shared_runtime,
        )

        return _cleanup_with_runtime(load_shared_runtime(), **cleanup_args)
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-name", default=DEFAULT_METADATA_NAME)
    parser.add_argument(
        "--component-count",
        type=int,
        default=DEFAULT_COMPONENT_COUNT,
    )
    parser.add_argument(
        "--max-pool-slots",
        type=int,
        default=DEFAULT_MAX_POOL_SLOTS,
    )
    parser.add_argument(
        "--legacy-cache-count",
        type=int,
        default=DEFAULT_LEGACY_CACHE_COUNT,
    )
    parser.add_argument(
        "--legacy-total-host-blocks",
        type=int,
        default=DEFAULT_LEGACY_TOTAL_HOST_BLOCKS,
    )
    parser.add_argument(
        "--pool-name",
        action="append",
        dest="pool_names",
        help="repeat for any additional demo-owned data pool",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    removed = cleanup_shared_kv_demo(
        metadata_name=args.metadata_name,
        pool_names=tuple(args.pool_names) if args.pool_names else None,
        component_count=args.component_count,
        max_pool_slots=args.max_pool_slots,
        legacy_cache_count=args.legacy_cache_count,
        legacy_total_host_blocks=args.legacy_total_host_blocks,
    )
    if removed:
        print(f"Removed {len(removed)} shared data pools and metadata {args.metadata_name!r}.")
    else:
        print(f"No shared metadata named {args.metadata_name!r} exists; nothing to clean.")


if __name__ == "__main__":
    main()
