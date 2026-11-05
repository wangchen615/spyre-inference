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

"""Opt-in large-arena and torchrun TP acceptance for persistent shared KV."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import socket
import subprocess
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from spyre_testing_plugin.pytest_plugin import spyre_device_count
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LookupResult, ReqContext, make_offload_key

from scripts.cleanup_shared_kv_demo import cleanup_pool
from scripts.shared_pool_delegator import SharedPoolCtl
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl
from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.pool_plan import parse_planned_pool
from spyre_inference.v1.kv_offload.pool_spec import planned_page_bytes
from spyre_inference.v1.kv_offload.shared_manager import SpyreSharedOffloadingManager
from spyre_inference.v1.kv_offload.shared_spec import component_manifest
from spyre_inference.v1.kv_offload.shared_types import (
    SharedBlockTransfer,
    SharedLoadStoreSpec,
)
from spyre_inference.v1.kv_offload.shared_worker import SpyreSharedOffloadingWorker
from tests.kv_offload.hw_helpers import _assert_bit_exact, _fill_all

MODEL = "spyre-shared-tp-hardware-test"
BLOCK_SIZE = 128
LOCAL_KV_HEADS = 4
HEAD_SIZE = 128
NUM_DEVICE_BLOCKS = 4
OFFLOAD_KEY = make_offload_key(bytes.fromhex("74" * 32), 0)
DELEGATOR_BUILD = Path(os.environ.get("DELEGATOR_BUILD_DIR", ""))
DELEGATOR = DELEGATOR_BUILD / "shared-pool-delegator"
POOLCTL = DELEGATOR_BUILD / "shared-poolctl"


def _model_spec(pool_name: str, tp_size: int, cpu_bytes: int) -> dict[str, object]:
    page_bytes = planned_page_bytes(
        torch.float16,
        BLOCK_SIZE,
        LOCAL_KV_HEADS,
        HEAD_SIZE,
        layout_kind="token-major",
    )
    return {
        "planningVersion": 1,
        "name": pool_name,
        "model": MODEL,
        "modelRevision": "",
        "kvDtype": "float16",
        "blockSize": str(BLOCK_SIZE),
        "tensorParallelSize": tp_size,
        "cpuBytesToUse": str(cpu_bytes),
        "rankCaches": [
            {
                "tpRank": rank,
                "cacheIndex": 0,
                "physicalPageBytes": str(page_bytes),
                "layoutKind": "token-major",
                "layoutVersion": 1,
                "localKvHeads": LOCAL_KV_HEADS,
                "headSize": HEAD_SIZE,
            }
            for rank in range(tp_size)
        ],
    }


def _large_offset_model_spec(pool_name: str, cpu_bytes: int) -> dict[str, object]:
    spec = _model_spec(pool_name, 2, cpu_bytes)
    spec["rankCaches"] = [
        {
            "tpRank": 0,
            "cacheIndex": 0,
            "physicalPageBytes": str(4 << 20),
            "layoutKind": "token-major",
            "layoutVersion": 1,
            "localKvHeads": LOCAL_KV_HEADS,
            "headSize": HEAD_SIZE,
        },
        {
            "tpRank": 0,
            "cacheIndex": 1,
            "physicalPageBytes": "4096",
            "layoutKind": "token-major",
            "layoutVersion": 1,
            "localKvHeads": LOCAL_KV_HEADS,
            "headSize": HEAD_SIZE,
        },
        {
            "tpRank": 1,
            "cacheIndex": 0,
            "physicalPageBytes": "4096",
            "layoutKind": "token-major",
            "layoutVersion": 1,
            "localKvHeads": LOCAL_KV_HEADS,
            "headSize": HEAD_SIZE,
        },
    ]
    return spec


def _manager(plan, tp_rank):
    return SpyreSharedOffloadingManager(
        plan=plan,
        tp_rank=tp_rank,
        cache_policy="lru",
        cache_policy_module_path=None,
        enable_events=False,
        store_threshold=0,
        max_tracker_size=64_000,
    )


def _start_delegator(root: Path) -> tuple[subprocess.Popen[str], SharedPoolCtl]:
    if not DELEGATOR.is_file() or not POOLCTL.is_file():
        pytest.skip("DELEGATOR_BUILD_DIR does not contain M4 delegator artifacts")
    socket_path = root / "delegator.sock"
    process = subprocess.Popen(
        [str(DELEGATOR), "--socket-path", str(socket_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ctl = SharedPoolCtl(binary=POOLCTL, socket_path=socket_path)
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError("shared-pool-delegator exited before becoming ready")
        health = subprocess.run(
            [str(POOLCTL), "--socket", str(socket_path), "health"],
            check=False,
            capture_output=True,
            text=True,
        )
        if health.returncode == 0:
            return process, ctl
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError("shared-pool-delegator did not become ready")


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _write_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        pickle.dump(value, target)
    temporary.replace(path)


def _wait_for(path: Path, process: subprocess.Popen[str], log_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise AssertionError(
                f"torchrun exited with {process.returncode}:\n"
                f"{log_path.read_text(encoding='utf-8', errors='replace')}"
            )
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {path}")


def _torchrun_command(tp_size: int, *worker_args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(tp_size),
        "--master-port",
        str(_free_port()),
        "--module",
        "tests.kv_offload.test_shared_tp_hw",
        "--rank-worker",
        *worker_args,
    ]


def test_torchrun_worker_uses_module_import_root() -> None:
    command = _torchrun_command(4, "peer", "/tmp/root")

    assert command[command.index("--module") + 1] == "tests.kv_offload.test_shared_tp_hw"
    assert command[command.index("--nproc-per-node") + 1] == "4"


def test_stop_process_terminates_live_child_and_reaps() -> None:
    calls = []

    class Process:
        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            return 0

    _stop_process(Process())

    assert calls == ["terminate", ("wait", 10)]


def _physical_cache(local_rank: int) -> tuple[object, SpyrePhysicalCaches]:
    torch.spyre.set_device(local_rank)
    torch.spyre._impl._lazy_init()
    spec = AttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=LOCAL_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )
    cache = SpyreAttentionImpl.allocate_pages(
        NUM_DEVICE_BLOCKS,
        spec,
        torch.device(f"spyre:{local_rank}"),
    )
    physical = SpyrePhysicalCaches(
        caches=((cache.k_pages, cache.v_pages),),
        layout_kinds=("token-major",),
        tensor_idx_to_cache={0: 0, 1: 0},
        num_blocks=NUM_DEVICE_BLOCKS,
    )
    return cache, physical


def _read_pickle(path: Path):
    with path.open("rb") as source:
        return pickle.load(source)


def _drain(worker: SpyreSharedOffloadingWorker):
    results = worker.get_finished()
    if len(results) != 1 or not results[0].success:
        raise AssertionError(f"shared worker transfer failed: {results}")
    return results[0]


def _rank_worker(arguments: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    root = Path(arguments.root)
    plan = parse_planned_pool(json.loads(Path(arguments.plan).read_text(encoding="utf-8")))
    if plan.tensor_parallel_size != world_size:
        raise AssertionError("torchrun world size disagrees with the immutable pool plan")

    torch.spyre.set_device(local_rank)
    dist.init_process_group(
        "cpu:gloo,spyre:spyreccl",
        timeout=timedelta(seconds=180),
    )
    workers: list[SpyreSharedOffloadingWorker] = []
    try:
        if arguments.phase == "store":
            source, source_physical = _physical_cache(local_rank)
            expected = _fill_all(source, {0: 100 + rank})[0]
            torch.save({"k": expected[0], "v": expected[1]}, root / f"expected-{rank}.pt")
            source_worker = SpyreSharedOffloadingWorker(
                physical=source_physical,
                manifest=component_manifest(source_physical),
                plan=plan,
                tp_rank=rank,
            )
            workers.append(source_worker)
            store_spec = _read_pickle(Path(arguments.store_spec))
            source_worker.submit_store(
                1,
                GPULoadStoreSpec([0], group_sizes=[1], block_indices=[0]),
                store_spec,
            )
            store_result = _drain(source_worker)
            dist.barrier()
            if rank == 0:
                (root / "store.done").touch()

            load_path = Path(arguments.load_spec)
            deadline = time.monotonic() + 180
            while not load_path.exists():
                if time.monotonic() >= deadline:
                    raise AssertionError("timed out waiting for same-replica load specification")
                time.sleep(0.05)
            destination, destination_physical = _physical_cache(local_rank)
            _fill_all(destination, {1: 900 + rank})
            destination_worker = SpyreSharedOffloadingWorker(
                physical=destination_physical,
                manifest=component_manifest(destination_physical),
                plan=plan,
                tp_rank=rank,
            )
            workers.append(destination_worker)
            destination_worker.submit_load(
                2,
                _read_pickle(load_path),
                GPULoadStoreSpec([1], group_sizes=[1], block_indices=[0]),
            )
            load_result = _drain(destination_worker)
            _assert_bit_exact(destination.k_pages[1], expected[0], "same-replica K")
            _assert_bit_exact(destination.v_pages[1], expected[1], "same-replica V")
            report = {
                "rank": rank,
                "storeBytes": store_result.transfer_size,
                "storeSeconds": store_result.transfer_time,
                "loadBytes": load_result.transfer_size,
                "loadSeconds": load_result.transfer_time,
            }
        else:
            destination, destination_physical = _physical_cache(local_rank)
            _fill_all(destination, {2: 1200 + rank})
            destination_worker = SpyreSharedOffloadingWorker(
                physical=destination_physical,
                manifest=component_manifest(destination_physical),
                plan=plan,
                tp_rank=rank,
            )
            workers.append(destination_worker)
            destination_worker.submit_load(
                3,
                _read_pickle(Path(arguments.load_spec)),
                GPULoadStoreSpec([2], group_sizes=[1], block_indices=[0]),
            )
            load_result = _drain(destination_worker)
            expected = torch.load(root / f"expected-{rank}.pt", weights_only=True)
            _assert_bit_exact(destination.k_pages[2], expected["k"], "peer-replica K")
            _assert_bit_exact(destination.v_pages[2], expected["v"], "peer-replica V")
            report = {
                "rank": rank,
                "loadBytes": load_result.transfer_size,
                "loadSeconds": load_result.transfer_time,
            }

        (root / f"{arguments.phase}-{rank}.json").write_text(
            json.dumps(report, sort_keys=True), encoding="utf-8"
        )
        dist.barrier()
    finally:
        for worker in reversed(workers):
            worker.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@pytest.mark.uses_subprocess
@pytest.mark.distributed
def test_torchrun_tp_shared_pool_round_trip(tmp_path: Path) -> None:
    if os.environ.get("SPYRE_RUN_SHARED_KV_TP_TEST") != "1":
        pytest.skip("set SPYRE_RUN_SHARED_KV_TP_TEST=1 for serial torchrun acceptance")
    if os.environ.get("FLEX_DEVICE", "").upper().startswith("MOCK"):
        pytest.skip("torchrun byte fidelity requires real Spyre cards")
    tp_size = int(os.environ.get("SPYRE_SHARED_KV_TP_SIZE", "2"))
    if tp_size <= 1:
        raise ValueError("SPYRE_SHARED_KV_TP_SIZE must be greater than one")
    if spyre_device_count() < tp_size:
        pytest.skip(f"TP{tp_size} needs at least {tp_size} real Spyre cards")

    process, ctl = _start_delegator(tmp_path)
    pool_name = f"/spyre-tp-hw-{os.getpid()}-{uuid.uuid4().hex}"
    plan_path = tmp_path / "planned-pool.json"
    managers_a = []
    manager_b = None
    store_process = peer_process = None
    generation = None
    try:
        page_bytes = planned_page_bytes(torch.float16, BLOCK_SIZE, LOCAL_KV_HEADS, HEAD_SIZE)
        document = ctl.create_model(_model_spec(pool_name, tp_size, tp_size * 2 * page_bytes * 4))
        plan_path.write_text(json.dumps(document), encoding="utf-8")
        plan = parse_planned_pool(document)
        generation = plan.identity.generation.hex()
        assert plan.tensor_parallel_size == tp_size
        assert len(plan.partitions) == tp_size

        # The controller attaches each rank on the mock runtime so it never
        # opens a card owned by a torchrun worker.
        original_flex_device = os.environ.get("FLEX_DEVICE")
        original_flex_compute = os.environ.get("FLEX_COMPUTE")
        os.environ["FLEX_DEVICE"] = "MOCK1p0"
        os.environ["FLEX_COMPUTE"] = "NULL"
        store_pages = []
        try:
            for rank in range(tp_size):
                manager = _manager(plan, rank)
                context = ReqContext(f"cold-store-{rank}")
                store = manager.prepare_store([OFFLOAD_KEY], context)
                assert store is not None and store.keys_to_store == [OFFLOAD_KEY]
                store_pages.extend(store.store_spec.transfers[0].pages)
                managers_a.append((manager, context))
        finally:
            if original_flex_device is None:
                os.environ.pop("FLEX_DEVICE", None)
            else:
                os.environ["FLEX_DEVICE"] = original_flex_device
            if original_flex_compute is None:
                os.environ.pop("FLEX_COMPUTE", None)
            else:
                os.environ["FLEX_COMPUTE"] = original_flex_compute
        assert len(store_pages) == 2 * tp_size
        store_spec = SharedLoadStoreSpec([SharedBlockTransfer(OFFLOAD_KEY, tuple(store_pages))])
        store_path = tmp_path / "store.pkl"
        same_load_path = tmp_path / "same-load.pkl"
        _write_pickle(store_path, store_spec)

        store_log = tmp_path / "store-torchrun.log"
        with store_log.open("w", encoding="utf-8") as output:
            store_process = subprocess.Popen(
                _torchrun_command(
                    tp_size,
                    "store",
                    str(tmp_path),
                    str(plan_path),
                    str(store_path),
                    str(same_load_path),
                ),
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
            )
        _wait_for(tmp_path / "store.done", store_process, store_log, 300)
        for manager, context in managers_a:
            manager.complete_store([OFFLOAD_KEY], context, success=True)
        manager_a = managers_a[0][0]
        same_context = ReqContext("same-replica-load")
        assert manager_a.lookup(OFFLOAD_KEY, same_context) is LookupResult.HIT
        same_load = manager_a.prepare_load([OFFLOAD_KEY], same_context)
        _write_pickle(same_load_path, same_load)
        assert store_process.wait(timeout=300) == 0, store_log.read_text(
            encoding="utf-8", errors="replace"
        )
        manager_a.complete_load([OFFLOAD_KEY], same_context)

        manager_b = _manager(plan, 0)
        peer_context = ReqContext("peer-replica-load")
        assert manager_b.lookup(OFFLOAD_KEY, peer_context) is LookupResult.HIT
        peer_load_path = tmp_path / "peer-load.pkl"
        _write_pickle(peer_load_path, manager_b.prepare_load([OFFLOAD_KEY], peer_context))
        peer_log = tmp_path / "peer-torchrun.log"
        with peer_log.open("w", encoding="utf-8") as output:
            peer_process = subprocess.Popen(
                _torchrun_command(
                    tp_size,
                    "peer",
                    str(tmp_path),
                    str(plan_path),
                    "-",
                    str(peer_load_path),
                ),
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
            )
        assert peer_process.wait(timeout=300) == 0, peer_log.read_text(
            encoding="utf-8", errors="replace"
        )
        manager_b.complete_load([OFFLOAD_KEY], peer_context)

        store_reports = [
            json.loads((tmp_path / f"store-{rank}.json").read_text()) for rank in range(tp_size)
        ]
        peer_reports = [
            json.loads((tmp_path / f"peer-{rank}.json").read_text()) for rank in range(tp_size)
        ]
        evidence = {
            "tpSize": tp_size,
            "generation": generation,
            "globalComponents": len(plan.components),
            "allocatedBytes": plan.allocated_bytes,
            "storeBytes": sum(item["storeBytes"] for item in store_reports),
            "sameReplicaLoadBytes": sum(item["loadBytes"] for item in store_reports),
            "peerReplicaLoadBytes": sum(item["loadBytes"] for item in peer_reports),
            "storeCopySeconds": max(item["storeSeconds"] for item in store_reports),
            "sameReplicaCopySeconds": max(item["loadSeconds"] for item in store_reports),
            "peerReplicaCopySeconds": max(item["loadSeconds"] for item in peer_reports),
        }
        print(f"SPYRE_SHARED_KV_TP_RESULT={json.dumps(evidence, sort_keys=True)}")
    finally:
        _stop_process(peer_process)
        _stop_process(store_process)
        try:
            if manager_b is not None:
                manager_b.shutdown()
        finally:
            try:
                for manager, _ in managers_a:
                    manager.shutdown()
            finally:
                try:
                    if generation is not None:
                        cleanup_pool(pool_name, generation, ctl=ctl)
                finally:
                    _stop_process(process)


@pytest.mark.uses_subprocess
def test_large_partitioned_arena_preserves_64_bit_offsets_and_edge_slots(
    tmp_path: Path,
) -> None:
    if os.environ.get("FLEX_RUN_LARGE_SHARED_POOL_TEST") != "1":
        pytest.skip("set FLEX_RUN_LARGE_SHARED_POOL_TEST=1 for >4-GiB acceptance")
    if os.environ.get("FLEX_DEVICE", "").upper().startswith("MOCK"):
        pytest.skip("large edge-slot copy requires a real Spyre card")
    if spyre_device_count() < 1:
        pytest.skip("large edge-slot copy requires one real Spyre card")

    process, ctl = _start_delegator(tmp_path)
    pool_name = f"/spyre-large-hw-{os.getpid()}-{uuid.uuid4().hex}"
    generation = None
    attachments = []
    by_name = {}
    pool = None
    pools = []
    try:
        budget = (4 << 30) + (8 << 20)
        document = ctl.create_model(_large_offset_model_spec(pool_name, budget))
        plan = parse_planned_pool(document)
        generation = plan.identity.generation.hex()
        assert plan.allocated_bytes > 4 << 30
        assert plan.allocated_bytes <= budget
        rank_zero = plan.partitions_for_rank(0)
        assert len(rank_zero) == 2
        assert rank_zero[-1].offset_bytes >= 4 << 30

        from torch_spyre import _C

        opened = _C.SharedPool.open_pool(pool_name)
        assert opened.code is _C.PoolResultCode.OK
        native_arenas = opened.descriptor.effective_config.data_pools
        assert len(native_arenas) == len(plan.arenas)
        for native, arena in zip(native_arenas, plan.arenas, strict=True):
            assert native.name == arena.name
            assert [item.offset_bytes for item in native.partitions] == [
                item.offset_bytes for item in arena.partitions
            ]
            assert [item.num_slots for item in native.partitions] == [
                item.num_slots for item in arena.partitions
            ]

        attached = _C.SharedPool.attach_pool(
            pool_name,
            plan.identity.generation,
            0,
        )
        assert attached.code is _C.PoolResultCode.OK
        attachment = attached.attachment
        assert attachment is not None
        by_name = {item.registration.name: item for item in attachment.data_pools}
        pools.extend(by_name[partition.partition_name].data_pool for partition in rank_zero)
        attachments.append(attachment)

        torch.spyre.set_device(0)
        torch.spyre._impl._lazy_init()
        for index, (partition, pool) in enumerate(zip(rank_zero, pools, strict=True)):
            slot = 0 if index == 0 else partition.num_slots - 1
            source = torch.arange(64, dtype=torch.float16, device="spyre") + 1000 * index
            destination = torch.empty_like(source)
            _C.copy_tensor_raw(source, pool, slot, False)
            _C.copy_tensor_raw(destination, pool, slot, True)
            assert torch.equal(source.cpu(), destination.cpu())
    finally:
        try:
            pools.clear()
            pool = None
            by_name.clear()
            from torch_spyre import _C

            for attachment in reversed(attachments):
                _C.SharedPool.detach_pool(attachment)
            attachments.clear()
            gc.collect()
            if generation is not None:
                cleanup_pool(pool_name, generation, ctl=ctl)
        finally:
            _stop_process(process)


def _parse_rank_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-worker", action="store_true")
    parser.add_argument("phase", choices=("store", "peer"))
    parser.add_argument("root")
    parser.add_argument("plan")
    parser.add_argument("store_spec")
    parser.add_argument("load_spec")
    return parser.parse_args()


if __name__ == "__main__":
    _rank_worker(_parse_rank_args())
