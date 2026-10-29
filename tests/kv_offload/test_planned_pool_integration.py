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

"""Real delegator-to-worker proof for one partitioned TP host arena."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key

from spyre_inference.v1.kv_offload.pool_plan import parse_planned_pool
from spyre_inference.v1.kv_offload.shared_manager import SpyreSharedOffloadingManager

DELEGATOR_BUILD = Path(
    os.environ.get(
        "DELEGATOR_BUILD_DIR",
        "/home/yzhu/dt-inductor/shared-pool-delegator/.worktrees/kvc-offload-m4/build",
    )
)
DAEMON = DELEGATOR_BUILD / "shared-pool-delegator"
CTL = DELEGATOR_BUILD / "shared-poolctl"
TORCH_SPYRE = Path("/home/yzhu/dt-inductor/torch-spyre/.worktrees/kvc-offload-m4")
FLEX_BUILD = Path("/home/yzhu/dt-inductor/flex-worktrees/persistent-shared-kv-pool-lifecycle/build")
OFFLOAD_KEY = make_offload_key(bytes.fromhex("55" * 32), 0)


def _run_ctl(socket: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(CTL), "--socket", str(socket), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _start_daemon(socket: Path) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [str(DAEMON), "--socket-path", str(socket)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(200):
        if process.poll() is not None:
            raise RuntimeError("delegator exited before publishing its socket")
        if socket.is_socket() and _run_ctl(socket, "health").returncode == 0:
            return process
        time.sleep(0.02)
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError("delegator did not become healthy")


def _model_spec(pool_name: str) -> dict[str, object]:
    cache_counts = (1, 2, 2, 1)
    rows = []
    for rank, cache_count in enumerate(cache_counts):
        for cache_index in range(cache_count):
            rows.append(
                {
                    "tpRank": rank,
                    "cacheIndex": cache_index,
                    "physicalPageBytes": str(3000 + 2000 * cache_index),
                    "layoutKind": "token-major",
                    "layoutVersion": 1,
                    "localKvHeads": 4 + cache_index,
                    "headSize": 128,
                }
            )
    return {
        "planningVersion": 1,
        "name": pool_name,
        "model": "host-integration-model",
        "modelRevision": "revision-1",
        "kvDtype": "float16",
        "blockSize": "128",
        "tensorParallelSize": 4,
        "cpuBytesToUse": str(
            2 * sum(2 * (4096 if row["cacheIndex"] == 0 else 8192) for row in rows)
        ),
        "rankCaches": rows,
    }


def _manager(plan):
    return SpyreSharedOffloadingManager(
        plan=plan,
        cache_policy="lru",
        cache_policy_module_path=None,
        enable_events=False,
        store_threshold=0,
        max_tracker_size=64_000,
    )


def _worker_program() -> str:
    return """
import json
import pickle
import sys

from torch_spyre import _C
from spyre_inference.v1.kv_offload.pool_plan import parse_planned_pool
from spyre_inference.v1.kv_offload.shared_attachment import SharedMetadataAttachment

document_path, token_path, rank_text = sys.argv[1:]
with open(document_path, encoding="utf-8") as source:
    plan = parse_planned_pool(json.load(source))
with open(token_path, "rb") as source:
    tokens = pickle.load(source)
rank = int(rank_text)
attachment = SharedMetadataAttachment.attach(plan)
try:
    resolved = attachment.resolve_rank(rank)
    assert [item.partition.partition_name for item in resolved] == [
        item.partition_name for item in plan.partitions_for_rank(rank)
    ]
    for token, size in tokens:
        attachment.metadata.publish(token, [_C.ChunkDescriptorEntry(rank, size)])
    print("READY", flush=True)
    assert sys.stdin.readline().strip() == "DETACH"
finally:
    attachment.close()
"""


def _worker_environment() -> dict[str, str]:
    python_path = os.pathsep.join(
        [str(Path.cwd()), str(TORCH_SPYRE), os.environ.get("PYTHONPATH", "")]
    )
    library_path = os.pathsep.join([str(FLEX_BUILD), os.environ.get("LD_LIBRARY_PATH", "")])
    return {
        **os.environ,
        "PYTHONPATH": python_path,
        "LD_LIBRARY_PATH": library_path,
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "FLEX_DEVICE": "MOCK1p0",
        "FLEX_COMPUTE": "NULL",
        "AIU_WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
    }


def test_delegator_created_tp4_arena_supports_global_publish_and_rank_local_attach(
    tmp_path,
):
    if not DAEMON.is_file() or not CTL.is_file():
        pytest.skip("delegator M4 binaries are not built")

    socket = tmp_path / "delegator.sock"
    pool_name = f"/planned-integration-{os.getpid()}-{uuid.uuid4().hex}"
    spec_path = tmp_path / "model-pool.json"
    spec_path.write_text(json.dumps(_model_spec(pool_name)))
    document_path = tmp_path / "planned-pool.json"
    daemon = _start_daemon(socket)
    workers: list[subprocess.Popen[str]] = []
    first_manager = None
    second_manager = None
    current_generation = None
    try:
        created = _run_ctl(socket, "create-model", "--spec", str(spec_path))
        assert created.returncode == 0, created.stderr
        document = json.loads(created.stdout)
        document_path.write_text(json.dumps(document))
        plan = parse_planned_pool(document)
        current_generation = plan.identity.generation.hex()

        assert len(document["plan"]["config"]["dataPools"]) == 1
        assert plan.arena.name == "host-0"
        assert len(plan.partitions) == 6
        assert {item.slot_bytes for item in plan.partitions} == {4096, 8192}
        assert [item.offset_bytes for item in plan.partitions] == sorted(
            item.offset_bytes for item in plan.partitions
        )

        first_manager = _manager(plan)
        store_context = ReqContext("store")
        prepared = first_manager.prepare_store([OFFLOAD_KEY], store_context)
        assert prepared is not None
        pages = prepared.store_spec.transfers[0].pages
        assert len(pages) == len(plan.components)

        worker_env = _worker_environment()
        for rank in range(plan.tensor_parallel_size):
            local_ids = {item.global_component_id for item in plan.components_for_rank(rank)}
            token_path = tmp_path / f"rank-{rank}-tokens.pkl"
            with token_path.open("wb") as target:
                pickle.dump(
                    [
                        (page.reservation, plan.components[page.component_id].physical_page_bytes)
                        for page in pages
                        if page.component_id in local_ids
                    ],
                    target,
                )
            workers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _worker_program(),
                        str(document_path),
                        str(token_path),
                        str(rank),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=worker_env,
                )
            )

        for worker in workers:
            assert worker.stdout is not None and worker.stderr is not None
            assert worker.stdout.readline().strip() == "READY", worker.stderr.read()

        first_manager.complete_store([OFFLOAD_KEY], store_context, success=True)
        second_manager = _manager(plan)
        load_context = ReqContext("load")
        assert second_manager.lookup(OFFLOAD_KEY, load_context) is LookupResult.HIT
        second_manager.on_request_finished(load_context)

        busy = _run_ctl(
            socket,
            "destroy",
            "--name",
            pool_name,
            "--generation",
            current_generation,
        )
        assert json.loads(busy.stdout)["result"] == "POOL_BUSY"

        for worker in workers:
            assert worker.stdin is not None and worker.stderr is not None
            worker.stdin.write("DETACH\n")
            worker.stdin.flush()
            assert worker.wait(timeout=20) == 0, worker.stderr.read()
        workers.clear()
        second_manager.shutdown()
        second_manager = None
        first_manager.shutdown()
        first_manager = None

        destroyed = _run_ctl(
            socket,
            "destroy",
            "--name",
            pool_name,
            "--generation",
            current_generation,
        )
        assert json.loads(destroyed.stdout)["result"] == "OK"

        recreated = _run_ctl(socket, "create-model", "--spec", str(spec_path))
        replacement = parse_planned_pool(json.loads(recreated.stdout))
        assert replacement.identity.generation != plan.identity.generation
        current_generation = replacement.identity.generation.hex()

        from torch_spyre import _C

        stale = _C.SharedPool.attach_pool(pool_name, plan.identity.generation)
        assert stale.code is _C.PoolResultCode.STALE_GENERATION
        assert stale.metadata is None
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)
        if second_manager is not None:
            second_manager.shutdown()
        if first_manager is not None:
            first_manager.shutdown()
        if current_generation is not None:
            _run_ctl(
                socket,
                "destroy",
                "--name",
                pool_name,
                "--generation",
                current_generation,
            )
        daemon.terminate()
        daemon.wait(timeout=10)
