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

"""Two-instance functional acceptance for shared Spyre KV offloading."""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count

from scripts.cleanup_shared_kv_demo import cleanup_pool
from scripts.shared_kv_two_instance_demo import PreparedDemoPool, prepare_demo_pool
from scripts.shared_pool_delegator import SharedPoolCtl

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
PROMPT_TOKEN_IDS = tuple(100 + (index * 37) % 1000 for index in range(320))
TOKENS_PER_BLOCK = 128
MODEL_LAYER_COUNT = 4
MAX_COMPONENTS = 2 * MODEL_LAYER_COUNT
CPU_BYTES = 512 * 1024 * 1024
METRIC_NAMES = (
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_load_time_total",
    "vllm:kv_offload_store_bytes_total",
)
LOAD_BYTES = METRIC_NAMES[0]
LOAD_TIME = METRIC_NAMES[1]
STORE_BYTES = METRIC_NAMES[2]
DELEGATOR_BUILD_DIR = Path(os.environ.get("DELEGATOR_BUILD_DIR", ""))
DELEGATOR = DELEGATOR_BUILD_DIR / "shared-pool-delegator"
POOLCTL = DELEGATOR_BUILD_DIR / "shared-poolctl"

pytestmark = [
    pytest.mark.uses_subprocess,
    pytest.mark.skipif(
        os.environ.get("RUN_SPYRE_SHARED_KV_E2E") != "1",
        reason="set RUN_SPYRE_SHARED_KV_E2E=1 to run the two-instance acceptance",
    ),
    pytest.mark.skipif(
        spyre_device_count() < 2,
        reason="needs at least two real Spyre cards",
    ),
    pytest.mark.skipif(
        os.environ.get("FLEX_DEVICE", "").upper().startswith("MOCK"),
        reason="cross-instance byte fidelity requires real Spyre cards",
    ),
    pytest.mark.skipif(
        not DELEGATOR.is_file() or not POOLCTL.is_file(),
        reason="DELEGATOR_BUILD_DIR must contain shared-pool-delegator and shared-poolctl",
    ),
]


@dataclass(frozen=True)
class _Completion:
    text: str
    token_ids: tuple[int, ...]


class _Server:
    def __init__(
        self,
        *,
        name: str,
        port: int,
        device: int,
        pool_name: str,
        pool_generation: str,
        data_pool_name: str,
        log_path: Path,
    ) -> None:
        self.name = name
        self.port = port
        self.log_path = log_path
        self._log_stream = log_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            _server_command(port, pool_name, pool_generation, data_pool_name),
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "PYTHONHASHSEED": "0",
                "SPYRE_DEVICES": str(device),
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "VLLM_LOGGING_LEVEL": "DEBUG",
            },
            start_new_session=True,
        )
        self._process_group = self._process.pid

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_until_ready(self, timeout: float = 360.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise AssertionError(
                    f"server {self.name} exited with {self._process.returncode}:\n{self.log_tail()}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.25)
        raise AssertionError(f"server {self.name} did not become healthy:\n{self.log_tail()}")

    def stop(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self._process_group, signal.SIGTERM)
        try:
            self._process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._process_group, signal.SIGKILL)
            self._process.wait(timeout=30)
        self._log_stream.close()

    def assert_alive(self) -> None:
        if self._process.poll() is not None:
            raise AssertionError(
                f"server {self.name} exited with {self._process.returncode}:\n{self.log_tail()}"
            )

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def log_tail(self, lines: int = 80) -> str:
        return "\n".join(self.log_text().splitlines()[-lines:])


def _server_command(
    port: int,
    pool_name: str,
    pool_generation: str,
    data_pool_name: str,
) -> list[str]:
    kv_transfer_config = {
        "kv_connector": "SpyreOffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "spyre_inference.v1.kv_offload.connector",
        "kv_connector_extra_config": {
            "spec_name": "SpyreSharedOffloadingSpec",
            "pool_name": pool_name,
            "pool_generation": pool_generation,
            "data_pool_name": data_pool_name,
            "cpu_bytes_to_use": CPU_BYTES,
        },
    }
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--tensor-parallel-size",
        "1",
        "--distributed-executor-backend",
        "uni",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--kv-transfer-config",
        json.dumps(kv_transfer_config, separators=(",", ":")),
    ]


def _free_ports(count: int) -> tuple[int, ...]:
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return tuple(sock.getsockname()[1] for sock in sockets)
    finally:
        for sock in sockets:
            sock.close()


def _get_text(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def _metric_snapshot(server: _Server) -> dict[str, float]:
    totals = dict.fromkeys(METRIC_NAMES, 0.0)
    for line in _get_text(f"{server.base_url}/metrics").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        sample_name = fields[0].split("{", 1)[0]
        if sample_name in totals:
            totals[sample_name] += float(fields[1])
    return totals


def _wait_for_positive_deltas(
    server: _Server,
    before: dict[str, float],
    names: tuple[str, ...],
    timeout: float = 30.0,
) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    last = dict.fromkeys(names, 0.0)
    while time.monotonic() < deadline:
        server.assert_alive()
        after = _metric_snapshot(server)
        last = {name: after[name] - before[name] for name in names}
        if all(value > 0 for value in last.values()):
            return last
        time.sleep(0.2)
    raise AssertionError(
        f"server {server.name} completed generation but did not report positive "
        f"offload deltas for {names}: {last}"
    )


def _completion(server: _Server) -> _Completion:
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": list(PROMPT_TOKEN_IDS),
            "add_special_tokens": False,
            "temperature": 0.0,
            "seed": 17,
            "ignore_eos": True,
            "max_tokens": 8,
            "return_token_ids": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{server.base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise AssertionError(
            f"server {server.name} rejected completion ({error.code}): {detail}"
        ) from error
    choice = body["choices"][0]
    assert isinstance(choice["text"], str)
    assert isinstance(choice["token_ids"], list)
    assert all(isinstance(token_id, int) for token_id in choice["token_ids"])
    return _Completion(choice["text"], tuple(choice["token_ids"]))


def _wait_for_log(server: _Server, pattern: re.Pattern[str], timeout: float = 15.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = server.log_text()
        if pattern.search(text):
            return text
        server.assert_alive()
        time.sleep(0.2)
    raise AssertionError(
        f"server {server.name} log did not match {pattern.pattern!r}:\n{server.log_tail()}"
    )


def _host_block_count(server: _Server) -> int | None:
    match = re.search(r"SpyreOffloadingSpec: (\d+) host block", server.log_text())
    return int(match.group(1)) if match else None


def _assert_one_pool_topology(
    server: _Server,
    pool_name: str,
    pool_generation: str,
    data_pool_name: str,
    num_blocks: int,
) -> None:
    inspection = r"""
import json
import sys
from torch_spyre._C import PoolResultCode, SharedPool

pool_name, generation_hex, data_pool_name = sys.argv[1:4]
attached = SharedPool.attach_pool(pool_name, bytes.fromhex(generation_hex))
if attached.code is not PoolResultCode.OK:
    raise RuntimeError(f"attach failed: {attached.code}: {attached.detail}")
directory = attached.metadata
try:
    registered = directory.find_pool(data_pool_name)
    if registered is None:
        raise RuntimeError(f"pool {data_pool_name!r} is not registered")
    pool = directory.resolve_pool(registered.pool_ref)
    if pool is None:
        raise RuntimeError(f"pool {data_pool_name!r} cannot be resolved")
    print(json.dumps({
        "pool_count": directory.pool_count(),
        "slot_count": registered.slot_count,
        "slot_bytes": registered.slot_bytes,
        "resolved_slot_count": pool.slot_count(),
        "resolved_slot_bytes": pool.slot_bytes(),
        "total_bytes": pool.total_bytes(),
    }))
finally:
    SharedPool.detach_pool(directory)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            inspection,
            pool_name,
            pool_generation,
            data_pool_name,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "FLEX_DEVICE": "MOCK1p0",
            "FLEX_COMPUTE": "NULL",
            "AIU_WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        },
    )
    topology = json.loads(result.stdout.strip().splitlines()[-1])
    assert topology["pool_count"] == 1

    pattern = re.compile(
        rf"Spyre shared KV pool {re.escape(pool_name)}/{re.escape(data_pool_name)}: "
        r"(?P<components>\d+) components, (?P<logical>\d+) logical blocks, "
        r"(?P<slots>\d+) slots, (?P<slot_bytes>\d+) bytes/slot, "
        r"(?P<actual>\d+) bytes actual"
    )
    match = pattern.search(server.log_text())
    assert match is not None, server.log_tail()
    assert int(match["components"]) == MAX_COMPONENTS
    assert int(match["logical"]) == num_blocks
    assert topology["slot_count"] == int(match["slots"])
    assert topology["slot_bytes"] == int(match["slot_bytes"])
    assert topology["resolved_slot_count"] == topology["slot_count"]
    assert topology["resolved_slot_bytes"] == topology["slot_bytes"]
    assert topology["total_bytes"] == int(match["actual"])


class _Delegator:
    def __init__(self, root: Path) -> None:
        self.socket = root / "shared-pool-delegator.sock"
        self.process = subprocess.Popen(
            [str(DELEGATOR), "--socket-path", str(self.socket)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.ctl = SharedPoolCtl(binary=POOLCTL, socket_path=self.socket)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("shared-pool-delegator exited before becoming ready")
            health = subprocess.run(
                [str(POOLCTL), "--socket", str(self.socket), "health"],
                check=False,
                capture_output=True,
                text=True,
            )
            if health.returncode == 0:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("shared-pool-delegator did not become ready")

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)


def test_two_instances_reload_from_one_shared_pool(tmp_path: Path) -> None:
    unique = f"/spyre-m4-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    isolate_b = os.environ.get("SPYRE_SHARED_KV_ISOLATE_B") == "1"
    ports = _free_ports(2)
    servers: list[_Server] = []
    prepared: list[PreparedDemoPool] = []
    num_blocks: int | None = None
    delegator = _Delegator(tmp_path)

    try:
        pool_a = prepare_demo_pool(
            ctl=delegator.ctl,
            pool_name=unique,
            data_pool_name="host-0",
            model=MODEL,
            cpu_bytes_to_use=CPU_BYTES,
        )
        prepared.append(pool_a)
        if isolate_b:
            pool_b = prepare_demo_pool(
                ctl=delegator.ctl,
                pool_name=f"{unique}-isolated",
                data_pool_name="host-0",
                model=MODEL,
                cpu_bytes_to_use=CPU_BYTES,
            )
            prepared.append(pool_b)
        else:
            pool_b = pool_a

        server_a = _Server(
            name="A",
            port=ports[0],
            device=0,
            pool_name=pool_a.pool_name,
            pool_generation=pool_a.generation,
            data_pool_name=pool_a.data_pool_name,
            log_path=tmp_path / "server-a.log",
        )
        servers.append(server_a)
        server_a.wait_until_ready()

        server_b = _Server(
            name="B",
            port=ports[1],
            device=1,
            pool_name=pool_b.pool_name,
            pool_generation=pool_b.generation,
            data_pool_name=pool_b.data_pool_name,
            log_path=tmp_path / "server-b.log",
        )
        servers.append(server_b)
        server_b.wait_until_ready()
        num_blocks = _host_block_count(server_a)
        assert num_blocks is not None

        store_before = _metric_snapshot(server_a)  # A computes and publishes.
        baseline = _completion(server_a)
        store = _wait_for_positive_deltas(server_a, store_before, (STORE_BYTES,))
        _assert_one_pool_topology(
            server_a,
            pool_a.pool_name,
            pool_a.generation,
            pool_a.data_pool_name,
            num_blocks,
        )

        a_before = _metric_snapshot(server_a)  # A reloads its released blocks.
        a_reload = _completion(server_a)
        a_load = _wait_for_positive_deltas(server_a, a_before, (LOAD_BYTES, LOAD_TIME))

        b_before = _metric_snapshot(server_b)  # B's first request must be a peer hit.
        b_reload = _completion(server_b)
        b_load = _wait_for_positive_deltas(server_b, b_before, (LOAD_BYTES, LOAD_TIME))

        assert a_reload.token_ids == baseline.token_ids
        assert a_reload.text.encode() == baseline.text.encode()
        assert b_reload.token_ids == baseline.token_ids
        assert b_reload.text.encode() == baseline.text.encode()

        hit_pattern = re.compile(r"hit [1-9]\d* offloaded tokens after 0 GPU hit tokens")
        logs = {server.name: _wait_for_log(server, hit_pattern) for server in servers}
        for name, log_text in logs.items():
            assert re.search(r"enable_prefix_caching['\" :=]+False", log_text), name
            assert "SpyreSharedOffloadingSpec" in log_text, name
            assert not re.search(r"disk[-_ ]tier.*(?:hit|load)", log_text, re.IGNORECASE), name

        complete_blocks = len(PROMPT_TOKEN_IDS) // TOKENS_PER_BLOCK
        evidence = {
            "prompt_tokens": len(PROMPT_TOKEN_IDS),
            "complete_shared_blocks": complete_blocks,
            "a_store_bytes": store[STORE_BYTES],
            "a_self_reload_bytes": a_load[LOAD_BYTES],
            "a_self_reload_seconds": a_load[LOAD_TIME],
            "a_self_reload_bytes_per_second": a_load[LOAD_BYTES] / a_load[LOAD_TIME],
            "b_peer_reload_bytes": b_load[LOAD_BYTES],
            "b_peer_reload_seconds": b_load[LOAD_TIME],
            "b_peer_reload_bytes_per_second": b_load[LOAD_BYTES] / b_load[LOAD_TIME],
        }
        print(f"SPYRE_SHARED_KV_E2E_RESULT={json.dumps(evidence, sort_keys=True)}")
    finally:
        for server in reversed(servers):
            server.stop()
        try:
            for pool in reversed(prepared):
                cleanup_pool(pool.pool_name, pool.generation, ctl=delegator.ctl)
        finally:
            delegator.close()
