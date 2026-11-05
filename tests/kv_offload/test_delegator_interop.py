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

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from scripts.cleanup_shared_kv_demo import cleanup_pool
from scripts.shared_kv_two_instance_demo import prepare_demo_pool
from scripts.shared_pool_delegator import SharedPoolCtl

GENERATION = "ab" * 16


def test_ctl_create_model_uses_versioned_planning_command(monkeypatch):
    calls = []
    ctl = SharedPoolCtl(binary="shared-poolctl", socket_path="/tmp/delegator.sock")

    def run(*args):
        calls.append(args)
        spec_path = Path(args[-1])
        assert json.loads(spec_path.read_text()) == {"planningVersion": 1}
        return {"code": "OK"}

    monkeypatch.setattr(ctl, "_run", run)

    assert ctl.create_model({"planningVersion": 1}) == {"code": "OK"}
    assert calls[0][0:2] == ("create-model", "--spec")


class FakeCtl:
    def __init__(self):
        self.calls = []

    def create(self, config):
        self.calls.append(("create", config))
        return {
            "result": "OK",
            "descriptor": {
                "name": config["name"],
                "generation": GENERATION,
                "state": "READY",
                "effectiveConfig": config,
            },
        }

    def destroy(self, name, generation):
        self.calls.append(("destroy", name, generation))
        return {"result": "OK"}


def test_demo_passes_exact_delegator_identity_to_both_instances():
    ctl = FakeCtl()

    result = prepare_demo_pool(
        ctl=ctl,
        pool_name="deployment-root",
        data_pool_name="host-0",
        model="granite-test",
        cpu_bytes_to_use=512 * 1024 * 1024,
    )

    assert result.extra_config == {
        "spec_name": "SpyreSharedOffloadingSpec",
        "pool_name": "deployment-root",
        "pool_generation": GENERATION,
        "data_pool_name": "host-0",
        "cpu_bytes_to_use": 512 * 1024 * 1024,
    }
    assert result.instance_a_extra_config == result.instance_b_extra_config
    [(_, config)] = ctl.calls
    assert config["metadata"] == {"maxChunks": 1}
    assert config["dataPools"] == [
        {
            "name": "host-0",
            "backend": "HOST_DRAM",
            "numSlots": "2048",
            "slotBytes": "262144",
            "compatibility": {
                "model": "granite-test",
                "kvDtype": "float16",
                "blockSize": "128",
                "tensorParallelSize": 1,
            },
        }
    ]


def test_existing_name_with_different_effective_config_is_rejected():
    class MismatchedCtl(FakeCtl):
        def create(self, config):
            result = super().create(config)
            result["result"] = "ALREADY_EXISTS"
            result["descriptor"]["effectiveConfig"] = {
                **config,
                "dataPools": [
                    {
                        **config["dataPools"][0],
                        "compatibility": {
                            **config["dataPools"][0]["compatibility"],
                            "model": "another-model",
                        },
                    }
                ],
            }
            return result

    with pytest.raises(RuntimeError, match="effective configuration"):
        prepare_demo_pool(
            ctl=MismatchedCtl(),
            pool_name="deployment-root",
            data_pool_name="host-0",
            model="granite-test",
            cpu_bytes_to_use=512 * 1024 * 1024,
        )


def test_cleanup_uses_generation_checked_destroy():
    ctl = FakeCtl()

    result = cleanup_pool("deployment-root", GENERATION, ctl=ctl)

    assert result == {"result": "OK"}
    assert ctl.calls == [("destroy", "deployment-root", GENERATION)]


@pytest.mark.parametrize("code", ["POOL_BUSY", "STALE_GENERATION", "NOT_FOUND"])
def test_cleanup_does_not_treat_non_ok_lifecycle_result_as_success(code):
    class FailingCtl(FakeCtl):
        def destroy(self, name, generation):
            self.calls.append(("destroy", name, generation))
            return {"result": code, "detail": "structured failure"}

    with pytest.raises(RuntimeError, match=code):
        cleanup_pool("deployment-root", GENERATION, ctl=FailingCtl())


def _external_artifacts():
    build = Path(os.environ.get("DELEGATOR_BUILD_DIR", ""))
    daemon = build / "shared-pool-delegator"
    ctl = build / "shared-poolctl"
    return build, daemon, ctl


def _wait_for_daemon(ctl: Path, socket: Path, process: subprocess.Popen) -> None:
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("shared-pool-delegator exited before becoming ready")
        result = subprocess.run(
            [str(ctl), "--socket", str(socket), "health"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        time.sleep(0.05)
    raise RuntimeError("shared-pool-delegator did not become ready")


def test_external_delegator_reports_busy_until_torch_spyre_detaches(tmp_path):
    _, daemon, ctl_binary = _external_artifacts()
    if not daemon.is_file() or not ctl_binary.is_file():
        pytest.skip("DELEGATOR_BUILD_DIR does not contain shared-pool-delegator and shared-poolctl")

    from scripts.shared_pool_delegator import SharedPoolCtl

    socket = tmp_path / "delegator.sock"
    process = subprocess.Popen(
        [str(daemon), "--socket-path", str(socket)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    prepared = None
    attached_process = None
    try:
        _wait_for_daemon(ctl_binary, socket, process)
        ctl = SharedPoolCtl(binary=ctl_binary, socket_path=socket)
        pool_name = f"/spyre-inference-{os.getpid()}-{uuid.uuid4().hex}"
        prepared = prepare_demo_pool(
            ctl=ctl,
            pool_name=pool_name,
            data_pool_name="host-0",
            model="interop-test",
            cpu_bytes_to_use=8192,
            slot_bytes=4096,
        )
        worker = """
import sys
from torch_spyre import _C
name, generation_hex, child = sys.argv[1:]
attached = _C.SharedPool.attach_pool(name, bytes.fromhex(generation_hex), 0)
assert attached.code is _C.PoolResultCode.OK, (attached.code, attached.detail)
attachment = attached.attachment
assert attachment is not None
by_name = {item.registration.name: item for item in attachment.data_pools}
assert child in by_name
assert by_name[child].data_pool is not None
print("attached", flush=True)
assert sys.stdin.readline().strip() == "detach"
by_name.clear()
_C.SharedPool.detach_pool(attachment)
print("detached", flush=True)
assert sys.stdin.readline().strip() == "exit"
"""
        attached_process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker,
                prepared.pool_name,
                prepared.generation,
                prepared.data_pool_name,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "FLEX_DEVICE": "MOCK1p0",
                "FLEX_COMPUTE": "NULL",
                "AIU_WORLD_SIZE": "1",
                "LOCAL_RANK": "0",
            },
        )
        assert attached_process.stdout is not None
        assert attached_process.stderr is not None
        ready = attached_process.stdout.readline().strip()
        assert ready == "attached"

        busy = ctl.destroy(prepared.pool_name, prepared.generation)
        assert busy.get("result") == "POOL_BUSY", busy

        assert attached_process.stdin is not None
        attached_process.stdin.write("detach\n")
        attached_process.stdin.flush()
        assert attached_process.stdout.readline().strip() == "detached"
        assert attached_process.poll() is None

        assert cleanup_pool(prepared.pool_name, prepared.generation, ctl=ctl)["result"] == "OK"
        prepared = None
        attached_process.stdin.write("exit\n")
        attached_process.stdin.flush()
        assert attached_process.wait(timeout=20) == 0, attached_process.stderr.read()
        attached_process = None
    finally:
        if attached_process is not None and attached_process.poll() is None:
            attached_process.terminate()
            attached_process.wait(timeout=10)
        if prepared is not None:
            subprocess.run(
                [
                    str(ctl_binary),
                    "--socket",
                    str(socket),
                    "destroy",
                    "--name",
                    prepared.pool_name,
                    "--generation",
                    prepared.generation,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        process.terminate()
        process.wait(timeout=10)
