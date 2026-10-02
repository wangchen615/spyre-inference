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

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.cleanup_shared_kv_demo import cleanup_shared_kv_demo

CURRENT_NAMES = ("demo.data",)
LEGACY_NAMES = tuple(
    f"demo.{family}.c{cache_index}.{role}"
    for family in ("a", "b")
    for cache_index in range(4)
    for role in ("k", "v")
)
CURRENT_CONFIG = (1, [], (1, 2048, 1))
LEGACY_CONFIG = (8, [], (16, 128, 1))


class _FakeMetadata:
    def __init__(self, pools):
        self._pools = pools

    def find_pool(self, name):
        return self._pools.get(name)

    def pool_count(self):
        return len(self._pools)


class _FakeRuntime:
    def __init__(
        self,
        metadata,
        *,
        accepted_config=None,
        attach_error: RuntimeError | None = None,
        shm_root: Path | None = None,
        remove_pools: bool = True,
        remove_metadata: bool = True,
    ):
        self.metadata = metadata
        self.accepted_config = accepted_config
        self.attach_error = attach_error
        self.unlinked_metadata = []
        self.unlinked_pools = []
        self.configs = []

        runtime = self

        class SharedMetadata:
            @staticmethod
            def create_or_attach(name, config):
                runtime.configs.append(config)
                if runtime.attach_error is not None:
                    raise runtime.attach_error
                if runtime.accepted_config is not None and config != runtime.accepted_config:
                    raise RuntimeError("SharedMetadata: fixed capacity configuration mismatch")
                return runtime.metadata

            @staticmethod
            def unlink_by_name(name):
                runtime.unlinked_metadata.append(name)
                if shm_root is not None and remove_metadata:
                    (shm_root / name.lstrip("/")).unlink(missing_ok=True)

        class SharedHostPool:
            @staticmethod
            def unlink_by_name(name):
                runtime.unlinked_pools.append(name)
                if shm_root is not None and remove_pools:
                    path = shm_root / name.lstrip("/")
                    path.unlink(missing_ok=True)
                    path.with_name(f"{path.name}.ctl").unlink(missing_ok=True)

        self.SharedMetadata = SharedMetadata
        self.SharedHostPool = SharedHostPool
        self.SharedMetadataCapacity = lambda max_pools, max_slots, max_compat: (
            max_pools,
            max_slots,
            max_compat,
        )
        self.SharedMetadataConfig = lambda max_chunks, pools, capacity: (
            max_chunks,
            pools,
            capacity,
        )


def _registered_pool(metadata_version, pool_id, pool_version=1):
    return SimpleNamespace(
        pool_ref=SimpleNamespace(
            metadata_version=metadata_version,
            pool_id=pool_id,
            pool_version=pool_version,
        )
    )


def _backing_filename(pool):
    pool_ref = pool.pool_ref
    return (
        f"flex_kv_{pool_ref.metadata_version:016x}_{pool_ref.pool_id}_{pool_ref.pool_version:016x}"
    )


def _create_shared_objects(tmp_path: Path, metadata_name: str, pools) -> None:
    (tmp_path / metadata_name).touch()
    for pool in pools.values():
        path = tmp_path / _backing_filename(pool)
        path.touch()
        path.with_name(f"{path.name}.ctl").touch()


def test_cleanup_removes_current_one_pool_layout(tmp_path: Path):
    pools = {CURRENT_NAMES[0]: _registered_pool(0x1234, 7)}
    runtime = _FakeRuntime(_FakeMetadata(pools), shm_root=tmp_path)
    _create_shared_objects(tmp_path, "demo", pools)

    removed = cleanup_shared_kv_demo(
        metadata_name="demo",
        proc_root=tmp_path / "proc",
        shm_root=tmp_path,
        runtime=runtime,
    )

    assert removed == [f"/flex_kv_{0x1234:016x}_7_{1:016x}"]
    assert runtime.unlinked_pools == removed
    assert runtime.unlinked_metadata == ["demo"]
    assert runtime.configs == [CURRENT_CONFIG]
    assert not (tmp_path / "demo").exists()
    assert not (tmp_path / _backing_filename(pools[CURRENT_NAMES[0]])).exists()


def test_cleanup_falls_back_to_and_removes_all_legacy_pools(tmp_path: Path):
    pools = {name: _registered_pool(0x5678, pool_id) for pool_id, name in enumerate(LEGACY_NAMES)}
    runtime = _FakeRuntime(
        _FakeMetadata(pools),
        accepted_config=LEGACY_CONFIG,
        shm_root=tmp_path,
    )
    _create_shared_objects(tmp_path, "demo", pools)

    removed = cleanup_shared_kv_demo(
        metadata_name="demo",
        proc_root=tmp_path / "proc",
        shm_root=tmp_path,
        runtime=runtime,
    )

    assert removed == [
        f"/flex_kv_{0x5678:016x}_{pool_id}_{1:016x}" for pool_id in range(len(LEGACY_NAMES))
    ]
    assert runtime.configs == [CURRENT_CONFIG, LEGACY_CONFIG]
    assert runtime.unlinked_pools == removed
    assert runtime.unlinked_metadata == ["demo"]


def test_cleanup_removes_explicit_additional_demo_pool(tmp_path: Path):
    names = ("demo.archive",)
    pools = {name: _registered_pool(0x1234, pool_id) for pool_id, name in enumerate(names)}
    runtime = _FakeRuntime(_FakeMetadata(pools), shm_root=tmp_path)
    _create_shared_objects(tmp_path, "demo", pools)

    removed = cleanup_shared_kv_demo(
        metadata_name="demo",
        pool_names=("demo.archive",),
        proc_root=tmp_path / "proc",
        shm_root=tmp_path,
        runtime=runtime,
    )

    assert len(removed) == 1
    assert runtime.unlinked_pools == removed
    assert runtime.unlinked_metadata == ["demo"]


def test_cleanup_refuses_while_vllm_uses_namespace(tmp_path: Path):
    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(
        b"vllm\0serve\0--kv-transfer-config\0shared_metadata_name=demo\0"
    )
    runtime = _FakeRuntime(_FakeMetadata({}))
    (tmp_path / "demo").touch()

    with pytest.raises(RuntimeError, match="PID 123"):
        cleanup_shared_kv_demo(
            metadata_name="demo",
            proc_root=proc_root,
            shm_root=tmp_path,
            runtime=runtime,
        )

    assert runtime.unlinked_pools == []
    assert runtime.unlinked_metadata == []


def test_cleanup_refuses_module_form_vllm_server(tmp_path: Path):
    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(
        b"python\0-m\0vllm.entrypoints.cli.main\0serve\0model\0"
        b"--kv-transfer-config\0shared_metadata_name=demo\0"
    )
    runtime = _FakeRuntime(_FakeMetadata({}), shm_root=tmp_path)
    (tmp_path / "demo").touch()

    with pytest.raises(RuntimeError, match="PID 123"):
        cleanup_shared_kv_demo(
            metadata_name="demo",
            proc_root=proc_root,
            shm_root=tmp_path,
            runtime=runtime,
        )

    assert runtime.unlinked_pools == []
    assert runtime.unlinked_metadata == []


def test_cleanup_refuses_process_with_metadata_mapping(tmp_path: Path):
    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python\0worker.py\0")
    (process / "maps").write_text(f"7f000-7f100 rw-s 00000000 00:00 0 {tmp_path / 'demo'}\n")
    runtime = _FakeRuntime(_FakeMetadata({}), shm_root=tmp_path)
    (tmp_path / "demo").touch()

    with pytest.raises(RuntimeError, match="PID 123"):
        cleanup_shared_kv_demo(
            metadata_name="demo",
            proc_root=proc_root,
            shm_root=tmp_path,
            runtime=runtime,
        )


def test_cleanup_ignores_shell_command_that_only_quotes_vllm_command(tmp_path: Path):
    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"/bin/bash\0-lc\0echo 'vllm serve demo'\0")
    runtime = _FakeRuntime(_FakeMetadata({}), shm_root=tmp_path)
    (tmp_path / "demo").touch()

    removed = cleanup_shared_kv_demo(
        metadata_name="demo",
        proc_root=proc_root,
        shm_root=tmp_path,
        runtime=runtime,
    )

    assert removed == []
    assert runtime.unlinked_metadata == ["demo"]


def test_cleanup_is_idempotent_when_metadata_is_absent(tmp_path: Path):
    runtime = _FakeRuntime(_FakeMetadata({}))

    removed = cleanup_shared_kv_demo(
        metadata_name="demo",
        proc_root=tmp_path / "proc",
        shm_root=tmp_path,
        runtime=runtime,
    )

    assert removed == []
    assert runtime.configs == []
    assert runtime.unlinked_metadata == []


def test_cleanup_loads_runtime_in_mock_mode_and_restores_environment(tmp_path: Path, monkeypatch):
    runtime = _FakeRuntime(_FakeMetadata({}), shm_root=tmp_path)
    (tmp_path / "demo").touch()
    observed = {}
    shared_runtime = ModuleType("spyre_inference.v1.kv_offload.shared_runtime")

    def load_shared_runtime():
        for name in (
            "VLLM_PLUGINS",
            "TORCH_DEVICE_BACKEND_AUTOLOAD",
            "FLEX_DEVICE",
            "FLEX_COMPUTE",
            "AIU_WORLD_SIZE",
            "LOCAL_RANK",
        ):
            observed[name] = os.environ.get(name)
        return runtime

    shared_runtime.load_shared_runtime = load_shared_runtime
    monkeypatch.setitem(sys.modules, shared_runtime.__name__, shared_runtime)
    original = {
        "VLLM_PLUGINS": "original-plugin",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "FLEX_DEVICE": "PF",
        "FLEX_COMPUTE": "SENTIENT",
        "AIU_WORLD_SIZE": "4",
        "LOCAL_RANK": "2",
    }
    for name, value in original.items():
        monkeypatch.setenv(name, value)

    cleanup_shared_kv_demo(
        metadata_name="demo",
        proc_root=tmp_path / "proc",
        shm_root=tmp_path,
    )

    assert observed == {
        "VLLM_PLUGINS": "",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "FLEX_DEVICE": "MOCK1p0",
        "FLEX_COMPUTE": "NULL",
        "AIU_WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
    }
    assert {name: os.environ.get(name) for name in original} == original


def test_cleanup_refuses_metadata_with_unknown_pool(tmp_path: Path):
    runtime = _FakeRuntime(
        _FakeMetadata({"foreign": _registered_pool(0x1234, 99)}),
        shm_root=tmp_path,
    )
    (tmp_path / "demo").touch()

    with pytest.raises(RuntimeError, match="outside the known demo layout"):
        cleanup_shared_kv_demo(
            metadata_name="demo",
            proc_root=tmp_path / "proc",
            shm_root=tmp_path,
            runtime=runtime,
        )

    assert runtime.unlinked_pools == []
    assert runtime.unlinked_metadata == []


def test_cleanup_does_not_hide_non_configuration_attach_failure(tmp_path: Path):
    runtime = _FakeRuntime(
        _FakeMetadata({}),
        attach_error=RuntimeError("permission denied"),
        shm_root=tmp_path,
    )
    (tmp_path / "demo").touch()

    with pytest.raises(RuntimeError, match="permission denied"):
        cleanup_shared_kv_demo(
            metadata_name="demo",
            proc_root=tmp_path / "proc",
            shm_root=tmp_path,
            runtime=runtime,
        )

    assert runtime.configs == [CURRENT_CONFIG]


def test_cleanup_preserves_metadata_when_backing_unlink_fails(tmp_path: Path):
    pools = {CURRENT_NAMES[0]: _registered_pool(0x1234, 0)}
    runtime = _FakeRuntime(
        _FakeMetadata(pools),
        shm_root=tmp_path,
        remove_pools=False,
    )
    _create_shared_objects(tmp_path, "demo", pools)

    with pytest.raises(RuntimeError, match="failed to unlink shared data pools"):
        cleanup_shared_kv_demo(
            metadata_name="demo",
            proc_root=tmp_path / "proc",
            shm_root=tmp_path,
            runtime=runtime,
        )

    assert runtime.unlinked_metadata == []
    assert (tmp_path / "demo").exists()


def test_cleanup_reports_metadata_unlink_failure(tmp_path: Path):
    pools = {CURRENT_NAMES[0]: _registered_pool(0x1234, 0)}
    runtime = _FakeRuntime(
        _FakeMetadata(pools),
        shm_root=tmp_path,
        remove_metadata=False,
    )
    _create_shared_objects(tmp_path, "demo", pools)

    with pytest.raises(RuntimeError, match="failed to unlink shared metadata"):
        cleanup_shared_kv_demo(
            metadata_name="demo",
            proc_root=tmp_path / "proc",
            shm_root=tmp_path,
            runtime=runtime,
        )

    assert runtime.unlinked_metadata == ["demo"]
    assert (tmp_path / "demo").exists()


def test_cleanup_disables_plugin_autoload_before_importing_runtime(tmp_path: Path, monkeypatch):
    runtime = _FakeRuntime(_FakeMetadata({}), shm_root=tmp_path)
    observed = {}
    shared_runtime = ModuleType("spyre_inference.v1.kv_offload.shared_runtime")

    def load_shared_runtime():
        observed["VLLM_PLUGINS"] = os.environ.get("VLLM_PLUGINS")
        observed["TORCH_DEVICE_BACKEND_AUTOLOAD"] = os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")
        return runtime

    shared_runtime.load_shared_runtime = load_shared_runtime
    monkeypatch.setitem(sys.modules, shared_runtime.__name__, shared_runtime)
    monkeypatch.setenv("VLLM_PLUGINS", "spyre_inference")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    (tmp_path / "demo").touch()

    cleanup_shared_kv_demo(
        metadata_name="demo",
        proc_root=tmp_path / "proc",
        shm_root=tmp_path,
    )

    assert observed == {
        "VLLM_PLUGINS": "",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
