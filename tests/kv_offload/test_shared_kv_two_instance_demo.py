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

import json
import os
import shlex
import subprocess
import sys
from argparse import Namespace

import pytest

from scripts import shared_kv_two_instance_demo as demo


@pytest.mark.parametrize(
    ("launcher", "port", "device"),
    (
        ("scripts/start_shared_kv_instance_a.sh", "18100", "0"),
        ("scripts/start_shared_kv_instance_b.sh", "18101", "1"),
    ),
)
def test_launcher_configures_the_one_shared_data_pool(launcher, port, device):
    result = subprocess.run(
        ["bash", launcher],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert f"SPYRE_DEVICES={device}" in result.stdout
    assert f"--port {port}" in result.stdout
    assert "--enforce-eager" not in result.stdout
    assert "--no-enable-prefix-caching" in result.stdout
    command = shlex.split(result.stdout)
    transfer_config = json.loads(command[command.index("--kv-transfer-config") + 1])
    extra_config = transfer_config["kv_connector_extra_config"]
    assert extra_config["pool_name"] == "spyre_manual_4096.data"
    assert "shared_pool_families" not in extra_config


def test_measurement_help_preserves_cleanup_command_line_continuation():
    result = subprocess.run(
        [sys.executable, "scripts/shared_kv_two_instance_demo.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:"
        "/home/yzhu/dt-inductor/sentient/runtime/lib:"
        "/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \\\n"
        "        uv run --no-sync python scripts/cleanup_shared_kv_demo.py" in result.stdout
    )


def test_complete_assembles_stream_and_measures_first_token(monkeypatch):
    class Response:
        def __enter__(self):
            return iter(
                (
                    b'data: {"choices":[{"text":"one","token_ids":[1]}]}\n',
                    b'data: {"choices":[{"text":" two","token_ids":[2]}]}\n',
                    b"data: [DONE]\n",
                )
            )

        def __exit__(self, *args):
            return False

    times = iter((10.0, 10.25, 11.0))
    monkeypatch.setattr(demo.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(demo.time, "monotonic", lambda: next(times))

    completion = demo._complete(
        "stream",
        "http://instance-a",
        "model",
        (10, 11),
        2,
        1,
        print_output=False,
    )

    assert completion.text == "one two"
    assert completion.token_ids == (1, 2)
    assert completion.ttft_seconds == 0.25
    assert completion.wall_seconds == 1.0


def test_wait_for_request_metrics_waits_for_copy_time(monkeypatch):
    incomplete = _empty_metrics()
    incomplete[demo.LOCAL_COMPUTE] = 4.0
    incomplete[demo.STORE_BYTES] = 64.0
    complete = dict(incomplete)
    complete[demo.STORE_TIME] = 0.125
    snapshots = iter((incomplete, complete))
    monkeypatch.setattr(demo, "_metric_snapshot", lambda *args, **kwargs: next(snapshots))
    monkeypatch.setattr(demo.time, "sleep", lambda *args: None)

    result = demo._wait_for_request_metrics(
        "http://instance-a",
        _empty_metrics(),
        request_timeout=1,
        metric_timeout=1,
    )

    assert result[demo.STORE_TIME] == 0.125


def _empty_metrics():
    return {
        demo.LOAD_BYTES: 0.0,
        demo.LOAD_TIME: 0.0,
        demo.STORE_BYTES: 0.0,
        demo.STORE_TIME: 0.0,
        demo.LOCAL_COMPUTE: 0.0,
        demo.EXTERNAL_TRANSFER: 0.0,
    }


def test_run_request_shows_target_and_prompt_before_sending(monkeypatch, capsys):
    shown_before_send = []
    monkeypatch.setattr(demo, "_verify_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo, "_prepare_prompt", lambda *args, **kwargs: (1, 2, 3, 4))
    monkeypatch.setattr(
        demo,
        "_post_json",
        lambda *args, **kwargs: {"prompt": "visible measured prompt"},
    )
    monkeypatch.setattr(demo, "_metric_snapshot", lambda *args, **kwargs: _empty_metrics())

    def complete(*args, **kwargs):
        del args, kwargs
        shown_before_send.append(capsys.readouterr().out)
        return demo.Completion("answer", (9,), 3.25, 11.75)

    monkeypatch.setattr(demo, "_complete", complete)
    deltas = _empty_metrics()
    deltas[demo.LOCAL_COMPUTE] = 4.0
    deltas[demo.STORE_BYTES] = 64.0
    deltas[demo.STORE_TIME] = 0.125
    monkeypatch.setattr(
        demo,
        "_wait_for_request_metrics",
        lambda *args, **kwargs: deltas,
    )

    result = demo.run_request(
        instance_name="A",
        host="10.1.2.3",
        port=18100,
        model="model",
        identifier="visual-demo-v1",
        prompt_tokens=4,
        output_tokens=1,
        request_timeout=1,
        metric_timeout=1,
    )

    assert "Instance: A" in shown_before_send[0]
    assert "Endpoint: http://10.1.2.3:18100" in shown_before_send[0]
    assert "Identifier: visual-demo-v1" in shown_before_send[0]
    assert "Prompt tokens: 4" in shown_before_send[0]
    assert "visible measured prompt" in shown_before_send[0]
    output = capsys.readouterr().out
    assert "=== Result: instance A measured request ===" in output
    assert "Outcome        COLD COMPUTE + STORE" in output
    assert "Prompt source  4 local-compute tokens" in output
    assert "TTFT           3.250 s" in output
    assert "E2E            11.750 s" in output
    assert "KV transfer    STORE 64 bytes in 125.000 ms" in output
    assert result["path"] == "local_compute_store"
    assert result["computed_prompt_tokens"] == 4


def test_run_request_reports_external_reload(monkeypatch, capsys):
    monkeypatch.setattr(demo, "_verify_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo, "_prepare_prompt", lambda *args, **kwargs: (1, 2, 3, 4))
    monkeypatch.setattr(demo, "_post_json", lambda *args, **kwargs: {"prompt": "prompt"})
    monkeypatch.setattr(demo, "_metric_snapshot", lambda *args, **kwargs: _empty_metrics())
    monkeypatch.setattr(
        demo,
        "_complete",
        lambda *args, **kwargs: demo.Completion("answer", (9,), 0.5, 9.0),
    )
    deltas = _empty_metrics()
    deltas[demo.EXTERNAL_TRANSFER] = 4.0
    deltas[demo.LOAD_BYTES] = 64.0
    deltas[demo.LOAD_TIME] = 0.0625
    monkeypatch.setattr(
        demo,
        "_wait_for_request_metrics",
        lambda *args, **kwargs: deltas,
    )

    result = demo.run_request(
        instance_name="B",
        host="10.1.2.4",
        port=18101,
        model="model",
        identifier="visual-demo-v1",
        prompt_tokens=4,
        output_tokens=1,
        request_timeout=1,
        metric_timeout=1,
    )

    output = capsys.readouterr().out
    assert "Outcome        EXTERNAL KV RELOAD" in output
    assert "Prompt source  4 external-transfer tokens" in output
    assert "KV transfer    LOAD 64 bytes in 62.500 ms" in output
    assert result["path"] == "external_kv_reload"
    assert result["loaded_prompt_tokens"] == 4


@pytest.mark.parametrize("show_json", (False, True))
def test_main_shows_machine_readable_result_only_when_requested(monkeypatch, capsys, show_json):
    monkeypatch.setattr(
        demo,
        "_parse_args",
        lambda: Namespace(
            instance="A",
            host="127.0.0.1",
            port=18100,
            model="model",
            identifier="visual-demo-v1",
            prompt_tokens=4,
            output_tokens=1,
            request_timeout=1,
            metric_timeout=1,
            show_json=show_json,
        ),
    )
    monkeypatch.setattr(
        demo,
        "run_request",
        lambda **kwargs: {"instance": kwargs["instance_name"], "path": "local_compute_store"},
    )

    demo.main()

    output = capsys.readouterr().out
    assert ("=== Machine-readable result ===" in output) is show_json


def test_measured_prompt_is_identical_for_a_a_and_b(monkeypatch):
    prepared_texts = []
    monkeypatch.setattr(demo, "_verify_server", lambda *args, **kwargs: None)

    def prepare(*args, **kwargs):
        del args
        prepared_texts.append(kwargs["text"])
        return (1, 2, 3, 4)

    monkeypatch.setattr(demo, "_prepare_prompt", prepare)
    monkeypatch.setattr(demo, "_post_json", lambda *args, **kwargs: {"prompt": "prompt"})
    monkeypatch.setattr(demo, "_metric_snapshot", lambda *args, **kwargs: _empty_metrics())
    monkeypatch.setattr(
        demo,
        "_complete",
        lambda *args, **kwargs: demo.Completion("answer", (9,), 0.5, 9.0),
    )
    deltas = _empty_metrics()
    deltas[demo.EXTERNAL_TRANSFER] = 4.0
    deltas[demo.LOAD_BYTES] = 64.0
    deltas[demo.LOAD_TIME] = 0.0625
    monkeypatch.setattr(
        demo,
        "_wait_for_request_metrics",
        lambda *args, **kwargs: deltas,
    )

    for instance_name, host, port in (
        ("A", "10.1.2.3", 18100),
        ("A", "10.1.2.3", 18100),
        ("B", "10.1.2.4", 18101),
    ):
        demo.run_request(
            instance_name=instance_name,
            host=host,
            port=port,
            model="model",
            identifier="visual-demo-v1",
            prompt_tokens=4,
            output_tokens=1,
            request_timeout=1,
            metric_timeout=1,
        )

    assert prepared_texts[0] == prepared_texts[1] == prepared_texts[2]
