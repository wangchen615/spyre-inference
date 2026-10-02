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

import subprocess
import sys

import pytest

from scripts import shared_kv_two_instance_demo as demo
from scripts import warmup_shared_kv_demo as warmup


def test_warmup_uses_unrelated_prompts_and_exercises_both_reload_paths(monkeypatch, capsys):
    calls = []

    def execute_request(**kwargs):
        calls.append(kwargs)
        return {
            "instance": kwargs["instance_name"],
            "identifier": kwargs["identifier"],
            "path": kwargs["expected_source"],
        }

    monkeypatch.setattr(demo, "execute_request", execute_request)

    result = warmup.run_warmup(
        instance_a_host="10.1.2.3",
        instance_a_port=18100,
        instance_b_host="10.1.2.4",
        instance_b_port=18101,
        model="model",
        warmup_id="warmup-123",
        prompt_tokens=4096,
        output_tokens=16,
        request_timeout=1,
        metric_timeout=1,
    )

    assert [call["label"] for call in calls] == [
        "A warmup compute/store",
        "B warmup compute/store",
        "A warmup self reload",
        "B warmup peer reload",
    ]
    assert [call["expected_source"] for call in calls] == [
        demo.LOCAL_COMPUTE,
        demo.LOCAL_COMPUTE,
        demo.EXTERNAL_TRANSFER,
        demo.EXTERNAL_TRANSFER,
    ]
    assert calls[0]["prompt_source"] != calls[1]["prompt_source"]
    assert calls[0]["prompt_source"] == calls[2]["prompt_source"]
    assert calls[0]["prompt_source"] == calls[3]["prompt_source"]
    assert [(call["host"], call["port"]) for call in calls] == [
        ("10.1.2.3", 18100),
        ("10.1.2.4", 18101),
        ("10.1.2.3", 18100),
        ("10.1.2.4", 18101),
    ]
    output = capsys.readouterr().out
    assert "Warmup identifier: warmup-123" in output
    assert "Instance A: http://10.1.2.3:18100" in output
    assert "Instance B: http://10.1.2.4:18101" in output
    assert len(result["requests"]) == 4


def test_warmup_script_can_run_directly_from_the_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/warmup_shared_kv_demo.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--instance-a-host" in result.stdout


def test_warmup_rejects_identical_instance_endpoints(monkeypatch):
    monkeypatch.setattr(
        demo,
        "execute_request",
        lambda **kwargs: pytest.fail(f"sent request unexpectedly: {kwargs}"),
    )

    with pytest.raises(ValueError, match="different endpoints"):
        warmup.run_warmup(
            instance_a_host="127.0.0.1",
            instance_a_port=18100,
            instance_b_host="127.0.0.1",
            instance_b_port=18100,
            model="model",
            warmup_id="warmup-123",
            prompt_tokens=4096,
            output_tokens=16,
            request_timeout=1,
            metric_timeout=1,
        )
