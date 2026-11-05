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


def test_warmup_cold_stores_then_cross_reloads_distinct_junk_prompts(monkeypatch, capsys):
    calls = []

    def execute_request(**kwargs):
        calls.append(kwargs)
        loaded = kwargs["expected_source"] == demo.EXTERNAL_TRANSFER
        return {
            "label": kwargs["label"],
            "instance": kwargs["instance_name"],
            "endpoint": f"http://{kwargs['host']}:{kwargs['port']}",
            "identifier": kwargs["identifier"],
            "path": "external_kv_reload" if loaded else "local_compute_store",
            "prompt_tokens": 4096,
            "output_tokens": 16,
            "ttft_seconds": 82.5,
            "wall_seconds": 132.25,
            "computed_prompt_tokens": 0 if loaded else 4096,
            "loaded_prompt_tokens": 4096 if loaded else 0,
            "store_bytes": 0 if loaded else 67_108_864,
            "store_copy_seconds": 0.0 if loaded else 0.0125,
            "load_bytes": 67_108_864 if loaded else 0,
            "load_copy_seconds": 0.0125 if loaded else 0.0,
            "token_ids": [1] * 16,
            "text": "junk output",
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
        "A junk cold compute/store",
        "B junk cold compute/store",
        "B reloads A junk prompt",
        "A reloads B junk prompt",
    ]
    assert [call["expected_source"] for call in calls] == [
        demo.LOCAL_COMPUTE,
        demo.LOCAL_COMPUTE,
        demo.EXTERNAL_TRANSFER,
        demo.EXTERNAL_TRANSFER,
    ]
    assert calls[0]["prompt_source"] != calls[1]["prompt_source"]
    assert "amber observatory" in calls[0]["prompt_source"]
    assert "cobalt reef" in calls[1]["prompt_source"]
    assert calls[0]["response_instruction"] != calls[1]["response_instruction"]
    assert "amber observatory" in calls[0]["response_instruction"]
    assert "cobalt reef" in calls[1]["response_instruction"]
    assert "host-to-device" not in calls[0]["prompt_source"]
    assert [(call["host"], call["port"]) for call in calls] == [
        ("10.1.2.3", 18100),
        ("10.1.2.4", 18101),
        ("10.1.2.4", 18101),
        ("10.1.2.3", 18100),
    ]
    assert calls[2]["identifier"] == calls[0]["identifier"]
    assert calls[2]["prompt_source"] == calls[0]["prompt_source"]
    assert calls[2]["response_instruction"] == calls[0]["response_instruction"]
    assert calls[3]["identifier"] == calls[1]["identifier"]
    assert calls[3]["prompt_source"] == calls[1]["prompt_source"]
    assert calls[3]["response_instruction"] == calls[1]["response_instruction"]
    output = capsys.readouterr().out
    assert "Warmup identifier: warmup-123" in output
    assert "Instance A: http://10.1.2.3:18100" in output
    assert "Instance B: http://10.1.2.4:18101" in output
    assert "=== Warmup summary ===" in output
    assert "A junk cold compute/store" in output
    assert "B junk cold compute/store" in output
    assert "B reloads A junk prompt" in output
    assert "A reloads B junk prompt" in output
    assert "82.500 s" in output
    assert "KV transfer" not in output
    assert "=== Machine-readable warmup result ===" not in output
    assert "Warmup complete: 4/4 requests completed" in output
    assert len(result["requests"]) == 4
    assert result["output_comparisons"] == [
        {
            "cold_label": "A junk cold compute/store",
            "reload_label": "B reloads A junk prompt",
            "token_ids_match": True,
            "text_matches": True,
        },
        {
            "cold_label": "B junk cold compute/store",
            "reload_label": "A reloads B junk prompt",
            "token_ids_match": True,
            "text_matches": True,
        },
    ]


@pytest.mark.parametrize("mismatch_call", [2, 3])
@pytest.mark.parametrize("mismatch_field", ["token_ids", "text"])
def test_warmup_reports_crossed_reload_output_mismatch(
    monkeypatch, capsys, mismatch_call, mismatch_field
):
    call_index = 0

    def execute_request(**kwargs):
        nonlocal call_index
        current_call = call_index
        call_index += 1
        loaded = kwargs["expected_source"] == demo.EXTERNAL_TRANSFER
        result = {
            "label": kwargs["label"],
            "instance": kwargs["instance_name"],
            "endpoint": f"http://{kwargs['host']}:{kwargs['port']}",
            "identifier": kwargs["identifier"],
            "path": "external_kv_reload" if loaded else "local_compute_store",
            "prompt_tokens": 4096,
            "output_tokens": 2,
            "ttft_seconds": 1.0,
            "wall_seconds": 2.0,
            "computed_prompt_tokens": 0 if loaded else 4096,
            "loaded_prompt_tokens": 4096 if loaded else 0,
            "store_bytes": 0 if loaded else 64,
            "store_copy_seconds": 0.0 if loaded else 0.1,
            "load_bytes": 64 if loaded else 0,
            "load_copy_seconds": 0.1 if loaded else 0.0,
            "token_ids": [1, 2],
            "text": "matching output",
        }
        if current_call == mismatch_call:
            result[mismatch_field] = [1, 3] if mismatch_field == "token_ids" else "different output"
        return result

    monkeypatch.setattr(demo, "execute_request", execute_request)

    result = warmup.run_warmup(
        instance_a_host="10.1.2.3",
        instance_a_port=18100,
        instance_b_host="10.1.2.4",
        instance_b_port=18101,
        model="model",
        warmup_id="warmup-123",
        prompt_tokens=4096,
        output_tokens=2,
        request_timeout=1,
        metric_timeout=1,
    )

    comparison = result["output_comparisons"][mismatch_call - 2]
    assert comparison["token_ids_match"] is (mismatch_field != "token_ids")
    assert comparison["text_matches"] is (mismatch_field != "text")
    assert "WARNING: reload output differs from cold output" in capsys.readouterr().out


def test_warmup_shows_json_only_when_requested(monkeypatch, capsys):
    monkeypatch.setattr(
        demo,
        "execute_request",
        lambda **kwargs: {
            "label": kwargs["label"],
            "instance": kwargs["instance_name"],
            "endpoint": f"http://{kwargs['host']}:{kwargs['port']}",
            "identifier": kwargs["identifier"],
            "path": "local_compute_store",
            "prompt_tokens": 4,
            "output_tokens": 1,
            "ttft_seconds": 1.0,
            "wall_seconds": 2.0,
            "computed_prompt_tokens": 4,
            "loaded_prompt_tokens": 0,
            "store_bytes": 64,
            "store_copy_seconds": 0.125,
            "load_bytes": 0,
            "load_copy_seconds": 0.0,
            "token_ids": [1],
            "text": "junk output",
        },
    )

    warmup.run_warmup(
        instance_a_host="10.1.2.3",
        instance_a_port=18100,
        instance_b_host="10.1.2.4",
        instance_b_port=18101,
        model="model",
        warmup_id="warmup-123",
        prompt_tokens=4,
        output_tokens=1,
        request_timeout=1,
        metric_timeout=1,
        show_json=True,
    )

    output = capsys.readouterr().out
    assert "=== Machine-readable warmup result ===" in output
    assert '"store_bytes": 64' in output


def test_warmup_script_can_run_directly_from_the_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/warmup_shared_kv_demo.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--instance-a-host" in result.stdout
    assert "--show-json" in result.stdout


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
