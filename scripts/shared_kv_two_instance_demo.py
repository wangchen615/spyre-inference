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

"""Demonstrate shared KV reloads across two running vLLM instances."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
LOAD_BYTES = "vllm:kv_offload_load_bytes_total"
LOAD_TIME = "vllm:kv_offload_load_time_total"
STORE_BYTES = "vllm:kv_offload_store_bytes_total"
STORE_TIME = "vllm:kv_offload_store_time_total"
PROMPT_SOURCE = "vllm:prompt_tokens_by_source_total"
LOCAL_COMPUTE = "local_compute"
EXTERNAL_TRANSFER = "external_kv_transfer"
RESPONSE_INSTRUCTION = (
    "\n\nResponse: Summarize the recurring symptoms and give three concrete "
    "recovery actions in plain language."
)
SETUP_HELP = """Requirements:
  - Start two already-running vLLM servers on separate Spyre devices.
  - Configure both with SpyreSharedOffloadingSpec, the same shared-pool
    metadata name and pool-family list, and the same PYTHONHASHSEED.
  - Use tensor parallel size 1, the uni executor, disabled prefix caching,
    and a model length of at least prompt tokens plus output tokens.
  - Keep both servers dedicated: send no other requests during this demo.

This script does not start or stop either server.

Example:
  uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py
"""


@dataclass(frozen=True)
class Completion:
    text: str
    token_ids: tuple[int, ...]
    wall_seconds: float


def metric_value(
    metrics_text: str,
    name: str,
    required_labels: dict[str, str] | None = None,
    *,
    required: bool = False,
) -> float:
    total = 0.0
    found = False
    expected_labels = required_labels or {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, value, *_ = line.split()
        sample_name, _, label_text = sample.partition("{")
        if sample_name != name:
            continue
        if any(f'{key}="{label}"' not in label_text for key, label in expected_labels.items()):
            continue
        found = True
        total += float(value)
    if required and not found:
        raise ValueError(f"Prometheus metric {name} is missing")
    return total


def select_prompt_token_ids(
    token_ids: list[int],
    count: int,
    *,
    suffix_token_ids: list[int] | None = None,
) -> tuple[int, ...]:
    suffix = tuple(suffix_token_ids or ())
    prefix_count = count - len(suffix)
    if prefix_count < 0:
        raise ValueError(f"response instruction uses {len(suffix)} tokens; limit is {count}")
    if len(token_ids) < prefix_count:
        required = count if not suffix else prefix_count
        raise ValueError(f"prompt tokenized to {len(token_ids)} tokens; need {required}")
    return tuple(token_ids[:prefix_count]) + suffix


def build_realistic_prompt(run_id: str | None = None) -> str:
    run_identifier = f"Run identifier: {run_id}.\n" if run_id else ""
    introduction = (
        f"{run_identifier}"
        "You are reviewing an accelerator inference incident. Read the request "
        "traces below, identify recurring symptoms, and propose a concise recovery "
        "plan. Preserve important times, component names, and observed behavior.\n\n"
    )
    traces = []
    for index in range(1, 65):
        traces.append(
            f"Request trace {index:03d}: The gateway accepted a long-context "
            f"generation request for tenant-{index % 7}. The scheduler assigned "
            f"batch-{1000 + index} to accelerator-{index % 2} and divided prefill "
            "into bounded chunks. Device memory pressure remained stable while the "
            "worker produced deterministic key and value pages. Completed pages "
            "were copied to the shared host-memory tier and published only after "
            "the transfer completed. A second serving instance queried the same "
            "content hash, resolved the shared slot, reloaded the page, and resumed "
            "generation without recomputing the cached prefix. Operators recorded "
            "request latency, transferred bytes, copy duration, cache source, and "
            "the generated token sequence for comparison. No disk-tier access, "
            "device fallback, stale page, or partial payload was observed.\n"
        )
    return introduction + "\n".join(traces)


def _get_text(url: str, timeout: float) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode()
    except urllib.error.URLError as error:
        raise RuntimeError(f"request failed for {url}: {error}") from error


def _post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"request failed for {url} ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request failed for {url}: {error}") from error


def _metric_snapshot(
    server: str,
    timeout: float,
    *,
    required_names: tuple[str, ...] = (),
) -> dict[str, float]:
    metrics = _get_text(f"{server}/metrics", timeout)
    try:
        return {
            LOAD_BYTES: metric_value(
                metrics,
                LOAD_BYTES,
                required=LOAD_BYTES in required_names,
            ),
            LOAD_TIME: metric_value(
                metrics,
                LOAD_TIME,
                required=LOAD_TIME in required_names,
            ),
            STORE_BYTES: metric_value(
                metrics,
                STORE_BYTES,
                required=STORE_BYTES in required_names,
            ),
            STORE_TIME: metric_value(
                metrics,
                STORE_TIME,
                required=STORE_TIME in required_names,
            ),
            LOCAL_COMPUTE: metric_value(
                metrics,
                PROMPT_SOURCE,
                {"source": LOCAL_COMPUTE},
                required=LOCAL_COMPUTE in required_names,
            ),
            EXTERNAL_TRANSFER: metric_value(
                metrics,
                PROMPT_SOURCE,
                {"source": EXTERNAL_TRANSFER},
                required=EXTERNAL_TRANSFER in required_names,
            ),
        }
    except ValueError as error:
        raise RuntimeError(f"{error} from {server}") from error


def _wait_for_deltas(
    server: str,
    before: dict[str, float],
    names: tuple[str, ...],
    request_timeout: float,
    metric_timeout: float,
) -> dict[str, float]:
    deadline = time.monotonic() + metric_timeout
    last = dict.fromkeys(before, 0.0)
    while time.monotonic() < deadline:
        after = _metric_snapshot(
            server,
            request_timeout,
            required_names=names,
        )
        last = {name: after[name] - before[name] for name in before}
        if all(last[name] > 0 for name in names):
            return last
        time.sleep(0.2)
    expected = {name: last[name] for name in names}
    raise RuntimeError(f"no positive metric delta for {names}: {expected}")


def _require_prompt_source(
    label: str,
    deltas: dict[str, float],
    prompt_tokens: int,
    expected_source: str,
) -> None:
    other_source = EXTERNAL_TRANSFER if expected_source == LOCAL_COMPUTE else LOCAL_COMPUTE
    if deltas[expected_source] != prompt_tokens or deltas[other_source] != 0:
        raise AssertionError(
            f"{label} expected {prompt_tokens} {expected_source} tokens and 0 "
            f"{other_source} tokens; observed {deltas[expected_source]} and "
            f"{deltas[other_source]}"
        )


def _complete(
    label: str,
    server: str,
    model: str,
    prompt: tuple[int, ...],
    output_tokens: int,
    request_timeout: float,
    *,
    print_output: bool = True,
) -> Completion:
    started = time.monotonic()
    body = _post_json(
        f"{server}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "add_special_tokens": False,
            "temperature": 0.0,
            "seed": 17,
            "ignore_eos": True,
            "max_tokens": output_tokens,
            "return_token_ids": True,
        },
        request_timeout,
    )
    choice = body["choices"][0]
    completion = Completion(
        text=choice["text"],
        token_ids=tuple(choice["token_ids"]),
        wall_seconds=time.monotonic() - started,
    )
    if len(completion.token_ids) != output_tokens:
        raise RuntimeError(
            f"{label} returned {len(completion.token_ids)} tokens; expected {output_tokens}"
        )
    if print_output:
        print(f"\n=== {label} output ===")
        print(f"wall time: {completion.wall_seconds:.3f} seconds")
        print(f"token IDs: {list(completion.token_ids)}")
        print(f"text: {json.dumps(completion.text, ensure_ascii=False)}")
    return completion


def _prepare_prompt(
    server: str,
    model: str,
    run_id: str,
    prompt_tokens: int,
    output_tokens: int,
    request_timeout: float,
) -> tuple[int, ...]:
    tokenized = _post_json(
        f"{server}/tokenize",
        {
            "model": model,
            "prompt": build_realistic_prompt(run_id=run_id),
            "add_special_tokens": False,
        },
        request_timeout,
    )
    response_instruction = _post_json(
        f"{server}/tokenize",
        {
            "model": model,
            "prompt": RESPONSE_INSTRUCTION,
            "add_special_tokens": False,
        },
        request_timeout,
    )
    prompt = select_prompt_token_ids(
        tokenized["tokens"],
        prompt_tokens,
        suffix_token_ids=response_instruction["tokens"],
    )
    max_model_len = int(tokenized["max_model_len"])
    if prompt_tokens + output_tokens > max_model_len:
        raise RuntimeError(
            f"{prompt_tokens} prompt + {output_tokens} output tokens exceed "
            f"the server maximum of {max_model_len}"
        )
    return prompt


def run_demo(
    *,
    server_a: str,
    server_b: str,
    model: str,
    prompt_tokens: int,
    output_tokens: int,
    request_timeout: float,
    metric_timeout: float,
) -> dict[str, Any]:
    server_a = server_a.rstrip("/")
    server_b = server_b.rstrip("/")
    if server_a == server_b:
        raise ValueError("instances A and B must use different server URLs")
    for name, server in (("A", server_a), ("B", server_b)):
        _get_text(f"{server}/health", request_timeout)
        metrics = _get_text(f"{server}/metrics", request_timeout)
        try:
            prefix_caching_disabled = metric_value(
                metrics,
                "vllm:cache_config_info",
                {"enable_prefix_caching": "False"},
                required=True,
            )
        except ValueError as error:
            raise RuntimeError(f"{error} from instance {name} at {server}") from error
        if prefix_caching_disabled <= 0:
            raise RuntimeError(f"instance {name} does not report prefix caching disabled")

    warmup_id = uuid.uuid4().hex
    warmup_a_prompt = _prepare_prompt(
        server_a,
        model,
        f"warmup-a-{warmup_id}",
        prompt_tokens,
        output_tokens,
        request_timeout,
    )
    warmup_b_prompt = _prepare_prompt(
        server_b,
        model,
        f"warmup-b-{warmup_id}",
        prompt_tokens,
        output_tokens,
        request_timeout,
    )

    print(
        f"Warming A and B with distinct {prompt_tokens}-token prompts "
        f"and {output_tokens} output tokens each"
    )
    before = _metric_snapshot(server_a, request_timeout)
    warmup_a = _complete(
        "A warmup",
        server_a,
        model,
        warmup_a_prompt,
        output_tokens,
        request_timeout,
        print_output=False,
    )
    warmup_a_metrics = _wait_for_deltas(
        server_a,
        before,
        (STORE_BYTES, STORE_TIME, LOCAL_COMPUTE),
        request_timeout,
        metric_timeout,
    )
    _require_prompt_source(
        "A warmup",
        warmup_a_metrics,
        len(warmup_a_prompt),
        LOCAL_COMPUTE,
    )

    before = _metric_snapshot(server_b, request_timeout)
    warmup_b = _complete(
        "B warmup",
        server_b,
        model,
        warmup_b_prompt,
        output_tokens,
        request_timeout,
        print_output=False,
    )
    warmup_b_metrics = _wait_for_deltas(
        server_b,
        before,
        (STORE_BYTES, STORE_TIME, LOCAL_COMPUTE),
        request_timeout,
        metric_timeout,
    )
    _require_prompt_source(
        "B warmup",
        warmup_b_metrics,
        len(warmup_b_prompt),
        LOCAL_COMPUTE,
    )
    print(f"A warmup wall time: {warmup_a.wall_seconds:.3f} seconds")
    print(f"B warmup wall time: {warmup_b.wall_seconds:.3f} seconds")
    print("Warmup complete; starting measured requests")

    run_id = uuid.uuid4().hex
    prompt = _prepare_prompt(
        server_a,
        model,
        run_id,
        prompt_tokens,
        output_tokens,
        request_timeout,
    )
    prompt_text = _post_json(
        f"{server_a}/detokenize",
        {"model": model, "tokens": prompt},
        request_timeout,
    )["prompt"]
    print(f"Run identifier: {run_id}")
    print(f"Prompt: exactly {len(prompt)} token IDs")
    print(f"Prompt preview: {prompt_text[:500]!r}")

    before = _metric_snapshot(server_a, request_timeout)
    baseline = _complete(
        "A data-cold compute/store",
        server_a,
        model,
        prompt,
        output_tokens,
        request_timeout,
    )
    cold = _wait_for_deltas(
        server_a,
        before,
        (STORE_BYTES, STORE_TIME, LOCAL_COMPUTE),
        request_timeout,
        metric_timeout,
    )
    _require_prompt_source("A data-cold", cold, len(prompt), LOCAL_COMPUTE)

    before = _metric_snapshot(server_a, request_timeout)
    a_reload = _complete(
        "A self shared-pool reload",
        server_a,
        model,
        prompt,
        output_tokens,
        request_timeout,
    )
    a_load = _wait_for_deltas(
        server_a,
        before,
        (LOAD_BYTES, LOAD_TIME, EXTERNAL_TRANSFER),
        request_timeout,
        metric_timeout,
    )
    _require_prompt_source("A self reload", a_load, len(prompt), EXTERNAL_TRANSFER)

    before = _metric_snapshot(server_b, request_timeout)
    b_reload = _complete(
        "B peer shared-pool reload",
        server_b,
        model,
        prompt,
        output_tokens,
        request_timeout,
    )
    b_load = _wait_for_deltas(
        server_b,
        before,
        (LOAD_BYTES, LOAD_TIME, EXTERNAL_TRANSFER),
        request_timeout,
        metric_timeout,
    )
    _require_prompt_source("B peer reload", b_load, len(prompt), EXTERNAL_TRANSFER)
    if not (cold[STORE_BYTES] == a_load[LOAD_BYTES] == b_load[LOAD_BYTES]):
        raise AssertionError(
            "store and reload transfer byte counts differ: "
            f"A store={cold[STORE_BYTES]}, A load={a_load[LOAD_BYTES]}, "
            f"B load={b_load[LOAD_BYTES]}"
        )

    outputs_identical = (
        a_reload.token_ids == baseline.token_ids
        and a_reload.text.encode() == baseline.text.encode()
        and b_reload.token_ids == baseline.token_ids
        and b_reload.text.encode() == baseline.text.encode()
    )
    if not outputs_identical:
        raise AssertionError("cold, self-reload, and peer-reload outputs differ")

    result = {
        "run_id": run_id,
        "prompt_tokens": len(prompt),
        "output_tokens": len(baseline.token_ids),
        "warmup": {
            "a_wall_seconds": warmup_a.wall_seconds,
            "b_wall_seconds": warmup_b.wall_seconds,
            "prompt_tokens_each": prompt_tokens,
            "output_tokens_each": output_tokens,
        },
        "a_cold": {
            "wall_seconds": baseline.wall_seconds,
            "computed_prompt_tokens": int(cold[LOCAL_COMPUTE]),
            "store_bytes": int(cold[STORE_BYTES]),
            "store_copy_seconds": cold[STORE_TIME],
        },
        "a_self_reload": {
            "wall_seconds": a_reload.wall_seconds,
            "loaded_prompt_tokens": int(a_load[EXTERNAL_TRANSFER]),
            "load_bytes": int(a_load[LOAD_BYTES]),
            "load_copy_seconds": a_load[LOAD_TIME],
        },
        "b_peer_reload": {
            "wall_seconds": b_reload.wall_seconds,
            "loaded_prompt_tokens": int(b_load[EXTERNAL_TRANSFER]),
            "load_bytes": int(b_load[LOAD_BYTES]),
            "load_copy_seconds": b_load[LOAD_TIME],
        },
        "outputs_identical": True,
    }
    print("\n=== Verification summary ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=SETUP_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server-a",
        default="http://127.0.0.1:18100",
        help="instance A base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--server-b",
        default="http://127.0.0.1:18101",
        help="instance B base URL (default: %(default)s)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--metric-timeout", type=float, default=120)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_demo(
        server_a=args.server_a,
        server_b=args.server_b,
        model=args.model,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        request_timeout=args.request_timeout,
        metric_timeout=args.metric_timeout,
    )


if __name__ == "__main__":
    main()
