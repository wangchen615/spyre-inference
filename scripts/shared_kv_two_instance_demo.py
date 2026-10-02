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
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
DEFAULT_IDENTIFIER = "spyre-shared-kv-visual-demo-v1"
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
WARMUP_RESPONSE_INSTRUCTION = "\n\nWarmup response: emit sixteen deterministic tokens."
MEASURED_PROMPT = """
You are reviewing an accelerator inference incident for a live demonstration.
The gateway accepted a long-context generation request and divided its prefill
into bounded chunks. The Spyre worker computed deterministic key and value
pages, copied completed pages into shared host memory, and published them only
after each transfer finished. A serving instance can later find the same
content, reload those pages, and continue generation without recomputing the
cached prefix. Review the incident, preserve the important component names,
and explain the expected recovery behavior in plain language.
""".strip()
CLEANUP_LIBRARY_PATH = (
    "/opt/ibm/spyre/spyre-comms/lib:"
    "/home/yzhu/dt-inductor/sentient/runtime/lib:"
    "/opt/ibm/spyre/runtime/lib"
)
SETUP_HELP = f"""Requirements:
  - Start two already-running vLLM servers on separate Spyre devices.
  - Configure both with SpyreSharedOffloadingSpec, the same shared-pool
    metadata name and one data-pool name, and the same PYTHONHASHSEED.
  - Use tensor parallel size 1, the uni executor, disabled prefix caching,
    and a model length of at least prompt tokens plus output tokens.
  - Keep both servers dedicated: send no other requests during this demo.
  - Before a new A -> A -> B sequence, stop both servers and clean the pools:
      LD_LIBRARY_PATH={CLEANUP_LIBRARY_PATH}:$LD_LIBRARY_PATH \\
        uv run --no-sync python scripts/cleanup_shared_kv_demo.py

Run this command once against A, again against A, and then against B. Keep the
identifier unchanged so all three invocations create exactly the same prompt.

Example:
  uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py \\
    --instance A --host 127.0.0.1 --port 18100
"""


@dataclass(frozen=True)
class Completion:
    text: str
    token_ids: tuple[int, ...]
    ttft_seconds: float
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


def build_measured_prompt(identifier: str) -> str:
    header = f"Shared KV demo identifier: {identifier}\n\n"
    return header + (MEASURED_PROMPT + "\n\n") * 128


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


def _wait_for_request_metrics(
    server: str,
    before: dict[str, float],
    request_timeout: float,
    metric_timeout: float,
) -> dict[str, float]:
    deadline = time.monotonic() + metric_timeout
    last = dict.fromkeys(before, 0.0)
    while time.monotonic() < deadline:
        after = _metric_snapshot(server, request_timeout)
        last = {name: after[name] - before[name] for name in before}
        computed = last[LOCAL_COMPUTE] > 0 and last[STORE_BYTES] > 0 and last[STORE_TIME] > 0
        loaded = last[EXTERNAL_TRANSFER] > 0 and last[LOAD_BYTES] > 0 and last[LOAD_TIME] > 0
        if computed or loaded:
            return last
        time.sleep(0.2)
    raise RuntimeError(
        f"request produced neither a compute/store nor an external-reload metric delta: {last}"
    )


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
    request = urllib.request.Request(
        f"{server}/v1/completions",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "add_special_tokens": False,
                "temperature": 0.0,
                "seed": 17,
                "ignore_eos": True,
                "max_tokens": output_tokens,
                "return_token_ids": True,
                "stream": True,
            }
        ).encode(),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    text_parts: list[str] = []
    token_ids: list[int] = []
    ttft_seconds: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ")
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if "error" in chunk:
                    raise RuntimeError(f"streaming completion failed: {chunk['error']}")
                choices = chunk.get("choices", ())
                if not choices:
                    continue
                choice = choices[0]
                chunk_token_ids = choice.get("token_ids") or ()
                chunk_text = choice.get("text", "")
                if ttft_seconds is None and (chunk_token_ids or chunk_text):
                    ttft_seconds = time.monotonic() - started
                token_ids.extend(chunk_token_ids)
                text_parts.append(chunk_text)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"request failed for {server}/v1/completions ({error.code}): {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request failed for {server}/v1/completions: {error}") from error

    if ttft_seconds is None:
        raise RuntimeError(f"{label} returned no output token")
    completion = Completion(
        text="".join(text_parts),
        token_ids=tuple(token_ids),
        ttft_seconds=ttft_seconds,
        wall_seconds=time.monotonic() - started,
    )
    if len(completion.token_ids) != output_tokens:
        raise RuntimeError(
            f"{label} returned {len(completion.token_ids)} tokens; expected {output_tokens}"
        )
    if print_output:
        print(f"\n=== {label} output ===")
        print(f"TTFT: {completion.ttft_seconds:.3f} seconds")
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
    *,
    text: str | None = None,
    response_instruction: str = RESPONSE_INSTRUCTION,
) -> tuple[int, ...]:
    tokenized = _post_json(
        f"{server}/tokenize",
        {
            "model": model,
            "prompt": text if text is not None else build_measured_prompt(run_id),
            "add_special_tokens": False,
        },
        request_timeout,
    )
    response_instruction_tokens = _post_json(
        f"{server}/tokenize",
        {
            "model": model,
            "prompt": response_instruction,
            "add_special_tokens": False,
        },
        request_timeout,
    )
    prompt = select_prompt_token_ids(
        tokenized["tokens"],
        prompt_tokens,
        suffix_token_ids=response_instruction_tokens["tokens"],
    )
    max_model_len = int(tokenized["max_model_len"])
    if prompt_tokens + output_tokens > max_model_len:
        raise RuntimeError(
            f"{prompt_tokens} prompt + {output_tokens} output tokens exceed "
            f"the server maximum of {max_model_len}"
        )
    return prompt


def _verify_server(
    instance_name: str,
    server: str,
    request_timeout: float,
) -> None:
    _get_text(f"{server}/health", request_timeout)
    metrics = _get_text(f"{server}/metrics", request_timeout)
    try:
        disabled = metric_value(
            metrics,
            "vllm:cache_config_info",
            {"enable_prefix_caching": "False"},
            required=True,
        )
    except ValueError as error:
        raise RuntimeError(f"{error} from instance {instance_name} at {server}") from error
    if disabled <= 0:
        raise RuntimeError(
            f"instance {instance_name} at {server} does not report prefix caching disabled"
        )


def _prompt_preview(prompt_text: str, limit: int = 800) -> str:
    if len(prompt_text) <= limit:
        return prompt_text
    tail = 160
    return (
        f"{prompt_text[: limit - tail]}\n"
        "... [prompt shortened for display] ...\n"
        f"{prompt_text[-tail:]}"
    )


def _format_bytes(byte_count: int) -> str:
    if byte_count >= 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.1f} MiB ({byte_count:,} bytes)"
    return f"{byte_count:,} bytes"


def _transfer_summary(result: dict[str, Any]) -> str:
    if result["path"] == "local_compute_store":
        action = "STORE"
        byte_count = result["store_bytes"]
        copy_seconds = result["store_copy_seconds"]
    else:
        action = "LOAD"
        byte_count = result["load_bytes"]
        copy_seconds = result["load_copy_seconds"]
    return f"{action} {_format_bytes(byte_count)} in {copy_seconds * 1000:.3f} ms"


def _print_result(result: dict[str, Any]) -> None:
    if result["path"] == "local_compute_store":
        outcome = "COLD COMPUTE + STORE"
        source = f"{result['computed_prompt_tokens']:,} local-compute tokens"
    else:
        outcome = "EXTERNAL KV RELOAD"
        source = f"{result['loaded_prompt_tokens']:,} external-transfer tokens"

    print(f"\n=== Result: {result['label']} ===")
    print(f"{'Target':<15}{result['instance']} @ {result['endpoint']}")
    print(f"{'Identifier':<15}{result['identifier']}")
    print(f"{'Outcome':<15}{outcome}")
    print(f"{'Prompt source':<15}{source}")
    print(f"{'TTFT':<15}{result['ttft_seconds']:.3f} s")
    print(f"{'E2E':<15}{result['wall_seconds']:.3f} s")
    print(f"{'KV transfer':<15}{_transfer_summary(result)}")
    print(f"{'Output':<15}{result['output_tokens']} tokens")
    print(f"{'Token IDs':<15}{result['token_ids']}")
    print(f"{'Text':<15}{json.dumps(result['text'], ensure_ascii=False)}")


def print_request_summary(title: str, requests: list[dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    for index, result in enumerate(requests, start=1):
        if result["path"] == "local_compute_store":
            outcome = f"COLD COMPUTE + STORE ({result['computed_prompt_tokens']:,} tokens)"
        else:
            outcome = f"EXTERNAL KV RELOAD ({result['loaded_prompt_tokens']:,} tokens)"
        print(f"{index}. {result['label']}")
        print(f"   Target       {result['instance']} @ {result['endpoint']}")
        print(f"   Outcome      {outcome}")
        print(f"   TTFT / E2E   {result['ttft_seconds']:.3f} s / {result['wall_seconds']:.3f} s")
        print(f"   KV transfer  {_transfer_summary(result)}")


def execute_request(
    *,
    label: str,
    instance_name: str,
    host: str,
    port: int,
    model: str,
    identifier: str,
    prompt_source: str,
    response_instruction: str,
    prompt_tokens: int,
    output_tokens: int,
    request_timeout: float,
    metric_timeout: float,
    expected_source: str | None = None,
) -> dict[str, Any]:
    if not host.strip():
        raise ValueError("host must not be empty")
    if not 1 <= port <= 65535:
        raise ValueError(f"port must be between 1 and 65535, got {port}")
    server = f"http://{host.strip()}:{port}"
    _verify_server(instance_name, server, request_timeout)

    prompt = _prepare_prompt(
        server,
        model,
        identifier,
        prompt_tokens,
        output_tokens,
        request_timeout,
        text=prompt_source,
        response_instruction=response_instruction,
    )
    prompt_text = _post_json(
        f"{server}/detokenize",
        {"model": model, "tokens": prompt},
        request_timeout,
    )["prompt"]

    print("\n=== Request ===")
    print(f"Request: {label}")
    print(f"Instance: {instance_name}")
    print(f"Endpoint: {server}")
    print(f"Identifier: {identifier}")
    print(f"Prompt tokens: {len(prompt)}")
    print("Prompt preview:")
    print(_prompt_preview(prompt_text))
    print("\nSending request...", flush=True)

    before = _metric_snapshot(server, request_timeout)
    completion = _complete(
        label,
        server,
        model,
        prompt,
        output_tokens,
        request_timeout,
        print_output=False,
    )
    deltas = _wait_for_request_metrics(
        server,
        before,
        request_timeout,
        metric_timeout,
    )

    computed = deltas[LOCAL_COMPUTE]
    loaded = deltas[EXTERNAL_TRANSFER]
    if computed > 0 and loaded > 0:
        raise AssertionError(
            f"request mixed {computed:g} locally computed tokens with "
            f"{loaded:g} externally loaded tokens"
        )
    if computed > 0:
        actual_source = LOCAL_COMPUTE
        _require_prompt_source(label, deltas, len(prompt), actual_source)
        path = "local_compute_store"
    else:
        actual_source = EXTERNAL_TRANSFER
        _require_prompt_source(label, deltas, len(prompt), actual_source)
        path = "external_kv_reload"
    if expected_source is not None and actual_source != expected_source:
        raise AssertionError(
            f"{label} expected prompt source {expected_source}, got {actual_source}"
        )

    result: dict[str, Any] = {
        "label": label,
        "instance": instance_name,
        "endpoint": server,
        "identifier": identifier,
        "path": path,
        "prompt_tokens": len(prompt),
        "output_tokens": len(completion.token_ids),
        "ttft_seconds": completion.ttft_seconds,
        "wall_seconds": completion.wall_seconds,
        "computed_prompt_tokens": int(computed),
        "loaded_prompt_tokens": int(loaded),
        "store_bytes": int(deltas[STORE_BYTES]),
        "store_copy_seconds": deltas[STORE_TIME],
        "load_bytes": int(deltas[LOAD_BYTES]),
        "load_copy_seconds": deltas[LOAD_TIME],
        "token_ids": list(completion.token_ids),
        "text": completion.text,
    }
    _print_result(result)
    return result


def run_request(
    *,
    instance_name: str,
    host: str,
    port: int,
    model: str,
    identifier: str,
    prompt_tokens: int,
    output_tokens: int,
    request_timeout: float,
    metric_timeout: float,
) -> dict[str, Any]:
    return execute_request(
        label=f"instance {instance_name} measured request",
        instance_name=instance_name,
        host=host,
        port=port,
        model=model,
        identifier=identifier,
        prompt_source=build_measured_prompt(identifier),
        response_instruction=RESPONSE_INSTRUCTION,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        request_timeout=request_timeout,
        metric_timeout=metric_timeout,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=SETUP_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--instance", required=True, help="display name, such as A or B")
    parser.add_argument("--host", required=True, help="instance IP address or host name")
    parser.add_argument("--port", required=True, type=int, help="instance HTTP port")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    parser.add_argument(
        "--identifier",
        default=DEFAULT_IDENTIFIER,
        help="stable prompt identifier; use the same value for A, A, and B",
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--metric-timeout", type=float, default=120)
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="also print the complete machine-readable result",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_request(
        instance_name=args.instance,
        host=args.host,
        port=args.port,
        model=args.model,
        identifier=args.identifier,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        request_timeout=args.request_timeout,
        metric_timeout=args.metric_timeout,
    )
    if args.show_json:
        print("\n=== Machine-readable result ===")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
