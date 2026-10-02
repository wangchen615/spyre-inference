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

"""Cold-warm both instances with prompts unrelated to the measured prompt."""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Any

if __package__:
    from scripts import shared_kv_two_instance_demo as demo
else:
    import shared_kv_two_instance_demo as demo

WARMUP_PROMPT = """
This is unrelated warmup material for the Spyre shared-KV visual demo. It does
not contain the measured incident prompt. The request fills the configured
prefill buckets, generates deterministic output, and stores completed key and
value pages in shared host memory before anyone records performance results.
Every sentence in this paragraph is disposable warmup data.
""".strip()


def build_warmup_prompt(identifier: str) -> str:
    header = f"Warmup-only identifier: {identifier}\n\n"
    return header + (WARMUP_PROMPT + "\n\n") * 192


def run_warmup(
    *,
    instance_a_host: str,
    instance_a_port: int,
    instance_b_host: str,
    instance_b_port: int,
    model: str,
    warmup_id: str,
    prompt_tokens: int,
    output_tokens: int,
    request_timeout: float,
    metric_timeout: float,
    show_json: bool = False,
) -> dict[str, Any]:
    if (
        instance_a_host.strip().lower(),
        instance_a_port,
    ) == (
        instance_b_host.strip().lower(),
        instance_b_port,
    ):
        raise ValueError("instances A and B must use different endpoints")

    identifier_a = f"{warmup_id}-a"
    identifier_b = f"{warmup_id}-b"
    prompt_a = build_warmup_prompt(identifier_a)
    prompt_b = build_warmup_prompt(identifier_b)

    print("=== Shared-KV warmup ===")
    print(f"Warmup identifier: {warmup_id}")
    print(f"Instance A: http://{instance_a_host}:{instance_a_port}")
    print(f"Instance B: http://{instance_b_host}:{instance_b_port}")
    print("Each instance cold-computes and stores one unrelated junk prompt.")

    common: dict[str, Any] = {
        "model": model,
        "response_instruction": demo.WARMUP_RESPONSE_INSTRUCTION,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "request_timeout": request_timeout,
        "metric_timeout": metric_timeout,
    }
    requests = [
        demo.execute_request(
            label="A junk cold compute/store",
            instance_name="A",
            host=instance_a_host,
            port=instance_a_port,
            identifier=identifier_a,
            prompt_source=prompt_a,
            expected_source=demo.LOCAL_COMPUTE,
            **common,
        ),
        demo.execute_request(
            label="B junk cold compute/store",
            instance_name="B",
            host=instance_b_host,
            port=instance_b_port,
            identifier=identifier_b,
            prompt_source=prompt_b,
            expected_source=demo.LOCAL_COMPUTE,
            **common,
        ),
    ]
    result = {"warmup_id": warmup_id, "requests": requests}
    demo.print_request_summary("Warmup summary", requests)
    print("Warmup complete: 2/2 requests cold-computed and stored shared KV.")
    if show_json:
        print("\n=== Machine-readable warmup result ===")
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-a-host", required=True)
    parser.add_argument("--instance-a-port", required=True, type=int)
    parser.add_argument("--instance-b-host", required=True)
    parser.add_argument("--instance-b-port", required=True, type=int)
    parser.add_argument("--model", default=demo.DEFAULT_MODEL)
    parser.add_argument(
        "--warmup-id",
        help="optional identifier; the default is unique for every warmup run",
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--metric-timeout", type=float, default=120)
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="also print the complete machine-readable warmup result",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_warmup(
        instance_a_host=args.instance_a_host,
        instance_a_port=args.instance_a_port,
        instance_b_host=args.instance_b_host,
        instance_b_port=args.instance_b_port,
        model=args.model,
        warmup_id=args.warmup_id or uuid.uuid4().hex,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        request_timeout=args.request_timeout,
        metric_timeout=args.metric_timeout,
        show_json=args.show_json,
    )


if __name__ == "__main__":
    main()
