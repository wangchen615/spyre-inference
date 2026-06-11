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

"""End-to-end M1 host-offload check on a real Spyre card.

This is the RFC A1.1 acceptance test, run in-process via the ``LLM`` API rather
than ``vllm serve`` (same engine path, easier to assert on). It is gated on a real
Spyre device and skips on CPU-only hosts.

What it asserts (the verified part of A1.1):
  - the engine BOOTS with OffloadingConnector + SpyreOffloadingSpec selected,
    i.e. our TorchSpyreModelRunner.initialize_kv_cache hook primes the spec and
    registers handlers without raising on the Spyre paged-list KV cache;
  - generation with the connector produces output BYTE-IDENTICAL to a no-connector
    baseline at temperature=0 (the connector is correct and transparent).

What it does NOT yet assert: that a KV block was physically transferred
device<->host. In this configuration the upstream scheduler's store-decision path
does not engage for small single-request workloads, so no GPU->CPU/CPU->GPU
transfer fires. See ``docs/architecture/rfcs/upstream-connector-port-TRACKING.md``
("Open question: triggering an actual transfer") and the runbook there for how to
reproduce and the current understanding. The handler exposes ``blocks_transferred``
counters and logs one line per transfer so a transfer is observable once triggered.

Prompt sizing is deliberate: this card hits a torch-spyre torch.compile
RecursionError in the Granite decode path at ~40+ tokens (reproduces with NO
connector — a torch-spyre issue, not M1's). A length sweep found ~39 tokens OK and
~52 crashing, so prompts are kept to ~34 tokens with a >1-block shared prefix.
"""

import pytest
import torch

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
MAX_LEN = 128

# ~28-token shared prefix (>1 block at block_size 16); ~34-token prompts. Stays
# under the ~40-token torch-spyre compile-recursion threshold on this card.
SHARED = (
    "Paris is the capital of France and a major European city located on "
    "the river Seine in the northern part of the country today."
)
PROMPT_A = SHARED + " Eiffel Tower."
PROMPT_B = SHARED + " Louvre museum."


def _spyre_available() -> bool:
    """Probe for a usable Spyre device.

    The ``spyre`` device only registers after ``torch_spyre._autoload()`` runs,
    which ``spyre_inference`` defers (``TORCH_DEVICE_BACKEND_AUTOLOAD=0``) until a
    worker sets the rank env vars. In a bare pytest process neither has happened,
    so we replicate the worker's bring-up (env defaults + autoload + set_device)
    before probing. This is best-effort: any failure means "no Spyre here".
    """
    try:
        import os

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        import torch_spyre

        torch_spyre._autoload()
        # Only ask whether a card exists; do NOT set_device or allocate here, so
        # we don't claim the device before the LLM's worker subprocess does.
        return torch.spyre.device_count() > 0
    except Exception:
        return False


def _make_llm(kv_cfg=None):
    from vllm import LLM
    from vllm.config import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    kwargs = dict(
        max_model_len=MAX_LEN,
        max_num_seqs=2,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )
    if kv_cfg is not None:
        kwargs["kv_transfer_config"] = kv_cfg
        # Cap GPU blocks hard so the (huge by default) Spyre KV cache exerts
        # eviction pressure; otherwise tiny prompts never touch the host tier.
        kwargs["num_gpu_blocks_override"] = 16
    return LLM(MODEL, **kwargs)


def _gen(llm):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=4)
    a = llm.generate([PROMPT_A], sp)[0].outputs[0].text
    b = llm.generate([PROMPT_B], sp)[0].outputs[0].text
    return a, b


@pytest.mark.spyre
@pytest.mark.uses_subprocess
def test_offload_connector_boots_and_matches_baseline():
    if not _spyre_available():
        pytest.skip("Spyre device not available")

    from vllm.config import KVTransferConfig

    # Baseline: no connector.
    base = _make_llm()
    base_a, base_b = _gen(base)
    del base

    # With the Spyre offloading connector selected.
    kv_cfg = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "SpyreOffloadingSpec",
            "cpu_bytes_to_use": 2_000_000_000,
        },
    )
    off = _make_llm(kv_cfg)
    off_a, off_b = _gen(off)

    # temperature=0 is deterministic: the connector must be transparent.
    assert off_a == base_a, f"prompt A diverged: base={base_a!r} off={off_a!r}"
    assert off_b == base_b, f"prompt B diverged: base={base_b!r} off={off_b!r}"
