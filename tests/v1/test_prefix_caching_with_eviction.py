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

"""Verify KV cache eviction happens under memory pressure with prefix caching."""

import os

import pytest

from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig, KVTransferConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

# Five distinct prompts, each ~150 tokens. Block size on Spyre is 128 tokens and
# the offloading scheduler only stores COMPLETE blocks (num_tokens // 128), so a
# prompt MUST exceed 128 committed tokens to produce a single offloadable block.
# Short (~40-token) prompts never fill a block and are never offloaded -- hence
# the deliberately long text below. With prefix caching disabled, each distinct
# prompt allocates fresh blocks, so the 4 GPU blocks (512 tokens) cannot hold all
# five, forcing the scheduler to offload device->host (GPU->CPU).
PROMPTS = {
    "A": "Paris is the capital and most populous city of France, situated on the banks of "
         "the river Seine in the north of the country. For centuries it has been a global "
         "center of art, fashion, gastronomy, and culture. Its nineteenth-century cityscape "
         "is crossed by wide boulevards and the river Seine. Beyond landmarks such as the "
         "Eiffel Tower and the Gothic Notre-Dame cathedral, the city is renowned for its cafe "
         "culture, its many museums including the Louvre and the Musee d'Orsay, and its "
         "reputation as a destination for romance, learning, and the arts throughout modern "
         "European history.",
    "B": "Tokyo is the capital and largest metropolitan area of Japan, located at the head of "
         "Tokyo Bay on the eastern coast of the main island of Honshu. Once a small fishing "
         "village known as Edo, it grew into one of the most populous and economically "
         "powerful urban regions in the world. The city blends ancient temples and quiet "
         "gardens with dense districts of neon, commerce, and technology. It serves as the "
         "political, financial, and cultural heart of the nation, hosting government "
         "institutions, global corporations, world-class transit, and a cuisine celebrated "
         "across the entire planet for its precision.",
    "C": "London is the capital and largest city of England and the United Kingdom, standing "
         "on the river Thames in the south-east of the island of Great Britain. With a history "
         "spanning nearly two millennia since its founding by the Romans as Londinium, it has "
         "grown into a leading global city for finance, commerce, law, education, and the arts. "
         "Famous landmarks include Big Ben, the Tower of London, Tower Bridge, Westminster "
         "Abbey, and Buckingham Palace. Its many museums, theatres, parks, and universities "
         "draw millions of visitors and students every year from every corner of the wider "
         "connected world.",
    "D": "Berlin is the capital and largest city of Germany, lying on the banks of the river "
         "Spree in the north-eastern part of the country. Once divided by a wall that came to "
         "symbolize the wider Cold War, it is today a unified, vibrant metropolis celebrated "
         "for its layered history, its progressive culture, and its thriving arts scene. The "
         "city is dotted with monuments, memorials, and museums that document centuries of "
         "triumph and tragedy, from the Brandenburg Gate to Museum Island. It has also become "
         "a magnet for artists, musicians, startups, and students drawn by its openness and "
         "its restless energy.",
    "E": "Rome is the capital city of Italy and a special municipality lying along the banks "
         "of the river Tiber in the central western portion of the Italian peninsula. As the "
         "former heart of the vast Roman Empire, it is often called the Eternal City and "
         "contains an extraordinary concentration of ancient ruins, monuments, and works of "
         "art. Visitors come to see the Colosseum, the Roman Forum, the Pantheon, and the many "
         "fountains and piazzas that fill the historic center. Surrounding the independent "
         "enclave of Vatican City, Rome remains a profound center of religion, history, "
         "architecture, and Renaissance art.",
}


def _spyre_available() -> bool:
    try:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        import torch_spyre
        torch_spyre._autoload()
        import torch
        return torch.spyre.device_count() > 0
    except Exception:
        return False


@pytest.mark.spyre
@pytest.mark.uses_subprocess
def test_kv_eviction_under_memory_pressure_with_prefix_caching(capfd):
    """KV cache eviction must occur under memory pressure, and an evicted
    sequence must reload from host when re-requested.

    Setup: max_num_seqs=1 and num_gpu_blocks_override=4 (4 * 128 = 512 tokens).
    Block size is 128 tokens and the scheduler offloads only complete blocks, so
    each prompt is ~150 tokens (>128) to produce one offloadable block. Prefix
    caching is disabled so each distinct prompt allocates fresh blocks; the five
    distinct sequences cannot all fit in 4 blocks, forcing GPU->CPU eviction.
    Re-requesting Prompt A after the others have evicted it must trigger a
    CPU->GPU reload from host.

    The SpyreOffloadingHandler "GPU->CPU" / "CPU->GPU" transfer lines are emitted
    in the worker subprocess, so they are NOT visible to caplog (which only hooks
    Python logging in the test process). vLLM forwards worker stdout/stderr to the
    parent's file descriptors (the "(Worker pid=...)" prefix), so we capture them
    with capfd (fd-level) instead. NOTE: capfd is disabled under `pytest -s`; run
    this test WITHOUT -s or the captured output will be empty.
    """
    if not _spyre_available():
        pytest.skip("Spyre device not available")

    kv_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "SpyreOffloadingSpec",
            "cpu_bytes_to_use": 2_000_000_000,  # 2GB host memory
        },
    )

    model = LLM(
        MODEL,
        max_model_len=512,
        max_num_seqs=1,  # one seq at a time -> sequential, forces eviction
        num_gpu_blocks_override=4,  # minimum required; 4 * 128 = 512 tokens
        enable_prefix_caching=False,  # each request allocates fresh blocks
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
        kv_transfer_config=kv_config,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=3, min_tokens=1)

    # Drain any output produced during LLM(...) construction so the first
    # run_step only sees lines from its own generate() call.
    capfd.readouterr()

    def run_step(prompt):
        out = model.generate(prompts=prompt, sampling_params=sampling_params)
        # readouterr() returns everything written to fd 1/2 since the last call
        # and drains the buffer, so each step sees only its own transfer lines.
        captured = capfd.readouterr()
        text = captured.out + captured.err
        evict = [ln for ln in text.splitlines() if "GPU->CPU" in ln]
        reload = [ln for ln in text.splitlines() if "CPU->GPU" in ln]
        return out, evict, reload

    # A -> B -> C -> D -> E overflows the 4 GPU blocks, then A again reloads.
    steps = [PROMPTS[k] for k in ("A", "B", "C", "D", "E", "A")]
    results = [run_step(p) for p in steps]

    total_evictions = sum(len(evict) for _, evict, _ in results)
    total_reloads = sum(len(reload) for _, _, reload in results)

    # Sanity: all requests produced output.
    for i, (out, _, _) in enumerate(results, start=1):
        assert out[0].outputs[0].text is not None, f"Request {i} failed"

    assert total_evictions > 0, (
        "GPU->CPU eviction NOT detected. Possible causes:\n"
        "  1. Memory pressure was not sufficient\n"
        "  2. Offloading connector not active\n"
        "  3. GPU blocks override not enforced\n"
        "  4. Worker subprocess logs not captured by caplog"
    )

    assert total_reloads > 0, (
        "CPU->GPU reload NOT detected when reusing Prompt A after evictions. "
        "Possible causes:\n"
        "  1. Evicted cache was not saved to host\n"
        "  2. Reload handler not triggered or not logging\n"
        "  3. Prompt A was recomputed instead of reloaded from host"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "not upstream"])
