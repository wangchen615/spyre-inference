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
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

import time
import pytest

from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig, KVTransferConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


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
def test_kv_eviction_under_memory_pressure_with_prefix_caching(caplog):
    """
    Test that KV cache eviction happens under memory pressure with prefix caching.

    Strategy: Use shorter prompts (30-40 tokens) and tight GPU blocks (4 blocks).
    With max_num_seqs=1, each new request forces the previous one to be evicted.

    Block size details:
    - Actual block size on Spyre: 128 tokens per block
    - With num_gpu_blocks_override=2: 2 blocks * 128 tokens/block = 256 tokens total on GPU
    - Model has 4 layers → each block eviction is logged as 4 logical blocks
    - Expected memory math:
      * Prompt A (45 tokens): fits in 1 block (128 tokens)
      * Prompt B (40 tokens): fits in 1 block (128 tokens)
      * GPU limit: only 2 blocks (256 tokens) available
      * When Prompt B arrives: Prompt A should be evicted to host (offload)
      * When Prompt C arrives: Prompt B should be evicted to host (offload)
    - If NO eviction logs appear: either memory pressure is insufficient OR
      eviction logic isn't being triggered (scheduler not requesting eviction)

    Scenario:
    1. Load request with prompt A (cold)
    2. Load request with same prompt A again (prefix cached, should be fast)
    3. Load request with different prompt B (triggers eviction of A?)
    4. Load request with different prompt C (triggers eviction of B?)

    The test verifies:
    - Prefix caching speedup between steps 1 and 2 (>1.2x)
    - Eviction logs ("GPU->CPU") appear when loading new sequences under pressure
    - Token counts help diagnose why eviction may or may not happen
    """
    if not _spyre_available():
        pytest.skip("Spyre device not available")

    # Use SHORTER prompts (30-40 tokens, safe and won't crash)
    prompt_a = (
        "Paris is the capital and largest city of France. It is situated on the Seine River. "
        "Famous for landmarks like the Eiffel Tower and the Louvre Museum."
    )

    prompt_b = (
        "Tokyo is the capital and largest metropolitan area of Japan. Located on the eastern coast "
        "of Kanto Plain. One of the world's largest and most developed cities."
    )

    prompt_c = (
        "London is the capital and largest city of the United Kingdom. Situated on the River Thames. "
        "Famous for landmarks like Big Ben, Tower Bridge, and Buckingham Palace."
    )

    prompt_d = (
        "Berlin is the capital and largest city of Germany. Located on the Spree River. "
        "Known for historical sites, museums, and vibrant cultural scene."
    )

    prompt_e = (
        "Rome is the capital and largest city of Italy. Situated on the Tiber River. "
        "Famous for ancient history, the Colosseum, Vatican City, and Renaissance art."
    )

    print(f"\n{'='*80}")
    print("TEST: KV cache eviction under memory pressure + prefix caching")
    print(f"{'='*80}")
    print(f"Strategy: Use SHORT prompts (30-40 tokens) + max_num_seqs=1")
    print(f"  Each new request forces eviction of the previous one")
    print(f"GPU block limit override: 4 blocks (minimum required)")
    print(f"{'='*80}\n")

    # Configure KV offload
    kv_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "SpyreOffloadingSpec",
            "cpu_bytes_to_use": 2_000_000_000,  # 2GB host memory
        },
    )

    # Create LLM with:
    # - max_num_seqs=1: Only one sequence at a time (forces eviction)
    # - num_gpu_blocks_override=4: Minimum required by program (4 blocks * 128 tokens = 512 tokens)
    model = LLM(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        max_model_len=512,
        max_num_seqs=1,  # ← KEY: Only 1 seq at a time, forces sequential + eviction
        num_gpu_blocks_override=4,  # ← MINIMUM REQUIRED
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum["CUSTOM"]
        ),
        #kv_transfer_config=kv_config,
    )

    # Capture logs at INFO level
    import logging
    caplog.set_level(logging.INFO)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        min_tokens=1,
    )

    # Tokenize prompts to get exact token counts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    )

    tokens_a = tokenizer.encode(prompt_a)
    tokens_b = tokenizer.encode(prompt_b)
    tokens_c = tokenizer.encode(prompt_c)
    tokens_d = tokenizer.encode(prompt_d)
    tokens_e = tokenizer.encode(prompt_e)

    print(f"Token counts:")
    print(f"  Prompt A: {len(tokens_a)} tokens")
    print(f"  Prompt B: {len(tokens_b)} tokens")
    print(f"  Prompt C: {len(tokens_c)} tokens")
    print(f"  Prompt D: {len(tokens_d)} tokens")
    print(f"  Prompt E: {len(tokens_e)} tokens")
    print()

    try:
        # === Request 1: Prompt A (cold) ===
        print("Step 1: First request (Prompt A, cold)")
        caplog.clear()
        start = time.time()
        output1 = model.generate(prompts=prompt_a, sampling_params=sampling_params)
        ttft1 = time.time() - start
        print(f"  TTFT: {ttft1:.4f}s")
        print(f"  Output: {output1[0].outputs[0].text}")

        # === Request 2: Prompt A again (prefix cached) ===
        print("\nStep 2: Second request (Prompt A cached)")
        caplog.clear()
        start = time.time()
        output2 = model.generate(prompts=prompt_a, sampling_params=sampling_params)
        ttft2 = time.time() - start
        print(f"  TTFT: {ttft2:.4f}s")
        print(f"  Output: {output2[0].outputs[0].text}")

        speedup_2 = ttft1 / ttft2 if ttft2 > 0 else 1.0
        print(f"  Speedup vs Step1: {speedup_2:.2f}x")
        if speedup_2 > 1.2:
            print(f"  ✓ Prefix caching ENABLED (>1.2x speedup detected)")
        else:
            print(f"  ⚠ Speedup modest, cache effect may be limited")

        # === Request 3: Prompt B (different, should trigger eviction of A) ===
        print("\nStep 3: Third request (Prompt B - SHOULD EVICT Prompt A)")
        print(f"  Expected: GPU->CPU log (Prompt A evicted to host)")
        caplog.clear()
        start = time.time()
        output3 = model.generate(prompts=prompt_b, sampling_params=sampling_params)
        ttft3 = time.time() - start
        print(f"  TTFT: {ttft3:.4f}s")
        print(f"  Output: {output3[0].outputs[0].text}")

        # Check for GPU->CPU in caplog records
        gpu_to_cpu_logs_3 = [
            r for r in caplog.records
            if "GPU->CPU" in r.message
        ]
        print(f"  GPU->CPU logs found: {len(gpu_to_cpu_logs_3)}")
        for log in gpu_to_cpu_logs_3:
            print(f"    → {log.message}")

        # === Request 4: Prompt C (different, should trigger eviction of B) ===
        print("\nStep 4: Fourth request (Prompt C - SHOULD EVICT Prompt B)")
        print(f"  Expected: GPU->CPU log (Prompt B evicted to host)")
        caplog.clear()
        start = time.time()
        output4 = model.generate(prompts=prompt_c, sampling_params=sampling_params)
        ttft4 = time.time() - start
        print(f"  TTFT: {ttft4:.4f}s")
        print(f"  Output: {output4[0].outputs[0].text}")

        gpu_to_cpu_logs_4 = [
            r for r in caplog.records
            if "GPU->CPU" in r.message
        ]
        print(f"  GPU->CPU logs found: {len(gpu_to_cpu_logs_4)}")
        for log in gpu_to_cpu_logs_4:
            print(f"    → {log.message}")

        # === Request 5: Prompt D (different, should trigger eviction) ===
        print("\nStep 5: Fifth request (Prompt D - SHOULD EVICT Prompt C)")
        print(f"  Expected: GPU->CPU log (Prompt C evicted to host)")
        caplog.clear()
        start = time.time()
        output5 = model.generate(prompts=prompt_d, sampling_params=sampling_params)
        ttft5 = time.time() - start
        print(f"  TTFT: {ttft5:.4f}s")
        print(f"  Output: {output5[0].outputs[0].text}")

        gpu_to_cpu_logs_5 = [
            r for r in caplog.records
            if "GPU->CPU" in r.message
        ]
        print(f"  GPU->CPU logs found: {len(gpu_to_cpu_logs_5)}")
        for log in gpu_to_cpu_logs_5:
            print(f"    → {log.message}")

        # === Request 6: Prompt E (different, should trigger eviction) ===
        print("\nStep 6: Sixth request (Prompt E - SHOULD EVICT Prompt D)")
        print(f"  Expected: GPU->CPU log (Prompt D evicted to host)")
        caplog.clear()
        start = time.time()
        output6 = model.generate(prompts=prompt_e, sampling_params=sampling_params)
        ttft6 = time.time() - start
        print(f"  TTFT: {ttft6:.4f}s")
        print(f"  Output: {output6[0].outputs[0].text}")

        gpu_to_cpu_logs_6 = [
            r for r in caplog.records
            if "GPU->CPU" in r.message
        ]
        print(f"  GPU->CPU logs found: {len(gpu_to_cpu_logs_6)}")
        for log in gpu_to_cpu_logs_6:
            print(f"    → {log.message}")

        # === Request 7: Prompt A again (after eviction, should reload from host) ===
        print("\nStep 7: Seventh request (Prompt A - reuse after eviction)")
        print(f"  Expected: CPU->GPU log (Prompt A reloaded from host)")
        caplog.clear()
        start = time.time()
        output7 = model.generate(prompts=prompt_a, sampling_params=sampling_params)
        ttft7 = time.time() - start
        print(f"  TTFT: {ttft7:.4f}s")
        print(f"  Output: {output7[0].outputs[0].text}")

        cpu_to_gpu_logs_7 = [
            r for r in caplog.records
            if "CPU->GPU" in r.message
        ]
        print(f"  CPU->GPU logs found: {len(cpu_to_gpu_logs_7)}")
        for log in cpu_to_gpu_logs_7:
            print(f"    → {log.message}")

        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Token counts and timing:")
        print(f"\nTiming:")
        print(f"  TTFT_1 (Prompt A, cold):     {ttft1:.4f}s")
        print(f"  TTFT_2 (Prompt A, cached):   {ttft2:.4f}s ({speedup_2:.2f}x speedup)")
        print(f"  TTFT_3 (Prompt B, cold):     {ttft3:.4f}s")
        print(f"  TTFT_4 (Prompt C, cold):     {ttft4:.4f}s")
        print(f"  TTFT_5 (Prompt D, cold):     {ttft5:.4f}s")
        print(f"  TTFT_6 (Prompt E, cold):     {ttft6:.4f}s")
        print(f"  TTFT_7 (Prompt A, reloaded): {ttft7:.4f}s")
        print(f"\nEviction evidence:")
        total_evictions = len(gpu_to_cpu_logs_3) + len(gpu_to_cpu_logs_4) + len(gpu_to_cpu_logs_5) + len(gpu_to_cpu_logs_6)
        total_reloads = len(cpu_to_gpu_logs_7)
        print(f"  GPU->CPU (evictions):      {'✓ YES' if total_evictions > 0 else '✗ NO'} ({total_evictions} total)")
        print(f"  CPU->GPU (reloads):        {'✓ YES' if total_reloads > 0 else '✗ NO'} ({total_reloads} total)")
        print(f"  Prefix cache:              {'✓ YES' if speedup_2 > 1.2 else '⚠ UNCLEAR'}")
        print(f"\nDiagnosis:")
        print(f"  GPU block limit: 4 blocks")
        print(f"  Total GPU capacity: 4 * 128 = 512 tokens")
        print(f"  Prompts A-E: ~{len(tokens_a)}-{len(tokens_e)} tokens each")
        print(f"  Strategy: 5 sequential requests (A→B→C→D→E) force eviction")
        print(f"  Then request Prompt A again (step 7) to verify reload from host")
        if total_evictions == 0:
            print(f"  ⚠ NO EVICTION DETECTED - This means:")
            print(f"    1. Scheduler may not be requesting eviction from the offloading manager")
            print(f"    2. Or eviction handler wasn't called despite memory pressure")
            print(f"    3. Check if OffloadingConnector is actually active in vLLM scheduler")
        print(f"{'='*80}\n")

        # Assertions
        assert ttft1 > 0, "Request 1 failed"
        assert ttft2 > 0, "Request 2 failed"
        assert ttft3 > 0, "Request 3 failed"
        assert ttft4 > 0, "Request 4 failed"
        assert ttft5 > 0, "Request 5 failed"
        assert ttft6 > 0, "Request 6 failed"
        assert ttft7 > 0, "Request 7 failed"

        # Prefix caching should show speedup
        assert speedup_2 > 1.0, (
            f"No speedup detected (Req1: {ttft1:.4f}s, Req2: {ttft2:.4f}s). "
            f"Prefix caching may not be enabled."
        )

        # CRITICAL: Must find GPU->CPU eviction (across steps 3-6)
        assert total_evictions > 0, (
            f"GPU->CPU eviction NOT detected across all steps.\n"
            f"This suggests either:\n"
            f"  1. Memory pressure was not sufficient\n"
            f"  2. Offloading connector not active\n"
            f"  3. GPU blocks override not enforced\n"
            f"  4. Handler logging not captured"
        )

        # CRITICAL: Must find CPU->GPU reload when reusing Prompt A in step 7
        assert total_reloads > 0, (
            f"CPU->GPU reload NOT detected in step 7 (Prompt A reuse after evictions).\n"
            f"This suggests either:\n"
            f"  1. Evicted cache was not properly saved to host\n"
            f"  2. Reload handler not triggered or not logging\n"
            f"  3. Prompt A was treated as a new sequence instead of reuse"
        )

        print("✓ TEST PASSED: KV cache eviction and reload verified with 5 prompts + reuse")

    finally:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "not upstream"])
