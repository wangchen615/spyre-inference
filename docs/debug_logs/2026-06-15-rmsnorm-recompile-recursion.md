# RMSNorm recompile → RecursionError on multi-length prompts

- **Date:** 2026-06-15
- **Branch:** `dev/kv-offload-m1-yzhu`
- **Status:** Root cause identified; **no fix applied** (design choice pending). Repo is at a clean baseline.
- **Captured output:** `docs/debug_logs/2026-06-15-rmsnorm-recompile-recursion-pytest-output.txt`

---

## Symptom

```
tests/v1/test_prefix_caching_with_eviction.py::test_kv_eviction_under_memory_pressure_with_prefix_caching
```

```bash
TORCH_DYNAMO_DISABLE=1 uv run pytest tests/v1/test_prefix_caching_with_eviction.py -v -s -m "not upstream"
```

The test issues **7 sequential requests** ("Steps", `max_num_seqs=1`), reusing 5 prompts of distinct
token lengths (A=45, B=40, C=46, D=37, E=48). Steps and prompts aren't 1:1 — Steps 2 and 7 reuse
Prompt A — so "Step 5" is the 5th request, which uses Prompt D:

| Step | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|------|---|---|---|---|---|---|---|
| Prompt | A (cold) | A (cached) | B | C | **D** | E | A |
| Result | ok | ok | ok | ok | **CRASH** | — | — |

Step 5 fails with `RecursionError: maximum recursion depth exceeded`, immediately preceded by
`torch._dynamo hit config.accumulated_recompile_limit (256)`. Steps 6–7 never run.

**KV offloading was OFF for this run.** The test defines an offload connector but the enabling line is
commented out (`#kv_transfer_config=kv_config`), so offloading is not involved — consistent with the
root cause below.

---

## Root cause (confirmed)

The infinite recursion lives **inside torch-spyre's per-op kernel dispatch**, not in our code:

1. RMSNorm input `x` is `[num_tokens, hidden_size]`; `num_tokens` changes with each new prompt length.
2. `SpyreRMSNorm.forward_spyre` calls `torch.rsqrt(...)` on a Spyre tensor.
3. torch-spyre registers `aten.rsqrt` (and ~40 other ops) as a kernel that is **itself
   `torch.compile`d with `dynamic=False`** (`compile_once` in `torch_spyre/ops/eager.py`).
4. `dynamic=False` ⇒ each new shape recompiles that kernel. Distinct prompt lengths therefore recompile
   `rsqrt` (and friends) repeatedly, across every layer and decode step.
5. These accumulate against dynamo's `accumulated_recompile_limit = 256`.
6. On hitting the limit, `skip_code_recursive_on_recompile_limit_hit = True` makes dynamo re-enter the
   just-disabled frame — but that frame is torch-spyre's own self-recompiling wrapper, so it re-enters
   its own compile path and loops forever → `RecursionError`.

**Evidence:** in the captured tail, the torch-spyre/dynamo cycle frames repeat ~244 times while our
`rms_norm.py` appears only once (the entry, not the loop), and dynamo names the recompiled op as
`aten.rsqrt`. The crash step (5) reflects when the *cumulative* recompile count crosses 256 — it is not
specific to Prompt D.

This matches the project guidance (CLAUDE.md / debug-spyre skill): most "Spyre is broken" bugs are
torch-spyre op/shape limitations, not local bugs.

---

## Error trace (excerpt)

Trimmed and shown repo-relative; the raw file has absolute paths and `(Worker pid=...)` prefixes.
Line numbers (`rms_norm.py:169`/`:208`) are from the Attempt-1 run, whose diagnostic patch shifted
lines down ~11; on the clean tree the `rsqrt` call is at `rms_norm.py:158`.

```
torch._dynamo hit config.accumulated_recompile_limit (256)
   last reason: 0/255: ___check_obj_id(fn, ...), type=<OpOverload(op='aten.rsqrt', overload='default')>
...
File "spyre_inference/custom_ops/rms_norm.py", line 169, in forward_spyre
    x = x * torch.rsqrt(variance + variance_epsilon_t)
File ".venv/.../torch_spyre/ops/eager.py", line 40, in wrapper
    return fn(*args, compiled=compiled, **kwargs)
File ".venv/.../torch_spyre/ops/eager.py", line 62, in dispatch_to_torch_compile
    return compiled(*args, **kwargs)
   ... [these frames repeat ~244 times] ...
RecursionError: maximum recursion depth exceeded
```

---

## What we tried

**Attempt 1 — run `forward_spyre` eagerly (bypass `maybe_compile`).**

Edited one file, `spyre_inference/custom_ops/rms_norm.py`, in `SpyreRMSNorm.__init__` (line 79).
Replaced the `maybe_compile` wrap with a direct assignment, and added a one-line diagnostic to log
whether `maybe_compile` would have actually wrapped the function:

```python
# before
self.maybe_compiled_forward_spyre = self.maybe_compile(self.forward_spyre)

# after (Attempt 1)
_probe = self.maybe_compile(self.forward_spyre)
logger.warning_once(
    "SpyreRMSNorm DIAGNOSTIC: maybe_compile wrapped forward_spyre = %s",
    _probe is not self.forward_spyre,
)
self.maybe_compiled_forward_spyre = self.forward_spyre   # run eagerly
```

**Result:** no effect — still crashed identically at Step 5. The diagnostic printed
`maybe_compile wrapped forward_spyre = False`, i.e. `maybe_compile` was *already* returning the
uncompiled function (`compilation_config.mode == CompilationMode.NONE`), so `forward_spyre` was never
being compiled in the first place. This confirmed the recompiling code object is torch-spyre's `rsqrt`
kernel, not our wrapper. **Both edits reverted**; `rms_norm.py` is back to the clean baseline (line 79
as shown in "before").

---

## Candidate fixes (none attempted yet)

1. **Pad/bucket `num_tokens` to a fixed size before RMSNorm** (in-tree). Aligning the token dim to a
   constant bucket gives one shape → one compile → no churn — the same tactic the attention backend
   already uses for KV length / query chunks. Preferred: matches existing repo patterns.
2. **Make the torch-spyre kernel shape-dynamic** (`compile_once(..., dynamic=True)`). Lives in
   installed site-packages, so it needs a torch-spyre change + standalone repro, and confirmation that
   Spyre supports a dynamic dim for these ops.
3. **Raise/disable the dynamo recompile limit** — mitigation only; delays the crash and leaves the
   per-call recompiles as a perf sink. Not recommended.

---

## Reproduction / key files

- Repro: the pytest command above, on Spyre hardware (single accelerator — never run two Spyre jobs at once).
- Captured output: `docs/debug_logs/2026-06-15-rmsnorm-recompile-recursion-pytest-output.txt`.
- Recompile constants: torch `_dynamo/config.py` (`accumulated_recompile_limit = 256`,
  `skip_code_recursive_on_recompile_limit_hit = True`).
- torch-spyre kernels: `torch_spyre/ops/eager.py` (`compile_once`, `register_torch_compile_kernel`).
- Our RMSNorm: `spyre_inference/custom_ops/rms_norm.py` (`forward_spyre`).

---

## Next actions

1. Choose fix #1 (pad/bucket, in-tree) vs #2 (dynamic kernel, torch-spyre escalation).
2. Apply, then re-run the full test; success = all 7 requests complete **and** the
   `accumulated_recompile_limit` warning is gone (separate from the test's own eviction/offload assertions).
