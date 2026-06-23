# RMSNorm recompile → RecursionError on multi-length prompts

- **First observed:** 2026-06-15
- **Branch:** `dev/kv-offload-m1-yzhu`
- **Current status:** **Raising the Dynamo recompile limit clears the crash** — the end-to-end run no
  longer hits the `RecursionError` (it now only fails on an unrelated eviction assertion). This is the
  current working fix; see [2026-06-23 — raise Dynamo recompile/cache limits](#2026-06-23--raise-dynamo-recompilecache-limits-mitigation-clears-the-crash).
  Note it is a *mitigation* (recompiles still happen, under budget). A complementary root-cause fix —
  padding RMSNorm's token dim — is in [2026-06-22 — pad/bucket fix](#2026-06-22--padbucket-fix-applied-crash-moved-to-a-second-op);
  it removes the RMSNorm recompiles but not the attention `spyre::overwrite` ones, so on its own the
  crash moves to that second op.
- **Repro:**
  ```bash
  TORCH_DEVICE_BACKEND_AUTOLOAD=1 TORCH_DYNAMO_DISABLE=1 \
    uv run pytest tests/v1/test_prefix_caching_with_eviction.py -v -s -m "not upstream"
  ```
- **Artifacts:**
  - `docs/debug_logs/2026-06-15-rmsnorm-recompile-recursion-pytest-output.txt` (original crash)
  - `.claude/skills/debug-spyre/logs/rmsnorm-pad-postfix.txt`, `…-recheck.txt` (post-fix runs)

> This file tracks the issue over time. **Status log is newest-first; scroll to the bottom for the
> original observation and root-cause analysis (stable reference).**

---

## Status log (newest first)

### 2026-06-23 — raise Dynamo recompile/cache limits (mitigation; clears the crash)

Parallel attempt, separate session. Instead of removing the recompiles, raise the limits so the long
eviction run never trips them. This is global, so it also covers the attention `spyre::overwrite` site
(unlike the RMSNorm-only padding). It is a **mitigation**, not a root-cause fix — the per-step
recompiles still happen, just under budget.

**What we changed.**

- `spyre_inference/v1/worker/spyre_worker.py` — in `init_device()`, right after
  `torch_spyre._autoload()`, raise both limits to **4096** and log before→after:
  ```python
  torch._dynamo.config.cache_size_limit = 4096            # was 1024 (set by torch_spyre)
  torch._dynamo.config.accumulated_recompile_limit = 4096 # was 256  (PyTorch default)
  ```
  Key point: this must run **in the worker process** (which traces the model), not the test/engine
  process. The worker is a subprocess, so a test-file-only override would not take effect.
- `tests/v1/test_prefix_caching_with_eviction.py` — removed
  `os.environ["TORCH_DYNAMO_DISABLE"] = "1"` so Dynamo is actually active (otherwise the limits are
  irrelevant), and added a print of the limits as the test process sees them.

**What we tested (real Spyre hardware).**

- The run reached all 7 requests and got to the assertions — **no `RecursionError` / recompile-limit
  crash** (the original symptom is gone).
- Worker log confirms the override landed: `Dynamo limits set in worker: cache_size_limit 1024 -> 4096,
  accumulated_recompile_limit 256 -> 4096`. The test process still prints `1024 / 256`, proving the
  worker is a separate process and a test-only edit wouldn't have worked.
- The test still **fails on the eviction assertion** (`total_evictions > 0`) — unrelated to recompiles:
  `kv_transfer_config=kv_config` is commented out, so offloading isn't enabled. Pre-existing test-setup
  issue, not a compilation problem.

**Status of these edits:** currently live in the working tree (`spyre_worker.py`,
`test_prefix_caching_with_eviction.py`), not yet committed.

**Relation to the 2026-06-22 padding fix.** Complementary, not competing: padding *eliminates* RMSNorm
recompiles (real fix, RMSNorm only); raising the limit *tolerates* recompiles everywhere (mitigation,
global). The limit-raise is what got the whole test past the recursion, including the attention
`overwrite` site the padding alone didn't cover.

### 2026-06-22 — pad/bucket fix applied; crash moved to a second op

**What we changed.** Padded RMSNorm's token (row) dimension up to a multiple of 32 so the Spyre
kernels see one fixed shape per bucket instead of recompiling on every new prompt/decode length.

- `spyre_inference/custom_ops/rms_norm.py` — added `TOKEN_DIM_ALIGNMENT = 32`; in
  `_forward_spyre_impl`, pad `x` (and `residual`) to the next multiple of 32 before the kernel call,
  then trim the extra rows off the result. Safe because RMSNorm normalizes each row independently.
  The compiled math (`forward_spyre`) was not touched.
- `tests/test_rms_norm.py` — added a test (written first, watched fail) for token counts
  1 / 31 / 33 / 45 / 64 / 65 with and without residual. Checks the kernel receives a 32-aligned shape
  and the output keeps the original row count and matches the CPU reference.

> **Snapshot of this attempt:** the exact diff and full copies of both changed files are saved at
> `rmsnorm-recompile-recursion/2026-06-22-pad-bucket/` (`changes.diff`, `rms_norm.py`,
> `test_rms_norm.py`, `README.md`). The live code stays in `spyre_inference/` and `tests/`; the
> snapshot is reference-only so this attempt is reproducible if the tree changes or the edits revert.

**Environment note (needed to run any Spyre test on this node).** Tests first failed with
`RuntimeError: ... device type ... : spyre`. The `spyre` device only registers when
`TORCH_DEVICE_BACKEND_AUTOLOAD=1`; it was `0` here. **Prefix Spyre test commands with
`TORCH_DEVICE_BACKEND_AUTOLOAD=1`.** Unrelated to this bug or the fix.

**What we tested (real Spyre hardware).**

- Baseline before the fix: `10 passed`.
- After the fix, with CPU fallback turned into an error
  (`-W "error::torch_spyre.ops.fallbacks.FallbackWarning"`): `22 passed` (10 original + 12 new) — the
  padding runs on-device, not via a CPU fallback.
- End-to-end repro, run twice: still fails, but **no longer in RMSNorm**.

**What the post-fix run shows (verified against the trace).**

- Still crashes at **Step 5**, still a `RecursionError` preceded by
  `accumulated_recompile_limit (256)` — same mechanism as the original.
- **RMSNorm is gone from the trace** — `rms_norm.py` appears 0 times in the worker traceback.
- The recursion is now entirely in the **attention KV-cache scatter**: repeating frames are
  `spyre_attn.py:87 _overwrite` → `torch.ops.spyre.overwrite` →
  `torch_spyre/_inductor/customops.py:270` (`overwrite` frame repeats ~214 times).
- The recompile is triggered by a **varying argument, not a tensor shape**: dynamo's last reason is
  `args[3][0] == 15` — a guard on the `offsets` passed to `overwrite`
  (`_overwrite(tok, output, [0], [q_start + i])`, so `offsets` changes every token).

**Conclusion.** RMSNorm recompile-recursion is fixed. The end-to-end test still fails because the same
pattern exists in a second op — the attention `spyre::overwrite` scatter, recompiling on a per-step
`offsets` value instead of a shape. That is a separate issue to investigate next, likely needing a
torch-spyre escalation since `overwrite` is their custom op.

### 2026-06-15 — Attempt 1: run `forward_spyre` eagerly (bypass `maybe_compile`)

Edited `SpyreRMSNorm.__init__` to skip the `maybe_compile` wrap and run `forward_spyre` directly,
plus a one-line diagnostic logging whether `maybe_compile` would have wrapped it.

**Result:** no effect — still crashed identically at Step 5. The diagnostic printed
`maybe_compile wrapped forward_spyre = False`, i.e. `maybe_compile` was *already* returning the
uncompiled function (`compilation_config.mode == CompilationMode.NONE`), so `forward_spyre` was never
being compiled. This confirmed the recompiling code object is torch-spyre's `rsqrt` kernel, not our
wrapper. Both edits reverted.

---

## Reference — original observation & root cause (2026-06-15)

### Symptom

```
tests/v1/test_prefix_caching_with_eviction.py::test_kv_eviction_under_memory_pressure_with_prefix_caching
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

### Root cause (confirmed)

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

### Error trace (excerpt)

Trimmed and shown repo-relative; the raw file has absolute paths and `(Worker pid=...)` prefixes.
Line numbers here are from the Attempt-1 run, whose diagnostic patch shifted lines down ~11; on the
clean tree the `rsqrt` call is at `rms_norm.py:158`.

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

### Candidate fixes considered

1. **Pad/bucket `num_tokens` to a fixed size before RMSNorm** (in-tree). Aligning the token dim to a
   constant bucket gives one shape → one compile → no churn — the same tactic the attention backend
   already uses for KV length / query chunks. **← chosen 2026-06-22.**
2. **Make the torch-spyre kernel shape-dynamic** (`compile_once(..., dynamic=True)`). Lives in
   installed site-packages, so it needs a torch-spyre change + standalone repro, and confirmation that
   Spyre supports a dynamic dim for these ops. Rejected for this change (external pinned dependency);
   still the right long-term upstream fix.
3. **Raise/disable the dynamo recompile limit** — mitigation only; delays the crash and leaves the
   per-call recompiles as a perf sink. Not recommended.

### Key files

- Recompile constants: torch `_dynamo/config.py` (`accumulated_recompile_limit = 256`,
  `skip_code_recursive_on_recompile_limit_hit = True`).
- torch-spyre kernels: `torch_spyre/ops/eager.py` (`compile_once`, `register_torch_compile_kernel`).
- Our RMSNorm: `spyre_inference/custom_ops/rms_norm.py` (`forward_spyre`, `_forward_spyre_impl`).
- Attention scatter (current failure site): `spyre_inference/v1/attention/backends/spyre_attn.py`
  (`_overwrite`), `torch_spyre/_inductor/customops.py` (`overwrite`).
