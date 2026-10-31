# Spyre Shared KV M4 Planned-Pool Full-Stack Runbook

This runbook creates one model-specific persistent pool through the
`shared-pool-delegator`, starts two independent TP1 vLLM replicas on two Spyre
cards, and proves that both replicas share the same host KV bytes.

The two-card layout is intentionally:

```text
card 0 -> replica A, TP1
card 1 -> replica B, TP1
```

It is not one TP2 replica. Using TP1 leaves one card for each simultaneously
attached replica, so the test can demonstrate a cold store by A, a self-reload
by A, and a peer-reload by B from one persistent generation.

Do not directly unlink any object under `/dev/shm`.

## What this run proves

1. Spyre-inference resolves the real model and physical KV-cache geometry.
2. The delegator plans and creates the metadata object plus one exact-sized,
   partitioned host arena before either worker starts.
3. Both vLLM replicas attach to the same immutable pool plan and generation;
   each TP1 worker's single attach call returns metadata plus all rank-0 data
   handles with their IOMMU mappings ready.
4. A cold-computes and publishes one complete prompt.
5. A reloads every prompt token from shared host KV.
6. B reloads the same prompt from the same pool while A is still attached.
7. Destroy returns `POOL_BUSY` while either replica remains attached.
8. Graceful detach followed by exact-generation destroy succeeds.

Prefix caching is disabled. A reload therefore cannot be satisfied by vLLM's
device-local prefix cache.

## Prerequisites

The following branches must already be built:

- Flex `kvc-offload-m4`;
- torch-spyre `kvc-offload-m4`;
- shared-pool-delegator `kvc-offload-m4`;
- spyre-inference `kvc-offload-m4`; and
- the spyre-comms revision compatible with the Flex M4 launch ABI.

The delegator and workers must run under the same numeric host UID and share
the same `/dev/shm` namespace. Both Spyre cards must be idle.

Set the verified development paths:

```bash
export DTI_PROJECT_ROOT=/home/yzhu/dt-inductor
export FLEX_M4="$DTI_PROJECT_ROOT/flex-worktrees/persistent-shared-kv-pool-lifecycle"
export TORCH_SPYRE_M4="$DTI_PROJECT_ROOT/torch-spyre/.worktrees/kvc-offload-m4"
export SPYRE_INFERENCE_M4="$DTI_PROJECT_ROOT/spyre-inference-kvc-offload-m4"
export DELEGATOR_M4="$DTI_PROJECT_ROOT/shared-pool-delegator/.worktrees/kvc-offload-m4"
export SPYRE_COMMS_M4="$DTI_PROJECT_ROOT/sentient/spyre_comms-m4-compat"
export DELEGATOR_SOCKET="/tmp/shared-pool-delegator-${UID}-m4-demo.sock"

export PYTHONPATH="$SPYRE_INFERENCE_M4:$TORCH_SPYRE_M4:${PYTHONPATH-}"
export LD_LIBRARY_PATH="$FLEX_M4/build:$SPYRE_COMMS_M4/lib:$DTI_PROJECT_ROOT/sentient/libaiupti/lib:$DTI_PROJECT_ROOT/sentient/runtime/lib:$DTI_PROJECT_ROOT/sentient/deeptools/lib:${LD_LIBRARY_PATH-}"

export MODEL=ibm-ai-platform/micro-g3.3-8b-instruct-1b
export MODEL_REVISION=main
export SHARED_POOL_NAME=/micro-g3-m4-planned-demo
export CPU_BYTES_TO_USE=536870912

cd "$SPYRE_INFERENCE_M4"
```

Keep the compatible spyre-comms directory ahead of any older installation in
`LD_LIBRARY_PATH`. Keep the DeepTools directory because `dbo-opt` needs
`libdcc.so`.

Confirm the required artifacts exist:

```bash
test -x "$DELEGATOR_M4/build/shared-pool-delegator"
test -x "$DELEGATOR_M4/build/shared-poolctl"
test -f "$TORCH_SPYRE_M4/torch_spyre/_C.so"
test -d "$FLEX_M4/build"
```

Check that no other process owns either card. The VFIO numbers are host
specific; use the two entries present on the target machine:

```bash
ls -l /dev/vfio
fuser -v /dev/vfio/95 /dev/vfio/161 || true
```

Do not continue if `fuser` reports another inference or test process.

## 1. Start and health-check the delegator

Run the delegator in the control terminal:

```bash
"$DELEGATOR_M4/build/shared-pool-delegator" \
  --socket-path "$DELEGATOR_SOCKET" \
  --socket-mode 0600 \
  >/tmp/spyre-shared-kv-m4-delegator.log 2>&1 &
export DELEGATOR_PID=$!

until "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" health >/dev/null 2>&1; do
  kill -0 "$DELEGATOR_PID"
  sleep 0.1
done
```

The `health` call proves that the Unix-socket service is responding. The
`kill -0` call sends no signal; it only fails if the delegator exited while the
loop was waiting.

The delegator is the lifecycle controller, not the lifetime owner. A READY
pool remains in shared memory if the delegator later restarts.

## 2. Resolve the model and create the whole pool

Use `prepare-model-pool`, not the legacy flat `prepare-pool` command:

```bash
POOL_JSON=$(uv run --active --no-sync python \
  scripts/shared_kv_two_instance_demo.py prepare-model-pool \
  --ctl "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" \
  --model "$MODEL" \
  --revision "$MODEL_REVISION" \
  --tensor-parallel-size 1 \
  --pool-name "$SHARED_POOL_NAME" \
  --cpu-bytes-to-use "$CPU_BYTES_TO_USE")

printf '%s\n' "$POOL_JSON" | jq .
```

The helper first resolves the model's KV dtype, block size, physical cache
allocations, layout, local KV heads, and head size. It then calls the
delegator's `create-model` operation with that versioned specification. The
delegator plans one exact physical arena, checks aggregate `/dev/shm` capacity,
calls Flex to create the root, metadata, and data objects, and returns the
complete immutable plan consumed by both replicas. This resolves model
configuration but does not load model weights.

Export the identity and build vLLM's connector argument directly from the
returned plan:

```bash
export SHARED_POOL_GENERATION=$(jq -r .poolGeneration <<<"$POOL_JSON")

export KV_TRANSFER_CONFIG=$(jq -c '{
  kv_connector: "SpyreOffloadingConnector",
  kv_role: "kv_both",
  kv_connector_module_path: "spyre_inference.v1.kv_offload.connector",
  kv_connector_extra_config: .instanceAExtraConfig
}' <<<"$POOL_JSON")
```

Inspect the effective layout:

```bash
jq '{
  result: .instanceAExtraConfig.pool_plan.code,
  generation: .poolGeneration,
  state: .instanceAExtraConfig.pool_plan.poolDescriptor.state,
  allocatedBytes: .instanceAExtraConfig.pool_plan.plan.allocatedBytes,
  bytesPerCompleteBlock: .instanceAExtraConfig.pool_plan.plan.bytesPerCompleteBlock,
  logicalBlockCapacity: .instanceAExtraConfig.pool_plan.plan.logicalBlockCapacity,
  partitions: .instanceAExtraConfig.pool_plan.plan.partitions,
  componentCount: (.instanceAExtraConfig.pool_plan.plan.components | length)
}' <<<"$POOL_JSON"
```

Inspect the resolved specification that the delegator planned:

```bash
jq '.instanceAExtraConfig.pool_plan.plan.normalizedSpec | {
  planningVersion,
  name,
  model,
  modelRevision,
  kvDtype,
  blockSize,
  tensorParallelSize,
  cpuBytesToUse,
  rankCaches
}' <<<"$POOL_JSON"
```

For the verified model, the resolver produced four TP1 physical-cache rows.
Every row used a 262,144-byte head-major page with eight local KV heads and a
128-element head size.

The verified 512 MiB plan contained:

- one `host-r0` physical arena;
- four logical partitions, `host-r0-c0` through `host-r0-c3`;
- 512 slots of 262,144 bytes in each partition;
- eight global components: one K and one V component per partition;
- 256 complete logical blocks;
- 2,097,152 bytes per complete block; and
- zero unused budget bytes.

## 3. Start two simultaneous TP1 replicas

Both replicas receive the exact same `KV_TRANSFER_CONFIG`, including the full
pool plan and generation. They differ only in card and HTTP port. Each process
has TP rank 0 because these are two independent TP1 replicas, not one TP2
replica.

```bash
export PYTHONHASHSEED=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_PLUGINS=spyre_inference

COMMON_ARGS=(
  "$MODEL"
  --revision "$MODEL_REVISION"
  --host 127.0.0.1
  --no-enable-prefix-caching
  --tensor-parallel-size 1
  --distributed-executor-backend uni
  --max-model-len 4352
  --max-num-seqs 1
  --max-num-batched-tokens 512
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
)

LOCAL_RANK=0 SPYRE_DEVICES=0 \
  uv run --active --no-sync vllm serve "${COMMON_ARGS[@]}" \
  --port 18100 >/tmp/spyre-shared-kv-m4-instance-a.log 2>&1 &
export INSTANCE_A_PID=$!

LOCAL_RANK=0 SPYRE_DEVICES=1 \
  uv run --active --no-sync vllm serve "${COMMON_ARGS[@]}" \
  --port 18101 >/tmp/spyre-shared-kv-m4-instance-b.log 2>&1 &
export INSTANCE_B_PID=$!
```

Wait for both servers. Compilation can take several minutes:

```bash
until curl -fsS http://127.0.0.1:18100/health >/dev/null; do
  kill -0 "$INSTANCE_A_PID"
  sleep 1
done

until curl -fsS http://127.0.0.1:18101/health >/dev/null; do
  kill -0 "$INSTANCE_B_PID"
  sleep 1
done
```

Useful log checks:

```bash
tail -f /tmp/spyre-shared-kv-m4-instance-a.log
tail -f /tmp/spyre-shared-kv-m4-instance-b.log
```

Each log should report the same pool name, four physical caches, eight
components, 256 logical blocks, and 536,870,912 actual host bytes.

Worker startup now performs one eager, rank-scoped attachment. For this TP1
run, spyre-inference calls `attach_pool(name, generation, expected_config, 0)`.
A successful result contains one aggregate `PoolAttachment`: its metadata
handle sees the global directory, while its `data_pools` contain all four
`host-r0-c*` partitions ready for DMA. There is no later `resolve_pool` call.

For a TP-N deployment, every worker supplies its local TP rank to the same
operation and receives only that rank's data partitions under the common root.
An absent child, invalid rank, data mapping failure, or IOMMU mapping failure is
an attachment failure and must prevent the server from reaching `/health`.

## 4. Prove live attachments block destroy

Destroying the exact generation must return `POOL_BUSY` while A and B retain
their aggregate attachments or any independently retained metadata, data-pool,
or read-pin handle:

```bash
if "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" destroy \
  --name "$SHARED_POOL_NAME" \
  --generation "$SHARED_POOL_GENERATION"; then
  echo "ERROR: destroy unexpectedly succeeded while workers were attached"
  exit 1
else
  echo "Expected POOL_BUSY while workers are attached"
fi
```

## 5. Warm both replicas, then measure a fresh 4K prompt

First give A and B different 4,096-token prompts, then cross them: B reloads
A's prompt and A reloads B's prompt. This removes each replica's first-request
overhead and exercises peer reloads in both directions before measurement.

```bash
export WARMUP_ID=spyre-shared-kv-planned-warmup-$(date +%s)

uv run --active --no-sync python -u \
  scripts/warmup_shared_kv_demo.py \
  --instance-a-host 127.0.0.1 --instance-a-port 18100 \
  --instance-b-host 127.0.0.1 --instance-b-port 18101 \
  --model "$MODEL" --warmup-id "$WARMUP_ID" \
  --prompt-tokens 4096 --output-tokens 16 \
  --show-json 2>&1 | tee /tmp/spyre-shared-kv-m4-warmup.log
```

The warmup must report two 4,096-token local-compute stores followed by two
4,096-token external-transfer reloads. It compares each crossed reload's token
IDs and text with its cold request and records the result in
`output_comparisons`. A difference is reported as a warning rather than a
failure: long cold-prefill and reload attention paths can make different
floating-point choices during autoregressive generation. The prompt-source
checks remain strict.

Now use a fresh identifier that did not appear in the warmup. Only these next
three requests are performance measurements:

```bash
export DEMO_ID=spyre-shared-kv-planned-measured-$(date +%s)

uv run --active --no-sync python -u \
  scripts/shared_kv_two_instance_demo.py \
  --instance A --host 127.0.0.1 --port 18100 \
  --identifier "$DEMO_ID" --prompt-tokens 4096 --output-tokens 16 \
  --show-json 2>&1 | tee /tmp/spyre-shared-kv-m4-a-cold.log

uv run --active --no-sync python -u \
  scripts/shared_kv_two_instance_demo.py \
  --instance A --host 127.0.0.1 --port 18100 \
  --identifier "$DEMO_ID" --prompt-tokens 4096 --output-tokens 16 \
  --show-json 2>&1 | tee /tmp/spyre-shared-kv-m4-a-reload.log

uv run --active --no-sync python -u \
  scripts/shared_kv_two_instance_demo.py \
  --instance B --host 127.0.0.1 --port 18101 \
  --identifier "$DEMO_ID" --prompt-tokens 4096 --output-tokens 16 \
  --show-json 2>&1 | tee /tmp/spyre-shared-kv-m4-b-reload.log
```

Expected prompt sources:

| Request | Prompt source | KV activity |
| --- | --- | --- |
| A cold | 4,096 local-compute tokens | store 67,108,864 bytes |
| A self-reload | 4,096 external-transfer tokens | load 67,108,864 bytes |
| B peer-reload | 4,096 external-transfer tokens | load 67,108,864 bytes |

Confirm all three measured responses generated identical tokens:

```bash
grep '^Token IDs' \
  /tmp/spyre-shared-kv-m4-a-cold.log \
  /tmp/spyre-shared-kv-m4-a-reload.log \
  /tmp/spyre-shared-kv-m4-b-reload.log
```

## Eager attachment API verification from 2026-10-31

The revised rank-scoped API and planner geometry were verified independently
of the historical serving measurements below. The following current-worktree
checks passed with the prerequisite `LD_LIBRARY_PATH` from this runbook:

| Repository | Verification | Result |
| --- | --- | ---: |
| torch-spyre | `pytest -q tests/test_shared_pool_lifecycle.py tests/test_shared_host_pool.py tests/test_shared_metadata.py` | 39 passed, 1 skipped |
| shared-pool-delegator | `cmake --build build -j8` followed by `ctest --test-dir build --output-on-failure` | build passed; 11 passed, 2 disabled |
| shared-pool-delegator | `DELEGATOR_BUILD_DIR=$PWD/build pytest -q tests` | 9 passed |
| spyre-inference | `pytest -q tests/kv_offload` | 337 passed, 49 skipped |

These checks cover eager aggregate attachment, independent child lifetimes,
rank-homogeneous planner output, real Flex lifecycle publication, the
independent torch-spyre worker, plan parsing and rank routing, and the full
KV-offload unit/integration suite. They do not replace the two-replica serving
rerun required to refresh the performance evidence.

## Historical verified result from 2026-10-30

The performance evidence below predates the 2026-10-31 eager rank-scoped
attachment API revision. The serving and transfer flow is unchanged, but these
numbers do not by themselves verify the revised startup contract. Re-run steps
1 through 6 with the current Flex, torch-spyre, delegator, and spyre-inference
worktrees before recording new acceptance evidence.

The commands above were run against the final M4 worktrees:

- Flex `6261ef4b5`;
- torch-spyre `a1c08b470`;
- shared-pool-delegator `ae5c1cc1b`; and
- spyre-inference `de9282fb0`.

Both real Spyre PF cards were used simultaneously. The pool was created from
the real micro-g3 model specification, not from a hand-written flat geometry.

| Request | TTFT | E2E | KV bytes | KV copy time | Effective bandwidth |
| --- | ---: | ---: | ---: | ---: | ---: |
| A cold/store | 0.4920 s | 0.9161 s | 67,108,864 stored | 0.012181 s | 5.509 GB/s |
| A self-reload | 0.0507 s | 0.4575 s | 67,108,864 loaded | 0.009780 s | 6.862 GB/s |
| B peer-reload | 0.0490 s | 0.4196 s | 67,108,864 loaded | 0.010106 s | 6.640 GB/s |

Before measurement, the four-request warmup cold-stored one 4,096-token prompt
on each replica and then externally reloaded A's prompt on B and B's prompt on
A. Both peer reloads transferred 67,108,864 bytes. The disposable warmup
prompts showed the documented cold/reload numerical divergence, which the
script reported in `output_comparisons`.

The three measured requests used a fresh identifier and produced the same 16
token IDs and decoded text. Destroy returned `POOL_BUSY` while both servers
were attached. After both replicas detached, exact-generation destroy returned
`OK`, and a subsequent open returned `NOT_FOUND`.

These values are one functional demonstration, not a benchmark. KV copy time
measures the raw Flex transfer only; TTFT and E2E include additional serving
work. Compilation time is excluded because both servers reached `/health`
before the measured sequence started.

## 6. Stop both replicas and destroy the exact generation

Stop B and A gracefully so their managers release read pins and local data
handles before closing each aggregate attachment. Destroy remains busy until
all independently retained child handles have also drained:

```bash
kill -TERM "$INSTANCE_B_PID"
wait "$INSTANCE_B_PID"

kill -TERM "$INSTANCE_A_PID"
wait "$INSTANCE_A_PID"
```

Destroy exactly the generation created in step 2:

```bash
uv run --active --no-sync python scripts/cleanup_shared_kv_demo.py \
  --ctl "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" \
  --pool-name "$SHARED_POOL_NAME" \
  --pool-generation "$SHARED_POOL_GENERATION"
```

Expected output:

```text
destroyed /micro-g3-m4-planned-demo generation <generation>: {"detail": "", "result": "OK"}
```

Finally stop the delegator:

```bash
kill -TERM "$DELEGATOR_PID"
wait "$DELEGATOR_PID"
```

## Recover and destroy a pool when its identity is unknown

Destroy intentionally requires both the pool name and its exact generation.
The current management API has no `ListPools` operation, so a caller that has
lost both values cannot discover pools through `shared-poolctl` alone. Do not
guess a generation or manually unlink `/dev/shm` objects.

If the original delegator is no longer running, start a temporary controller.
The controller does not own the pool, and its per-name management guard safely
serializes this recovery with other management operations:

```bash
export RECOVERY_SOCKET="/tmp/shared-pool-delegator-${UID}-recovery-$$.sock"

"$DELEGATOR_M4/build/shared-pool-delegator" \
  --socket-path "$RECOVERY_SOCKET" \
  --socket-mode 0600 \
  >"/tmp/shared-pool-delegator-${UID}-recovery.log" 2>&1 &
export RECOVERY_DELEGATOR_PID=$!

trap 'kill -TERM "$RECOVERY_DELEGATOR_PID" 2>/dev/null || true' EXIT

until "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$RECOVERY_SOCKET" health >/dev/null 2>&1; do
  kill -0 "$RECOVERY_DELEGATOR_PID"
  sleep 0.1
done
```

When even the pool name is unknown, use the metadata-child filenames only to
produce candidate root names. This is an emergency node-local discovery step,
not a replacement for `OpenPool`. It must run under the same host UID and IPC
namespace as the process that created the pool:

```bash
mapfile -t POOL_CANDIDATES < <(
  find /dev/shm -maxdepth 1 -type f -uid "$(id -u)" -printf '%f\n' |
    sed -n 's/\.metadata\.[[:xdigit:]]\{32\}$//p' |
    sort -u |
    while read -r root; do
      test -f "/dev/shm/$root" && printf '/%s\n' "$root"
    done
)

printf 'candidate: %s\n' "${POOL_CANDIDATES[@]}"
```

Treat every filename as untrusted. Ask Flex to validate the root header and
print only real pool descriptors:

```bash
for candidate in "${POOL_CANDIDATES[@]}"; do
  if OPEN_JSON=$("$DELEGATOR_M4/build/shared-poolctl" \
    --socket "$RECOVERY_SOCKET" open --name "$candidate" 2>/dev/null); then
    jq -c 'select(.result == "OK") | {
      name: .descriptor.name,
      generation: .descriptor.generation,
      state: .descriptor.state,
      effectiveConfig: .descriptor.effectiveConfig
    }' <<<"$OPEN_JSON"
  fi
done
```

Select the intended pool from those validated descriptors. Re-open that name
and extract the generation from the descriptor, never from a child filename:

```bash
export RECOVERED_POOL_NAME=/the-validated-pool-name

OPEN_JSON=$("$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$RECOVERY_SOCKET" open --name "$RECOVERED_POOL_NAME")

jq -e '.result == "OK" and .descriptor.state == "READY"' \
  <<<"$OPEN_JSON" >/dev/null

export RECOVERED_POOL_GENERATION=$(jq -er \
  '.descriptor.generation | select(test("^[0-9a-fA-F]{32}$"))' \
  <<<"$OPEN_JSON")

printf 'destroy target: %s generation %s\n' \
  "$RECOVERED_POOL_NAME" "$RECOVERED_POOL_GENERATION"
```

Confirm that the target deployment's workers have exited. If ownership of a
remaining worker is uncertain, do not destroy the pool. The root flock remains
the authoritative check and destroy returns `POOL_BUSY` while any attachment
is alive:

```bash
pgrep -af 'vllm|EngineCore' || true

uv run --active --no-sync python scripts/cleanup_shared_kv_demo.py \
  --ctl "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$RECOVERY_SOCKET" \
  --pool-name "$RECOVERED_POOL_NAME" \
  --pool-generation "$RECOVERED_POOL_GENERATION"
```

Verify exact-generation cleanup, then stop the temporary controller:

```bash
VERIFY_JSON=$("$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$RECOVERY_SOCKET" open \
  --name "$RECOVERED_POOL_NAME" 2>/dev/null || true)

jq -e '.result == "NOT_FOUND"' <<<"$VERIFY_JSON" >/dev/null

kill -TERM "$RECOVERY_DELEGATOR_PID"
wait "$RECOVERY_DELEGATOR_PID"
trap - EXIT
```

If several valid descriptors are reported, compare their effective model,
tensor-parallel size, layout, and capacity before selecting one. Never destroy
all candidates as a cleanup shortcut.

## Troubleshooting

### Reload mixes local and external prompt tokens

The prompt length was not a whole number of KV blocks. For example, a
4,160-token prompt with a 128-token block size reloads 32 complete blocks and
recomputes the remaining 64 tokens. Use exactly 4,096 tokens for this run.

### `POOL_BUSY` during final cleanup

At least one metadata attachment, data view, or read pin is still alive.
Confirm both vLLM process groups exited, then retry the same exact generation.
Do not manually unlink `/dev/shm` objects.

### `NOT_FOUND`

The pool does not exist in the worker's IPC namespace. Check the pool name,
host UID, delegator host, and `/dev/shm` namespace. Workers never create a
missing replacement.

### `STALE_GENERATION`

The pool name was destroyed and reused. Obtain the replacement plan from the
delegator. Never apply the old generation to the replacement pool.

### Configuration mismatch

The model, revision, TP size, KV dtype, block size, or physical cache layout
does not match the existing READY generation. Use a different deployment pool
name or explicitly destroy the unused old generation before replanning.

### `libspyre_comms.so` reports an undefined Flex symbol

An older spyre-comms library was selected. Put
`$SPYRE_COMMS_M4/lib` immediately after the M4 Flex build in
`LD_LIBRARY_PATH`, as shown in the prerequisites.

### `dbo-opt` cannot load `libdcc.so`

The DeepTools library directory was omitted from `LD_LIBRARY_PATH`. Preserve
`$DTI_PROJECT_ROOT/sentient/deeptools/lib`.

## Captured artifacts

The procedure writes:

- `/tmp/spyre-shared-kv-m4-delegator.log`;
- `/tmp/spyre-shared-kv-m4-instance-a.log`;
- `/tmp/spyre-shared-kv-m4-instance-b.log`;
- `/tmp/spyre-shared-kv-m4-warmup.log`;
- `/tmp/spyre-shared-kv-m4-a-cold.log`;
- `/tmp/spyre-shared-kv-m4-a-reload.log`; and
- `/tmp/spyre-shared-kv-m4-b-reload.log`.

Keep these logs with reported results so pool geometry, model revision,
generation, prompt size, and native-stack failures remain auditable.
