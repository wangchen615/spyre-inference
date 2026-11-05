# Spyre KV M4-Marvell: Two Replicas on a Marvell Pool, Shared vs Private

This runbook starts two independent TP1 vLLM replicas on two Spyre cards and
offloads their KV cache to a Marvell (Hillock) card, in two layouts:

```text
shared:   card A -> replica A --\
                                  +--> one Marvell pool (marvell-r0)
          card B -> replica B --/

private:  card A -> replica A -----> Marvell pool A (marvell-0, BAR2 window 1)
          card B -> replica B -----> Marvell pool B (marvell-0, BAR2 window 2)
```

Both layouts use the same cards, model, pool size per replica and request
flow, so the difference between them is what sharing buys: in the shared
layout B reloads the prompt A computed; in the private layout B cannot see
A's KV and computes it again.

It is the Marvell counterpart of
[2026-10-30-spyre-shared-kv-m4-full-stack-runbook.md](2026-10-30-spyre-shared-kv-m4-full-stack-runbook.md),
which uses a host DRAM pool.

Do not directly unlink any object under `/dev/shm`, and never reset a Spyre or
Marvell card on your own: see [Hardware safety](#hardware-safety).

## What this run proves

Shared layout:

1. The delegator plans and creates one Marvell pool; Flex places it in the
   card's BAR2 and reports the placement.
2. Both replicas attach the same generation at the same time.
3. Destroy returns `POOL_BUSY` while either replica is attached.
4. A cold-computes and stores a 4K prompt; A reloads it; B, on the other
   card, reloads the same prompt from A's KV with no recompute.

Private layout:

1. The delegator creates two flat Marvell pools, one per replica; Flex gives
   each its own BAR2 window on the same card, without overlap.
2. Each replica attaches only its own pool.
3. A and B each cold-compute and reload the same 4K prompt in their own pool.

In both layouts all responses to the measured prompt are token-identical, and
every pool is destroyed by its exact generation.

## Stack

| Repo | Branch | Base |
|---|---|---|
| flex | `kvc-offload-m4-marvell` | `kvc-offload-m3` 3fca45e1 |
| torch-spyre | `kvc-offload-m4-marvell` | `kvc-offload-m3` 6acd8092 |
| spyre-inference | `kvc-offload-m4-marvell` | `kvc-offload-m3` 7921f92 |
| shared-pool-delegator | `kvc-offload-m4-marvell` | `kvc-offload-m4` f237519 |
| kvc-experiment | `kvc-offload-m4-marvell` | `master` ddc2528 |

Image: `icr.io/ai_sw_accel_exp/kvc-offload-exp/spyre-inference:m3-syncfix-b32280`
(`sha256:8d715b65`): vLLM 0.28.0, torch 2.13.0, Python 3.12. The m4-marvell
flex, torch-spyre and spyre-inference are layered over the image's own stack;
the image itself is never modified.

## Prerequisites

### Hardware

The verified host, worker-26:

| Device | PCI address | Role |
|---|---|---|
| Spyre | `0000:c8:00.0` | replica A |
| Spyre | `0000:ec:00.0` | replica B |
| Marvell CXL | `0000:ed:00.0` | KV pool memory (BAR2, 512 GiB window) |

Both Spyre cards must reach the Marvell card peer-to-peer. On worker-26 all
three sit under one Broadcom switch fabric (upstream port `c2:00.0`; `c8` on
downstream port `c3:04.0`; `ec` and `ed` on `c3:1c.0` -> `da:00.0` ->
`db:10.0` / `db:14.0`). Peer-to-peer fails with
`RAS::RUNTIMESCHEDULER::DmaHardwareError` if ACS redirect is on, so check
every downstream port on the paths first:

```bash
for p in c3:04.0 c3:1c.0 db:10.0 db:14.0; do
  sudo lspci -s "$p" -vvv | grep ACSCtl    # each must show ReqRedir- CmpltRedir-
done
```

Also check that no card's BAR0 reads back as zero:

```bash
for b in c8:00.0 ec:00.0 ed:00.0; do
  od -An -tx4 -j16 -N8 /sys/bus/pci/devices/0000:$b/config
done
```

On another host, map the cards and switch ports first
(`readlink -f /sys/bus/pci/devices/<bdf>`) and substitute your addresses in
every command below. The scripts take them from the environment: `A_BDF` and
`B_BDF` for the two Spyre cards, and `FLEX_TEST_PCI_BDF` for the Marvell card,
which `devbox.sh` sets from `MARVELL_BDF`.

### Checkouts and environment

All repos are checked out on `kvc-offload-m4-marvell` under one root:

```bash
export DT=/home/yzhu/dt-inductor
export W=$DT/m4-marvell                     # flex, torch-spyre, spyre-inference,
                                            # shared-pool-delegator, kvc-experiment
export H=$W/kvc-experiment/hillock/m4       # the harness: two_replica_*.sh
export S=$H/dev                             # devbox.sh and the in-container build helpers
export MODEL_DIR=$DT/models/micro-g3.3-8b-instruct-1b
```

The runs use a local copy of `ibm-ai-platform/micro-g3.3-8b-instruct-1b`
(4 layers, 8 KV heads, head size 128, fp16). Download it once:

```bash
podman run --rm --user 0 -v $DT:$DT -e HF_HOME=$DT/.cache/huggingface \
  icr.io/ai_sw_accel_exp/kvc-offload-exp/spyre-inference:m3-syncfix-b32280 \
  python3 -c "from huggingface_hub import snapshot_download as d; print(d('ibm-ai-platform/micro-g3.3-8b-instruct-1b'))"
ln -sfn "$(ls -d $DT/.cache/huggingface/hub/models--ibm-ai-platform--micro-g3.3-8b-instruct-1b/snapshots/*)" $MODEL_DIR
```

A local path is used because Flex limits a pool's model string to 128 bytes,
and offline vLLM substitutes a longer snapshot path for a Hugging Face id.

## 1. Start the dev container

One long-lived container runs the delegator and both replicas, so they share
one `/dev/shm` and one UID. Both Spyre cards and the Marvell card are visible,
but no card is locked by the container itself:

```bash
$S/devbox.sh start
```

On another host, pass your cards: `SPYRE_BDFS=<a>,<b> MARVELL_BDF=<marvell> $S/devbox.sh start`.

`devbox.sh card <bdf>[,<bdf>] -- <command>` runs a command inside it while
holding `$DT/.locks/spyre-<bdf>.lock` for each listed card, so other users of
`run_in_image.sh` on the host wait instead of opening the same card. Every
hardware step below goes through it.

Persistent pools live in the container's `/dev/shm`, so they last as long as
the container. `devbox.sh stop` removes the container and every pool with it.

## 2. Build the overlay

Inside the container, flex installs to `$STAGE/runtime` and torch-spyre and
spyre-inference install editable into `$STAGE/venv`, which layers over the
image's `/opt/.venv` (torch and vLLM stay the image's):

```bash
# flex -> $STAGE/runtime, and $STAGE/flex-root for the delegator (about 90 s cold)
podman exec kvc-m4-marvell bash $S/lib/build-flex-in-container.sh $W/flex

# torch-spyre _C.so against the staged flex, editable into $STAGE/venv (about 60 s)
podman exec kvc-m4-marvell bash $S/lib/build-torch-spyre-in-container.sh $W/torch-spyre

# spyre-inference, editable, --no-deps: vLLM 0.28.0 and torch stay the image's
podman exec kvc-m4-marvell bash $S/lib/install-spyre-inference-in-container.sh $W/spyre-inference

# the m4-marvell delegator -> $BUILD/m4marvell/delegator-flex (about 4 min cold: fetches gRPC)
podman exec kvc-m4-marvell bash -c \
  'source $STAGE/env.sh; export BUILD=$BUILD/m4marvell; bash $SCRIPTS/build-delegator-in-container.sh $DT/m4-marvell/shared-pool-delegator flex'
```

After a flex edit, `$S/flexbuild.sh $W/flex` rebuilds incrementally in a few
seconds; it needs a stage the first command above has already set up.
`$STAGE/runtime/FLEX_REVISION` records the flex commit, with `-dirty` if the
checkout has uncommitted changes.

Check that the overlay resolves to the m4-marvell checkouts and the staged
flex:

```bash
podman exec kvc-m4-marvell bash -c 'source $STAGE/env.sh
  python -c "import torch, torch_spyre, spyre_inference, vllm; print(torch_spyre.__file__, spyre_inference.__file__, vllm.__version__)"
  ldd $DT/m4-marvell/torch-spyre/torch_spyre/_C.so | grep libflex'
```

Expected: both modules under `$DT/m4-marvell/...`, vLLM `0.28.0`, and
`libflex.so => $DT/stage-env/m4marvell/runtime/lib/libflex.so`.

A Marvell pool needs the m4-marvell delegator. The unchanged `kvc-offload-m4`
delegator only creates host DRAM pools.

## 3. Verify Spyre-to-Marvell DMA from both cards

Before any vLLM run, move a few KV pages between each Spyre card and the
Marvell card and compare them byte for byte. This is Flex's own test, bounded
to a 64 MiB BAR2 window and 256 KiB pages; it stops on the first failure:

```bash
F='RuntimeStreamCopyRawPciUnittest.MarvellPageRangeRoundTripInPlace'
F+=':RuntimeStreamCopyRawPciUnittest.MarvellPageRangeRoundTripRelocated'
F+=':RuntimeStreamCopyRawPciUnittest.MarvellPageRangeBatchedScatter'
for card in ec c8; do
  $S/devbox.sh card 0000:$card:00.0 -- bash -c \
    "export FLEX_TEST_PCI_BDF=0000:ed:00.0 FLEX_TEST_BAR2_SIZE=67108864
     timeout 300 \$BUILD/flex/tests/flex_unit_test --gtest_filter='$F' --gtest_break_on_failure"
done
```

Each card must report `[  PASSED  ] 3 tests.` Stop if either one reports a
RAS error: see [Hardware safety](#hardware-safety).

Optionally measure the link from each card (one verified 64 MiB transfer
each way):

```bash
for card in ec c8; do
  $S/devbox.sh card 0000:$card:00.0 -- bash -c \
    "export FLEX_TEST_PCI_BDF=0000:ed:00.0 FLEX_TEST_BAR2_SIZE=67108864
     timeout 300 \$BUILD/flex/tests/flex_unit_test --gtest_filter='RuntimeStreamCopyRawPciUnittest.BandwidthSpyreMarvell64MiB'"
done
```

Verified on worker-26: both cards reach the Marvell card at the same rate.

| Card | Spyre -> Marvell | Marvell -> Spyre |
|---|---|---|
| `ec` | 23.12 GB/s | 42.21 GB/s |
| `c8` | 22.71 GB/s | 41.76 GB/s |

## 4. Run the shared layout

`$H/two_replica_runbook.sh` (kvc-experiment
`hillock/m4/two_replica_runbook.sh`) runs the whole shared sequence in one go, holding both card
locks:

```bash
$S/devbox.sh card 0000:c8:00.0,0000:ec:00.0 -- bash -c '
  export RUNBOOK_TIER=marvell
  export DELEGATOR_BUILD=$BUILD/m4marvell/delegator-flex
  export SPYRE_INFERENCE_SRC=$DT/m4-marvell/spyre-inference
  export LD_LIBRARY_PATH=$STAGE/flex-root/build:$LD_LIBRARY_PATH
  bash $DT/m4-marvell/kvc-experiment/hillock/m4/two_replica_runbook.sh runbook-marvell'
```

Results go to `$DT/logs/e2e/runbook-marvell/`; `summary.txt` has the outcome.

What the script does, in order:

1. Starts the delegator on a private socket.
2. Plans and creates one 512 MiB Marvell pool from the model:

   ```bash
   python scripts/shared_kv_two_instance_demo.py prepare-model-pool \
     --ctl $DELEGATOR_BUILD/shared-poolctl --socket $SOCK \
     --model $MODEL_DIR --tensor-parallel-size 1 --pool-name /kvc-m4-runbook-<tag> \
     --marvell-bdf 0000:ed:00.0 --marvell-bytes-to-use 536870912
   ```

   and fails unless the planned arena is `MARVELL marvell-r0` on
   `0000:ed:00.0`.
3. Starts replica A on `c8` (port 18100) and replica B on `ec` (port 18101).
   Each is TP1 at `LOCAL_RANK=0` with a one-card senlib config, so it opens
   only its own card:

   ```bash
   LOCAL_RANK=0 SPYRE_DEVICES=0 AIU_WORLD_SIZE=1 AIU_WORLD_RANK_0=<bdf> \
   PCIDEVICE_IBM_COM_AIU_PF=<bdf> SENLIB_DEVEL_CONFIG_FILE=/cfg/senlib_config-<bdf>.json \
   vllm serve $MODEL_DIR --host 127.0.0.1 --port <port> --no-enable-prefix-caching \
     --tensor-parallel-size 1 --distributed-executor-backend uni --max-model-len 4352 \
     --max-num-seqs 1 --max-num-batched-tokens 512 --kv-transfer-config "$KV_TRANSFER_CONFIG"
   ```

   Both get the same `kv_connector_extra_config`: the complete pool plan,
   with `spec_name: SpyreSharedOffloadingSpec`.
4. Destroys the exact generation while both are attached and expects
   `POOL_BUSY`.
5. Warms up with `scripts/warmup_shared_kv_demo.py`: two cold stores, then
   two crossed peer reloads.
6. Measures one fresh 4,096-token prompt: A cold, A self-reload, B peer
   reload, with `scripts/shared_kv_two_instance_demo.py --show-json`.
7. Stops both replicas and destroys the exact generation.

## 5. Run the private layout

`$H/two_replica_private.sh` (kvc-experiment
`hillock/m4/two_replica_private.sh`) runs the private sequence with the same cards, pool size and
prompt:

```bash
$S/devbox.sh card 0000:c8:00.0,0000:ec:00.0 -- bash -c '
  export PRIVATE_TIER=marvell
  export DELEGATOR_BUILD=$BUILD/m4marvell/delegator-flex
  export SPYRE_INFERENCE_SRC=$DT/m4-marvell/spyre-inference
  export LD_LIBRARY_PATH=$STAGE/flex-root/build:$LD_LIBRARY_PATH
  bash $DT/m4-marvell/kvc-experiment/hillock/m4/two_replica_private.sh private-marvell'
```

Results go to `$DT/logs/e2e/private-marvell/`.

What the script does, in order:

1. Starts the delegator.
2. Creates two flat 512 MiB Marvell pools, one per replica:

   ```bash
   python scripts/shared_kv_two_instance_demo.py prepare-pool \
     --ctl $DELEGATOR_BUILD/shared-poolctl --socket $SOCK \
     --model $MODEL_DIR --pool-name /kvc-m4-private-<tag>-A \
     --cpu-bytes-to-use 536870912 --marvell-bdf 0000:ed:00.0
   # and the same with -B
   ```

   `--marvell-bdf` makes `--cpu-bytes-to-use` the pool's size on the
   Marvell card. The request carries no BAR2 offset; Flex picks one (see
   [Where each private pool goes in BAR2](#where-each-private-pool-goes-in-bar2)).
   The script fails unless each pool is `MARVELL marvell-0` on
   `0000:ed:00.0`, and logs where Flex placed it.
3. Starts A on `c8` with pool A and B on `ec` with pool B, using the same
   `vllm serve` line as the shared layout but with each replica's own
   connector config and the local spec:

   ```json
   {"pool_name": "/kvc-m4-private-<tag>-A", "pool_generation": "<A's generation>",
    "data_pool_name": "marvell-0", "cpu_bytes_to_use": 536870912,
    "spec_name": "SpyreOffloadingSpec",
    "spec_module_path": "spyre_inference.v1.kv_offload.spec"}
   ```

   `prepare-pool` prints `spec_name: SpyreSharedOffloadingSpec` because it is
   the two-instance shared demo's command; the script replaces it with the
   local spec, which has no shared metadata on the KV path. Each worker logs
   `SpyreOffloadingSpec: 256 host block(s) ... shared child /kvc-m4-private-<tag>-A/marvell-0`
   (`host block` is only the spec's name for its offload blocks).
4. Destroys pool A while A is attached and expects `POOL_BUSY`.
5. Warms up each replica on its own prompt: one cold store, one reload.
   The shared warmup script cannot be used, because it requires crossed peer
   reloads.
6. Measures one fresh 4,096-token prompt: A cold, A self-reload, B cold,
   B self-reload.
7. Stops both replicas and destroys both pools.

### Where each private pool goes in BAR2

Neither pool's config names a BAR2 offset. Flex chooses one inside
`SharedPool::CreatePool` (flex `src/memory_interface/shared_pool_lifecycle.cpp`,
`PlaceMarvellArenas`):

1. It takes a per-card lock, `/dev/shm/flex-shared-pool-guards/marvell-<bdf>.lock`,
   so two creates cannot pick the same window.
2. It reads the card's BAR2 size from sysfs and scans every live pool root
   for windows already taken on that card.
3. It takes the lowest free offset that fits, records it in the new root, and
   returns it in the effective config.

The script creates A first, so A gets offset 0 and B gets the first offset
after A's 512 MiB, 536870912. The offset depends only on creation order and on
what is already live. Destroying a pool frees its window. An explicit
`"barOffset"` in the config is used as given, but `CreatePool` rejects it if
it overlaps a live pool.

## 6. Check the results

Each run's `summary.txt` must end in `runbook PASS` or `private PASS`.
A shared run looks like this:

```text
pool /kvc-m4-runbook-runbook-marvell generation=089d1183... arena: MARVELL marvell-r0 0000:ed:00.0 0
A on 0000:c8:00.0 :18100, B on 0000:ec:00.0 :18101, both healthy
destroy while attached: POOL_BUSY
A cold:        local_compute_store computed=4096 loaded=0 store=67108864 load=0 ttft=0.8523
A self-reload: external_kv_reload computed=0 loaded=4096 store=0 load=67108864 ttft=0.0921
B peer-reload: external_kv_reload computed=0 loaded=4096 store=0 load=67108864 ttft=0.1337
tokens: all three identical
destroy after both stopped: OK
runbook PASS
```

A private run:

```text
pool A /kvc-m4-private-private-marvell-A generation=d217d81e... arena: MARVELL marvell-0 0000:ed:00.0 0
pool B /kvc-m4-private-private-marvell-B generation=76047dd5... arena: MARVELL marvell-0 0000:ed:00.0 536870912
A pcie_address=0000:c8:00.0 multiproc_executor
B pcie_address=0000:ec:00.0 multiproc_executor
destroy pool A while A attached: POOL_BUSY
A cold:        local_compute_store computed=4096 loaded=0 store=67108864 load=0 ttft=1.1247
A self-reload: external_kv_reload computed=0 loaded=4096 store=0 load=67108864 ttft=0.0656
B cold:        local_compute_store computed=4096 loaded=0 store=67108864 load=0 ttft=1.1213
B self-reload: external_kv_reload computed=0 loaded=4096 store=0 load=67108864 ttft=0.1109
tokens: all four identical
destroy pool A: OK
destroy pool B: OK
private PASS
```

Every request reports where its prompt tokens came from, from vLLM's
`vllm:prompt_tokens_by_source_total`: a store is all `computed`, a reload all
`loaded`. The demo client rejects a request that mixes the two.

Check that the pools are gone afterwards:

```bash
podman exec kvc-m4-marvell bash -c 'ls /dev/shm | grep -v flex-shared-pool-guards'
```

## Verified result from 2026-11-05

Worker-26, 3 runs of each layout, alternating shared and private. Time to
first token for the 4,096-token prompt, median (range):

| Request | Shared pool | Private pools |
|---|---|---|
| A cold | 0.828 s (0.802-0.852) | 1.135 s (1.125-1.154) |
| A reload | 0.092 s (0.087-0.096) | 0.072 s (0.066-0.103) |
| **B on A's prompt** | **0.127 s (0.118-0.134)**, peer reload | **1.081 s (1.063-1.121)**, cold compute |
| B reload of its own copy | - | 0.092 s (0.086-0.111) |

- With a shared pool, the second replica gets a prompt the first one already
  computed 8.5x faster, and the card holds one copy of its KV instead of two.
- Reloading from the Marvell card costs the same in both layouts: the median
  of all six private reloads is 0.089 s, against 0.092 s shared.
- The cold requests differ by about 0.3 s because of the executor, not the
  pool. The shared spec needs `uni` at TP1, because its scheduler-side
  attachment opens the card. On this vLLM 0.28 base, `CpuPlatform` turns
  `uni` into `mp`, and the platform keeps `uni` only for the shared spec, so
  the private replicas run on `mp`. Compare reload with cold within one
  layout, not cold across layouts.

Every run: token-identical outputs, `POOL_BUSY` while attached, all pools
destroyed, `/dev/shm` clean. In every private run Flex placed the pools at
BAR2 offsets 0 and 536870912.

## Hardware safety

- Check ACS and each card's BAR0 before any Marvell run (see
  [Prerequisites](#prerequisites)).
- Run nothing else on either card while a measured run is in progress: on
  worker-26 both Spyre cards share one PCIe upstream path, so a run on one
  card skews the other's timings.
- A Marvell pool's bytes are card memory: anything that writes the card
  outside the pool lifecycle (the m3 image, raw DMA tests) does not see the
  pools' BAR2 windows. Destroy every m4-marvell pool on the card before such a
  run.
- On a hang, `RAS::PCI::BusFence`, `DmaHardwareError` or `PCIeFailure`: stop,
  kill only your own process, and leave the cards and `/dev/shm` as they are.
  Do not retry a card that reported `PCIeFailure`; do not reset a card
  without authorization.

## Troubleshooting

### A run fails with `no pool`

Read `pool-*.err` or `pool.err` in the run directory. If `pool.json` starts
with vLLM `INFO` lines, `VLLM_LOGGING_STREAM=ext://sys.stderr` was not set:
vLLM logs to stdout by default and the JSON cannot be parsed. Both scripts
set it.

### A replica does not start: `RAS::VFIO::DeviceOpenFail`

A Spyre card opens in one process only. Either the other replica, or another
user's run, already holds the card; check `$DT/.locks/` and the container's
processes. For the shared spec at TP1 this is also what happens if it runs on
the `mp` executor: its scheduler-side attachment opens the card a second time.

### `Address already in use`

Two `vllm serve` processes in the one container asked for the same port. The
scripts use 18100 and 18101; do not run another server on those ports at the
same time.

### `ResourceError` about a BAR2 window

`Marvell BAR2 window overlaps pool '<name>'` or `does not fit in BAR2` means
a live pool on the card already owns that range, or the request is larger
than the free space. List the live pools (`ls /dev/shm`) and destroy the ones
you no longer need through the delegator, by exact generation.

### A pool is left behind

Destroy it through the delegator with its exact generation; never unlink
`/dev/shm` objects directly:

```bash
podman exec kvc-m4-marvell bash -c 'source $STAGE/env.sh
  bash $DT/m4-marvell/kvc-experiment/hillock/m4/dev/destroy-pools.sh "kvc-m4-*"'
```

## Captured artifacts

Each run directory `$DT/logs/e2e/<tag>/` holds:

| File | Contents |
|---|---|
| `summary.txt` | the outcome, as above |
| `pool.json` / `pool-A.json`, `pool-B.json` | the delegator's create result, including the pool plan |
| `serve-A.log`, `serve-B.log` | each replica's vLLM log: its card, executor, spec and pool |
| `warmup*.log`, `a-*.log`, `b-*.log` | every request's prompt source, KV bytes, TTFT and output tokens |
| `delegator.log` | every create, open and destroy |
