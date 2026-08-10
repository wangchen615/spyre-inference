# KV-offload eviction/reload experiments

Measurement scripts written while validating the KV-offload PoC
(`dev/0807-kv-offload-yzhu`) on the `torch-aiu-runtime-dev` dev pod.

**This branch deliberately does not modify the offload implementation.** It branches
from the pristine PoC commit (`75fb119`) and only *adds* this directory, so the
diff against `dev/0807-kv-offload-yzhu` stays reviewable and the branch does not
deviate from upstream.

Full write-up, results, and build instructions:
`hillock-vmem@experiments/kvc-offload-evict/`.

## What each script is for

| script | purpose |
|---|---|
| `verify_correctness.py` | 24-token greedy comparison, offload on vs off — proves the reload path returns the *right* KV, not just fast KV |
| `run_length_sweep.sh` | offload on vs off across lengths (the headline speedup curve) |
| `traffic_three_way.py` | recompute / host-reload / HBM-reuse in **one** engine, paths selected by request order |
| `run_traffic_sweep.sh` | driver: fresh engine per length, N cycles each |
| `three_way.py` | earlier 3-engine variant; superseded (it had to vary the pool between cases) |
| `run_tight_hbm.sh` | tightest-legal device pool, roomy host pool |
| `run_matrix.sh` | prefix-caching × offload matrix |

## Portability

The `run_*.sh` drivers were written for one specific pod and hardcode:

- `NEW=/tmp/work-kvoffload` — the build tree
- `source ~/spyre-build/repro/scripts/env.sh` — sets `DTI_PROJECT_ROOT`,
  `SEN_PROJECT_SRC`, and **appends** to `LD_LIBRARY_PATH`
- `SENLIB_DEVEL_CONFIG_FILE=/etc/ibm/spyre/senlib_config.json`
- `OMP_NUM_THREADS=8` — the pod's cgroup CPU quota is 8 while the container sees
  192; without pinning, thread oversubscription costs ~13%

Adjust these paths before running elsewhere. The Python scripts have no such
assumptions and can be run directly with `uv run --no-sync python <script>`.

## Findings that affect anyone using `ttft_evict.py`

1. **The `--cpu-bytes` default (2.0 GB) is too small for L=65536.** Two 65536-token
   prompts need ~2.15 GB at 2.10 MB/block, so the host tier thrashes: only 440 of
   512 blocks reload, `stored` climbs without plateau, and timings never converge
   (360 → 148 → 147 s, still descending). At 8 GB it plateaus at 18.579 s ±1.6%.
   The default silently yields a ~8× pessimistic result.

2. **Engine-construction failures are silently swallowed.** `TransferCounter` dups
   fd 1/2 to a temp file *before* the engine is built and restores them only in
   `close()`. If `LLM()` raises, the visible log ends after ~12 lines with no error
   and the traceback goes to `/tmp/tmp*.ttft_evict.log`. The scripts here restore
   descriptors in a `finally` block instead — worth porting into `ttft_evict.py`.

3. **Eviction geometry decides whether a "reload" is real.** Sizing the junk/other
   prompt equal to the measured prompt only displaces `pool - blk` blocks, so half
   the "reload" is actually a device hit. See `TRAFFIC_DESIGN.md` in the hillock
   repo for the full derivation.

## Build note (not committed here)

Yue's `pyproject.toml` pins torch-spyre to `88964c010ac8eb9586ba5deac938ae4624c2f5a1`,
which is correct — it carries the DeepTools header-relocation fix needed against
`ibm-deeptools 2.0.0-0.main.1+1861.3bb8422` in the pod base image. However
`uv.lock` resolves to an **older** revision predating that fix, so a plain `uv sync`
builds a torch-spyre that will not compile.

Workaround used (intentionally **not** committed, as it is environment-specific):
point the dependency at a local checkout already at the right commit.

```toml
torch-spyre = { path = "../torch-spyre", editable = true }
```

Consequences: `--frozen` is no longer possible and numpy resolves to 2.3.5, outside
the declared `<2.3` bound (no failures observed). The proper fix is to regenerate
`uv.lock` against the pinned revision.
