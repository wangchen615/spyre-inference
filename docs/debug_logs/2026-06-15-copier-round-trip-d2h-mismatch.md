# Copier round-trip: device→host mismatch on `arange` fp16 pattern

**Date:** 2026-06-15
**Test:** `tests/v1/kv_offload/test_copier_round_trip.py::test_copier_round_trip_spyre`
**Status:** ❌ Failing on real Spyre hardware
**Raw log:** `2026-06-15-copier-round-trip-d2h-mismatch-pytest-output.txt`

## What we tried

Ran the round-trip test for `SpyreKvDmaCopier` on real Spyre hardware:

```bash
uv run pytest -m "not upstream" tests/v1/kv_offload/test_copier_round_trip.py
```

The test exercises a device→host→mutate→host→device→host round trip:

1. Build a known fp16 pattern `torch.arange(2*16*64).reshape(2, 16, 64)` and move it to the `spyre` device.
2. `copier.copy_d2h(src_spyre, host)` — copy device → host.
3. **Assert `torch.equal(host, pattern)`** ← fails here.
4. (never reached) mutate host `+2.0`, `copy_h2d` into a fresh device tensor, `copy_d2h` back, assert equals `pattern + 2.0`.

The device *was* available — the run got past the `_spyre_available()` skip gate and actually executed on hardware (vLLM/Qwen3 platform init is visible in the captured setup output). So this is a genuine hardware result, not a CPU-only skip.

## What we saw

The first assertion (`torch.equal(host, pattern)`) failed. It is **not** a wholesale garbage mismatch — the two tensors agree on the low-magnitude elements and diverge only on the high-magnitude tail. Comparing the reprs in the failure:

| element region | host (after `copy_d2h`) | expected `pattern` |
|---|---|---|
| start of tensor | `0, 1, 2, … 61, 62, 63` | `0, 1, 2, … 61, 62, 63` (match) |
| tail row | `1984.0, 1984.0, 1986.0, … 2044.0, 2046.0, 2046.0` | `1984.0, 1985.0, 1986.0, … 2045.0, 2046.0, 2047.0` |

The differences are all **±1 ULP at fp16 magnitudes ≥ ~1024**, where fp16 step size is 1.0 (and 2.0 above 2048). i.e. values like `1985 → 1984`, `2045 → 2044`, `2047 → 2046`. Small values round-trip exactly; large values are off by one representable step.

`torch.equal` requires bitwise-exact equality, so any single ULP difference fails it.

## Interpretation / leads

The pattern (exact at small magnitudes, ±1 ULP at large magnitudes) is the signature of an **fp16 rounding / accumulation difference introduced somewhere in the copy path**, not random corruption or a layout/shape bug:

- A pure DMA byte copy would be bit-exact — values would either match perfectly or be wildly wrong. ±1 ULP localized to large values rules out a plain `memcpy`-style transfer.
- More likely the `copy_d2h` path performs a **dtype round-trip or a non-trivial cast/compute** (e.g. via an fp32 intermediate, or an op that re-rounds) so the largest fp16 values land on a neighboring representable value.
- Worth confirming whether `SpyreKvDmaCopier.copy_d2h` uses `torch_spyre._C.copy_tensor` (per recent commit `ef2c3a7`) as a true device copy, or whether the `.to(device)` of the source pattern and/or the readback introduces a cast. The `pattern.to(device)` upload itself could already be the lossy step (large arange → fp16 on device), since the comparison is against the *host* `pattern` which never went to the device.

## Suggested next steps

1. **Check fallbacks first** — re-run with `-W "error::torch_spyre.ops.fallbacks.FallbackWarning"` to see if `copy_d2h`/the upload silently routes through a CPU fallback that re-rounds.
2. **Decide whether bitwise equality is the right assertion.** If the copier is allowed to round-trip through fp32 or re-round, switch `torch.equal` → `torch.testing.assert_close` with a 1-ULP-appropriate tolerance. If the copier is contractually a byte-exact DMA, then ±1 ULP is a real bug in the transfer path.
3. **Isolate the lossy hop** — compare `src_spyre` read straight back (`copy_d2h` of the freshly uploaded tensor) against `pattern` to determine whether the loss is on upload (`pattern.to(device)`) or on the device→host copy itself.
