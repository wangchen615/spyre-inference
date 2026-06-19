# Copier round-trip: device→host mismatch on `arange` fp16 pattern

**Date:** 2026-06-15
**Test:** `tests/v1/kv_offload/test_copier_round_trip.py::test_copier_round_trip_spyre`
**Status:** ❌ Failing on real Spyre hardware — root cause identified (device-tensor layout mismatch); fix proposed (raw conversion-free copy)
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

> **Superseded.** The leads in this section were the initial guesses and turned out to be wrong. The confirmed cause is a device-tensor **layout** mismatch, not an fp16 cast — see [Root cause: why `.to()` works but `copy_tensor()` does not](#root-cause-why-to-works-but-copy_tensor-does-not) below. Struck-through text is retained for the record.

~~The pattern (exact at small magnitudes, ±1 ULP at large magnitudes) is the signature of an **fp16 rounding / accumulation difference introduced somewhere in the copy path**, not random corruption or a layout/shape bug:~~ The pattern (exact at small magnitudes, ±1 ULP at large magnitudes) *is* the signature of a layout-driven element misplacement whose effect only becomes visible where the fp16 grid is coarse — not an fp16 cast in the copy path.

- ~~A pure DMA byte copy would be bit-exact — values would either match perfectly or be wildly wrong. ±1 ULP localized to large values rules out a plain `memcpy`-style transfer.~~ This DMA is *not* a plain byte copy: it applies a tiled-layout conversion driven by the device tensor's stride map, so a layout mismatch can misplace individual elements without producing wholesale garbage.
- ~~More likely the `copy_d2h` path performs a **dtype round-trip or a non-trivial cast/compute** (e.g. via an fp32 intermediate, or an op that re-rounds) so the largest fp16 values land on a neighboring representable value.~~ There is no fp32 intermediate or re-rounding op; the values land on neighboring representable values because the wrong source elements are gathered.
- ~~Worth confirming whether `SpyreKvDmaCopier.copy_d2h` uses `torch_spyre._C.copy_tensor` (per recent commit `ef2c3a7`) as a true device copy, or whether the `.to(device)` of the source pattern and/or the readback introduces a cast. The `pattern.to(device)` upload itself could already be the lossy step (large arange → fp16 on device), since the comparison is against the *host* `pattern` which never went to the device.~~ Confirmed: `copy_d2h` uses `_C.copy_tensor`. The lossy step is the layout conversion this DMA applies (see below), not the `pattern.to(device)` upload.

## Suggested next steps

> **Superseded.** Items 1–2 below were written before the root cause was known; item 1 (fallback re-rounding) and the fp32 framing in item 2 do not apply. The actionable fix is the [Proposed solution](#proposed-solution) — a raw, conversion-free copy path. Original items retained for the record.

1. ~~**Check fallbacks first** — re-run with `-W "error::torch_spyre.ops.fallbacks.FallbackWarning"` to see if `copy_d2h`/the upload silently routes through a CPU fallback that re-rounds.~~ Not a fallback: the conversion is the C++ DMA's own layout transform.
2. ~~**Decide whether bitwise equality is the right assertion.** If the copier is allowed to round-trip through fp32 or re-round, switch `torch.equal` → `torch.testing.assert_close` with a 1-ULP-appropriate tolerance.~~ The copier is contractually a byte-exact DMA, so ±1 ULP is a real bug in the transfer path — relaxing the assertion would mask it. Keep `torch.equal`.
3. **Isolate the lossy hop** — compare `src_spyre` read straight back (`copy_d2h` of the freshly uploaded tensor) against `pattern` to determine whether the loss is on upload (`pattern.to(device)`) or on the device→host copy itself.

## Proposed solution

Add support for raw (byte-for-byte) host↔device transfers that bypass layout/dtype data conversion, exposed through both the C++ stream API and a Python binding.

- Add a `raw_copy` parameter to `SpyreStream::copyAsync()`.
- Add a `_C.copy_tensor_raw` Python binding.

KV-cache (KVC) offloading moves device tensors to/from host storage to free up device memory. For this, we want to move the device tensor's raw bytes without applying any tiled-layout or dtype conversion.

Today, `SpyreStream::copyAsync()` (`torch_spyre/csrc/spyre_stream.cpp:142`) always builds a `DataConversionInfo` (`dci`) via `generate_dci(...)` and passes it into `copyAsyncImpl`, which drives the layout/dtype conversion during DMA. There is currently no way to perform a conversion-free transfer from Python.

## Root cause: why `.to()` works but `copy_tensor()` does not

### `.to()` round-trips a large tensor cleanly; the unit test does not

A large-tensor round trip through `Tensor.to()` (device → host → device) was tested directly via the torch-spyre unit tests and showed **no** value mismatch. The same large-tensor data moved through `_C.copy_tensor()` in the KVC offloading round-trip test reproduces the ±1 ULP tail mismatch documented above. Both APIs ultimately call the same underlying C++ DMA routine in `spyre_mem.cpp`, so the difference is not in the copy primitive itself — it is in how the **device tensor's layout metadata** is established before the copy.

### Why the difference exists

The DMA descriptor (`DataConversionInfo`) is built **entirely from the device tensor's `SpyreTensorImpl`** — its `spyre_layout` (the `stl`), `dma_sizes`, and `dma_strides` (`spyre_mem.cpp:389–430`). The host (CPU) tensor contributes only its raw `sizes()`/`strides()`; **no layout metadata is read off the host side** (`spyre_mem.cpp:407–423`). The stick-tiling stride map that scatters/gathers fp16 elements between contiguous host RAM and tiled device storage is derived from `stl.stride_map` / `stl.device_size` (`get_device_stride_infos`, `spyre_mem.cpp:176–183`).

The two paths differ in **which layout the device tensor carries**, not in whether metadata is present:

| Path | Device-tensor allocator | Layout used |
|---|---|---|
| `.to()` (with `device_layout`) | `spyre_empty_with_layout` (`_monkey_patch.py:136`) | caller-supplied / stride-aware `SpyreTensorLayout(size, stride, …)` |
| copier test | C++ `spyre_empty` via `torch.zeros` (not monkey-patched) | **default** `SpyreTensorLayout(size, dtype)` (`spyre_mem.cpp:468`) |

The monkey-patch in `_monkey_patch.py` overrides only `torch.empty` and `torch.Tensor.to` — **not** `torch.zeros`. So the test's `dst_spyre = torch.zeros(2, 16, 64, device="spyre")` is allocated by the C++ `spyre_empty` path, which does set `spyre_layout`/`dma_sizes`/`dma_strides` (`spyre_mem.cpp:484–487`) — but with a **default contiguous** layout built from the logical shape alone, with no `stride`/`device_layout` argument. `.to()` instead routes through the stride-aware `spyre_empty_with_layout`, so the layout the data is written under matches the layout it is read back under.

Because the DMA stride map is computed solely from the device tensor's layout, any mismatch between that default contiguous layout and the actual tiled storage places/reads the boundary elements of each 64-element stick differently. The misplacement only becomes *visible* where the fp16 grid is coarse enough (ULP ≥ 1.0, i.e. values ≥ ~1024) that an off-by-one element lands on a different representable value — exactly the "exact at small magnitudes, ±1 ULP at the tail" pattern observed in [What we saw](#what-we-saw). This supersedes the fp16-cast guesses in [Interpretation / leads](#interpretation--leads): the apparent re-rounding is a symptom of layout-driven element misplacement, not a dtype cast in the copy itself.

The proposed `raw_copy` / `_C.copy_tensor_raw` path sidesteps this entirely: a byte-for-byte transfer that bypasses `DataConversionInfo` does not consult the device tensor's stride map, so a default-contiguous device tensor no longer corrupts the boundary elements.
