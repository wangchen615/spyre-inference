# Snapshot — 2026-06-22 pad/bucket attempt (RMSNorm token-dim padding)

Files as-tried for the 2026-06-22 entry in `../../rmsnorm-recompile-recursion.md`.
These are reference copies so the attempt is reproducible even if the working tree
changes or the edits are reverted. **Do not import from here** — the live code is in
`spyre_inference/` and `tests/`.

- `changes.diff` — `git diff` of exactly what was changed (apply with `git apply` from repo root).
- `rms_norm.py` — full copy of `spyre_inference/custom_ops/rms_norm.py` as-tried.
- `test_rms_norm.py` — full copy of `tests/test_rms_norm.py` as-tried.

## What changed

### `spyre_inference/custom_ops/rms_norm.py`

- Added module constant `TOKEN_DIM_ALIGNMENT = 32`.
- In `_forward_spyre_impl`, before the compiled-kernel call:
  ```python
  num_tokens = x.shape[0]
  pad_rows = (-num_tokens) % TOKEN_DIM_ALIGNMENT
  if pad_rows:
      x = torch.nn.functional.pad(x, (0, 0, 0, pad_rows))
      if residual is not None:
          residual = torch.nn.functional.pad(residual, (0, 0, 0, pad_rows))
  ```
  and on the way out, trim the padding rows in the result `tree_map`:
  `convert(el[:num_tokens], ...)`.
- Updated the `_forward_spyre_impl` docstring to describe the padding (it previously,
  incorrectly, claimed "Pads to 64").
- `forward_spyre` (the compiled math) unchanged.

### `tests/test_rms_norm.py`

- Added `test_spyre_rmsnorm_pads_token_dim_to_alignment`, parametrized over
  `num_tokens ∈ {1, 31, 33, 45, 64, 65}` × `use_residual ∈ {False, True}`.
- It spies on `maybe_compiled_forward_spyre` to assert the kernel receives a row count
  aligned to `TOKEN_DIM_ALIGNMENT`, and asserts the returned tensors keep the original
  (unpadded) row count and match the CPU reference.

## Result

RMSNorm unit tests pass on Spyre hardware (`22 passed`, fallback-as-error). The end-to-end
repro no longer crashes in RMSNorm; it now crashes in the attention `spyre::overwrite` scatter
(separate issue). See the dated entry in the parent tracker for the verified trace details.
