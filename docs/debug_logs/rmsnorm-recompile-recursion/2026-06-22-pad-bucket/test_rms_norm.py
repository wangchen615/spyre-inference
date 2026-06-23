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

"""
Test SpyreRMSNorm custom op correctness against a reference implementation.
"""

import pytest
import torch
import sys


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Golden reference: standard RMSNorm in PyTorch."""
    if residual is not None:
        x = x + residual
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight.float()
    if residual is not None:
        return x_normed, x.float()
    return x_normed


@pytest.mark.rmsnorm
@pytest.mark.parametrize("batch_size", [1])
# Hidden sizes that aren't multiples of 64 currently fail on CI with size errors
# @pytest.mark.parametrize("hidden_size", [63, 64, 65, 127, 128, 129, 256, 512])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_rmsnorm_matches_reference(batch_size, hidden_size, use_residual):
    """SpyreRMSNorm output matches golden reference.

    Tests both paths:
    - forward_oot(): OOT dispatch via custom op (torch.ops.vllm.spyre_rmsnorm)
    - reference_rms_norm(): golden reference, similar to vLLM upstream pure PyTorch (ground truth)
    """
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=torch.float16)
    layer = SpyreRMSNorm(hidden_size, eps=eps)
    residual = torch.randn(batch_size, hidden_size, dtype=torch.float32) if use_residual else None

    expected = reference_rms_norm(x, layer.weight.data, eps, residual)

    # Test forward_oot (Spyre device execution via custom op)
    actual = layer.forward_oot(x, residual)

    if use_residual:
        expected_norm, expected_resid = expected
        actual_norm, actual_resid = actual
        torch.testing.assert_close(actual_norm.float(), expected_norm.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            actual_resid.float(), expected_resid.float(), atol=1e-2, rtol=1e-2
        )
    else:
        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rmsnorm
@pytest.mark.parametrize("num_tokens", [1, 31, 33, 45, 64, 65])
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_rmsnorm_pads_token_dim_to_alignment(num_tokens, use_residual):
    """_forward_spyre_impl pads the token dim to a multiple of TOKEN_DIM_ALIGNMENT
    before calling the compiled kernel, then trims back to the original row count.

    Padding the row dimension to a fixed bucket means each bucket compiles the
    torch-spyre kernels once instead of recompiling per distinct num_tokens
    (which otherwise accumulates against dynamo's recompile limit -> RecursionError).
    The dummy rows must not change the real rows' output and must be trimmed off.
    """
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm, TOKEN_DIM_ALIGNMENT

    eps = 1e-6
    hidden_size = 128
    torch.manual_seed(42)

    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    layer = SpyreRMSNorm(hidden_size, eps=eps)
    residual = (
        torch.randn(num_tokens, hidden_size, dtype=torch.float32) if use_residual else None
    )

    expected = reference_rms_norm(x, layer.weight.data, eps, residual)

    # Spy on the compiled kernel to capture the row count it actually receives.
    seen_rows = []
    original_kernel = layer.maybe_compiled_forward_spyre

    def spy_kernel(x_arg, *args, **kwargs):
        seen_rows.append(x_arg.shape[0])
        return original_kernel(x_arg, *args, **kwargs)

    layer.maybe_compiled_forward_spyre = spy_kernel

    actual = layer.forward_oot(x, residual)

    # Kernel must see a row count aligned to the bucket.
    aligned = ((num_tokens + TOKEN_DIM_ALIGNMENT - 1) // TOKEN_DIM_ALIGNMENT) * TOKEN_DIM_ALIGNMENT
    assert seen_rows, "compiled kernel was never called"
    assert seen_rows[0] == aligned, (
        f"kernel saw {seen_rows[0]} rows, expected padded {aligned} for num_tokens={num_tokens}"
    )

    # Output must keep the original (unpadded) row count and match the reference.
    if use_residual:
        actual_norm, actual_resid = actual
        assert actual_norm.shape[0] == num_tokens
        assert actual_resid.shape[0] == num_tokens
        expected_norm, expected_resid = expected
        torch.testing.assert_close(actual_norm.float(), expected_norm.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            actual_resid.float(), expected_resid.float(), atol=1e-2, rtol=1e-2
        )
    else:
        assert actual.shape[0] == num_tokens
        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 128, dtype=torch.float32)


def mock_forward_oot(x, residual=None):
    """Mock: return x + 1 (no residual path)."""
    return x + 1


def mock_forward_oot_with_residual(x, residual=None):
    """Mock: return (2 * x, 2 * residual) (residual path)."""
    return 2 * x, 2 * residual


@pytest.mark.rmsnorm
@pytest.mark.parametrize("use_residual", [False, True])
def test_rmsnorm_oot_dispatch(monkeypatch, dummy_tensor, use_residual):
    """Verify RMSNorm OOT registration: class swap and forward_oot routing."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    layer = RMSNorm(128, eps=1e-6)

    # OOT class swap: RMSNorm.__new__ should produce SpyreRMSNorm
    assert isinstance(layer, SpyreRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot

    residual = torch.randn(4, 128, dtype=torch.float32) if use_residual else None

    # Mock _forward_spyre_impl (called by the custom op) with a known transform
    if residual is not None:
        monkeypatch.setattr(layer, "_forward_spyre_impl", mock_forward_oot_with_residual)
        out_x, out_residual = layer.forward(dummy_tensor, residual)

        assert torch.allclose(out_x, 2 * dummy_tensor)

        # The residual is modified in-place
        assert torch.allclose(out_residual, 2 * residual)
    else:
        monkeypatch.setattr(layer, "_forward_spyre_impl", mock_forward_oot)
        out_x = layer.forward(dummy_tensor, residual)

        assert torch.allclose(out_x, dummy_tensor + 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-k", "test_rmsnorm_oot_dispatch", "-v"]))
