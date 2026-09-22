"""Minimal repro: torch.compile on spyre produces NaN where eager does not.

    python repro.py            # compiled (expect NaN)
    EAGER=1 python repro.py    # eager (expect finite)
"""
import os
import torch
import torch_spyre  # noqa: F401

EAGER = os.environ.get("EAGER", "0") not in ("0", "false", "")
torch.manual_seed(0)


def block(x, w1, w2, g):
    # RMSNorm-ish + gated MLP, the shape a transformer block reduces to.
    v = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * g
    h = v @ w1
    h = h * torch.sigmoid(h)
    return x + h @ w2


D, F, T = 512, 1024, 8
x = torch.randn(1, T, D, dtype=torch.float16)
w1 = (torch.randn(D, F, dtype=torch.float16) * 0.02)
w2 = (torch.randn(F, D, dtype=torch.float16) * 0.02)
g = torch.ones(D, dtype=torch.float16)

ref = block(x, w1, w2, g)
xs, w1s, w2s, gs = (t.to("spyre") for t in (x, w1, w2, g))
fn = block if EAGER else torch.compile(block, fullgraph=True)
out = fn(xs, w1s, w2s, gs).cpu().float()

print("mode:", "eager" if EAGER else "compiled")
print("has_nan:", bool(torch.isnan(out).any()))
print("nan_frac:", float(torch.isnan(out).float().mean()))
print("max_abs_diff_vs_cpu:", float((out - ref.float()).abs().max()))
