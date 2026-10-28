"""Triton wide_attention tiling for the MLA prefill shape (16 heads, 4 groups, keys 720, values 256)."""
import os
import sys, time, torch
sys.path.insert(0, os.path.join(os.environ.get("OPENDECIDER_ROOT", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), "third_party", "opendecider")), "src"))
from opendecider.wide_attention import wide_attention
torch.manual_seed(0)
T, S = 2048, 8192
q = torch.randn(1, 16, T, 448, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 4, S, 448, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 4, S, 256, device="cuda", dtype=torch.bfloat16)
for cfg in [dict(BM=64, BN=32, BD=64, num_warps=4, num_stages=1), dict(BM=64, BN=64, BD=64, num_warps=4, num_stages=1),
            dict(BM=64, BN=32, BD=64, num_warps=4, num_stages=2), dict(BM=128, BN=32, BD=64, num_warps=8, num_stages=1),
            dict(BM=64, BN=32, BD=64, num_warps=8, num_stages=1), dict(BM=32, BN=32, BD=64, num_warps=4, num_stages=1), dict(BM=64, BN=32, BD=32, num_warps=4, num_stages=1),
            dict(BM=32, BN=64, BD=32, num_warps=4, num_stages=1), dict(BM=64, BN=64, BD=16, num_warps=4, num_stages=1),
            dict(BM=64, BN=32, BD=64, num_warps=4, num_stages=1), dict(BM=128, BN=32, BD=32, num_warps=8, num_stages=1),
            dict(BM=64, BN=32, BD=32, num_warps=8, num_stages=1)]:
    try:
        o = wide_attention(q, k, v, 0.0625, **cfg); torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            o = wide_attention(q, k, v, 0.0625, **cfg)
        torch.cuda.synchronize()
        dt = (time.time() - t) / 3
        fl = 2 * 16 * T * (S - T / 2) * (448 + 256)
        print(cfg, f"{dt*1e3:.1f} ms, {fl / dt / 1e12:.2f} TFLOP/s", flush=True)
    except Exception as e:
        print(cfg, "failed:", str(e)[:100], flush=True)
# PyTorch SDPA (efficient backend) with the same shapes, GQA expanded
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
qq, kk, vv = q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
for be in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION):
    try:
        with sdpa_kernel(be):
            mask = torch.ones(T, S, dtype=torch.bool, device="cuda").tril(S - T)
            o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=0.0625); torch.cuda.synchronize()
            t = time.time()
            for _ in range(3):
                o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=0.0625)
            torch.cuda.synchronize()
            dt = (time.time() - t) / 3
            print(be, f"{dt*1e3:.1f} ms, {2 * 16 * T * (S - T / 2) * 704 / dt / 1e12:.2f} TFLOP/s", flush=True)
    except Exception as e:
        print(be, "failed:", str(e)[:120], flush=True)
print("WA_DONE")
