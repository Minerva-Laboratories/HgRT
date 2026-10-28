"""nf4_mma (int8 tensor cores) vs nf4_gemv (dp4a): correctness against the fp64 reference of the integer math, M
invariance (bit), and speed for M = 1..4 on the engine's matrices."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'src'))
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'tests'))
from hgrt.mercurius_engine import MercuriusEngine
from hgrt.mercurius_kernels import ext, mma_layout
from gemv_tune import bench

OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")


@torch.no_grad()
def main():
    from test_mercurius import nf4_ref, rel
    E = MercuriusEngine(os.path.join(OUT, "weights.pt"), Tmax=4096)
    X = ext()
    d0, a0 = E.layers[E.lin_ids[0]], E.layers[E.attn_ids[0]]
    lins = {"mlp gate|up": d0["gu"], "mlp down": d0["down"], "gdn qkv|z": d0["qkvz"], "gdn out": d0["out"],
            "mla q|down|krms": a0["qdk"], "mla o": a0["o"], "lm head": E.head}
    for name, L in lins.items():
        packed = L.codes.view(torch.uint8).view(L.N, L.K // 2)
        mc = mma_layout(packed)
        f32 = name == "lm head"
        nbytes = L.codes.numel() * 4 + L.scales.numel() * 4
        line = f"{name:18s} N={L.N:6d} K={L.K:5d}"
        outs = {}
        for M in (1, 2, 3, 4):
            x = torch.randn(M, L.K, device="cuda").to(torch.bfloat16)
            xq, xd = torch.empty(M * L.K, dtype=torch.int8, device="cuda"), torch.empty(M * L.K // 32, device="cuda")
            X.quant_rows(x, L.K, xq, xd, M, L.K)
            y = torch.empty(M, L.N, dtype=torch.float32 if f32 else torch.bfloat16, device="cuda")
            best = None
            for ks in (1, 2, 4, 8, 101, 102, 104, 108):
                if L.K // 128 < ks % 100:
                    continue
                t = bench(lambda: X.nf4_mma(mc, L.scales, xq, xd, None, 0, y, L.N, M, ks))
                if best is None or t < best[1]:
                    best = (ks, t)
            X.nf4_mma(mc, L.scales, xq, xd, None, 0, y, L.N, M, best[0])
            if M == 3 and name != "lm head":
                ref = nf4_ref(L, E.levels, x)
                ok = rel(y.float(), ref) < 4e-3
                line += f" | rel {rel(y.float(), ref):.1e} {'ok' if ok else 'BAD'}"
                y1 = torch.empty(1, L.N, dtype=y.dtype, device="cuda")
                xq1, xd1 = torch.empty(L.K, dtype=torch.int8, device="cuda"), torch.empty(L.K // 32, device="cuda")
                X.quant_rows(x[2:3], L.K, xq1, xd1, 1, L.K)
                X.nf4_mma(mc, L.scales, xq1, xd1, None, 0, y1, L.N, 1, best[0])
                line += f" bit(M1==M3 row2) {torch.equal(y1[0], y[2])}"
            tg = bench(lambda: X.nf4_gemv(L.codes, L.scales, xq, xd, None, 0, y, L.N, M,
                                          __import__('hgrt.mercurius_engine', fromlist=['rows_for']).rows_for(L.N, L.K, M)))
            line += f" | M={M}: mma ks={best[0]} {best[1]*1e6:.0f}us ({nbytes / best[1] / 1e9:.0f} GB/s) dp4a {tg*1e6:.0f}us"
        print(line, flush=True)
    print("MMA_DONE")


if __name__ == "__main__":
    main()
