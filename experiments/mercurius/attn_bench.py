"""MLA attention from the TurboQuant codes: time vs split count NS, query rows NQ and context length (synthetic cache)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'src'))
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'experiments', 'mercurius'))
from cairn.mercurius_engine import HD, NG, NH, RD, MercuriusEngine
from cairn.mercurius_kernels import ext
from gemv_tune import bench

OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")


@torch.no_grad()
def main():
    E = MercuriusEngine(os.path.join(OUT, "weights.pt"), Tmax=4096)
    X = ext()
    for i in (E.attn_ids[2], E.attn_ids[-1]):                  # r = 390 and r = 656
        d = E.layers[i]
        r, Ep, RW = d["r"], d["Ep"], d["RW"]
        for L in (1024, 8192, 32768):
            cache = torch.zeros(L + 8, RW, dtype=torch.uint8, device="cuda")
            kk = torch.randn(4, NG * HD, device="cuda").to(torch.bfloat16)
            for a in range(0, L, 4):                           # cheap fill: the same 4 rows encoded at every position
                pass
            kr = torch.randn(4, RD, device="cuda").to(torch.bfloat16).float()
            cr = torch.randn(4, r, device="cuda").to(torch.bfloat16).float()
            X.mla_encode(kk, NG * HD, kr, cr, r, d["cbl"], d["cbr"], torch.tensor([0], device="cuda"), cache, RW, d["k_eps"], 4)
            cache[4:L + 4] = cache[0:4].repeat(L // 4 + 1, 1)[:L]
            cache[:L + 4, 0:4] = torch.arange(L + 4, device="cuda", dtype=torch.int32).view(torch.uint8).view(-1, 4)
            Q = (torch.randn(4, NH, Ep, device="cuda") * 0.3).half()
            line = f"layer {i} r={r} L={L}:"
            for NQ in (1, 2, 3, 4):
                NS = 32
                po = torch.empty(NS, NQ, NH, Ep - RD, device="cuda")
                pm = torch.empty(NS, NQ * NH * 2, device="cuda")
                o = torch.empty(NQ, NH, r, device="cuda")
                p = torch.tensor([L - NQ], device="cuda")
                for fname in ("mla_attn", "mla_attn_fd"):
                    f = getattr(X, fname)
                    t = bench(lambda: f(Q[:NQ].contiguous(), Ep, cache, RW, d["R0h"], d["cbl"], d["cbr"], p,
                                        E.inv_freq, po, pm, o, r, d["scale"], NQ), it=10)
                    line += f" NQ{NQ} {'v2' if fname == 'mla_attn' else 'fd'} {t*1e3:.3f}"
            print(line, f"| code bytes {L * RW / 1e6:.1f} MB", flush=True)
    print("ATTN_DONE")


if __name__ == "__main__":
    main()
