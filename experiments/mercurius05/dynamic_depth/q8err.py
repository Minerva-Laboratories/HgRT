import os, sys, torch, subprocess
sys.path.insert(0, os.environ["HGRT_SRC"])
from hgrt.mercurius_packed import PackedMercuriusEngine
e = PackedMercuriusEngine(os.path.expanduser("~/Desktop/Models/Mercurius-0.5-4B-Preview"), Tmax=256)
g = torch.Generator().manual_seed(0)
for li in (0, 10, 22):
    d = [d for d in e.layers if d["kind"] == "gdn2"][li]
    lin = d["gates"]
    W = lin.dequant_exact().double()
    for M in (1, 3):
        x = torch.randn(M, W.shape[1], generator=g).to("cuda", torch.bfloat16)
        ref = x.double() @ W.T
        y = lin(x, M, torch.empty(M, W.shape[0], device="cuda"))
        err = ((y.double() - ref).abs().max() / ref.abs().max()).item()
        rms = ((y.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"{os.environ.get('HGRT_TMP_Q8FMA') and 'fma' or 'mma'} layer {li} M={M}: max rel {err:.2e} rms rel {rms:.2e}")
