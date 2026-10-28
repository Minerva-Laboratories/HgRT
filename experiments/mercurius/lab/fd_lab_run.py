"""instrumented attention: per-phase clock cycles per chunk (block 0, thread 0)"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'), 'src'))
from torch.utils.cpp_extension import load
from hgrt.mercurius_engine import HD, NG, NH, RD, MercuriusEngine
from hgrt.mercurius_kernels import ext
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
lab = load("fd_lab", [os.path.join(os.path.dirname(os.path.abspath(__file__)), "fd_lab.cu")],
           build_directory=os.path.expanduser("~/.cache/hgrt/fd_lab"), extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler"])
E = MercuriusEngine(os.path.join(os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius"), "weights.pt"), Tmax=4096)
X = ext()
names = ["loop end->S1", "S1 sync", "meta+raw rope", "rope gemm", "rope apply", "S", "softmax", "PV"]
for i in (E.attn_ids[2], E.attn_ids[-1]):
    d = E.layers[i]; r, Ep, RW = d["r"], d["Ep"], d["RW"]
    L = 32768
    cache = torch.zeros(L + 8, RW, dtype=torch.uint8, device="cuda")
    kk = torch.randn(4, NG * HD, device="cuda").to(torch.bfloat16)
    X.mla_encode(kk, NG * HD, torch.randn(4, RD, device="cuda").to(torch.bfloat16).float(), torch.randn(4, r, device="cuda").to(torch.bfloat16).float(),
                 r, d["cbl"], d["cbr"], torch.tensor([0], device="cuda"), cache, RW, d["k_eps"], 4)
    cache[4:L + 4] = cache[0:4].repeat(L // 4 + 1, 1)[:L]
    cache[:L + 4, 0:4] = torch.arange(L + 4, device="cuda", dtype=torch.int32).view(torch.uint8).view(-1, 4)
    Q = (torch.randn(4, NH, Ep, device="cuda") * 0.3).half()
    for NQ in (1, 4):
        NS = 32
        po = torch.empty(NS, NQ, NH, Ep - RD, device="cuda"); pm = torch.empty(NS, NQ * NH * 2, device="cuda")
        o = torch.empty(NQ, NH, r, device="cuda"); p = torch.tensor([L - NQ], device="cuda")
        prof = torch.zeros(16, dtype=torch.int64, device="cuda")
        for _ in range(30):
            lab.fd(Q[:NQ].contiguous(), Ep, cache, RW, d["R0h"], d["cbl"], d["cbr"], p, E.inv_freq, po, pm, o, r, d["scale"], NQ, prof)
        prof.zero_(); torch.cuda.synchronize()
        import time; t = time.time()
        for _ in range(20):
            lab.fd(Q[:NQ].contiguous(), Ep, cache, RW, d["R0h"], d["cbl"], d["cbr"], p, E.inv_freq, po, pm, o, r, d["scale"], NQ, prof)
        torch.cuda.synchronize(); dt = (time.time() - t) / 20
        nch = (L + AC - 1) // 32 if (AC := 32) else 0
        per = prof.cpu().double() / 20 / (nch / NS)
        tot = float(per.sum())
        print(f"layer {i} r={r} NQ={NQ}: {dt*1e3:.3f} ms; cycles per chunk {tot:.0f} (block 0): " +
              ", ".join(f"{n} {float(per[k]):.0f}" for k, n in enumerate(names)), flush=True)
print("FDLAB_DONE")
