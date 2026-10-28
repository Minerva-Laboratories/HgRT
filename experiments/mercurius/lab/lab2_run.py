"""attention lab: v2 kernel (a) vs flattened decode (b), correctness (bit-equal) and time"""
import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'), 'src')); sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'), 'experiments', 'mercurius'))
from torch.utils.cpp_extension import load
from cairn.mercurius_engine import HD, NG, NH, RD, MercuriusEngine
from cairn.mercurius_kernels import ext
from gemv_tune import bench
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
bd = os.path.expanduser("~/.cache/cairn/mla_lab2"); os.makedirs(bd, exist_ok=True)
lab = load("mla_lab2", [os.path.join(os.path.dirname(os.path.abspath(__file__)), "mla_lab2.cu")], build_directory=bd,
           extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler"])
E = MercuriusEngine(os.path.join(os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius"), "weights.pt"), Tmax=4096)
X = ext()
for i in (E.attn_ids[2], E.attn_ids[-1]):
    d = E.layers[i]; r, Ep, RW = d["r"], d["Ep"], d["RW"]
    for L in (8192, 32768):
        cache = torch.zeros(L + 8, RW, dtype=torch.uint8, device="cuda")
        kk = torch.randn(64, NG * HD, device="cuda").to(torch.bfloat16)
        for a in range(0, 64, 4):
            X.mla_encode(kk[a:a+4], NG * HD, torch.randn(4, RD, device="cuda").to(torch.bfloat16).float(), torch.randn(4, r, device="cuda").to(torch.bfloat16).float(),
                         r, d["cbl"], d["cbr"], torch.tensor([a], device="cuda"), cache, RW, d["k_eps"], 4)
        cache[64:L + 4] = cache[0:64].repeat(L // 64 + 1, 1)[:L - 60]
        cache[:L + 4, 0:4] = torch.arange(L + 4, device="cuda", dtype=torch.int32).view(torch.uint8).view(-1, 4)
        Q = (torch.randn(4, NH, Ep, device="cuda") * 0.3).half()
        line = f"layer {i} r={r} L={L}:"
        for NQ in (1, 4):
            VS = 32
            outs = {}
            for v in ("attn_a", "attn_c"):
                po = torch.empty(VS, NQ, NH, Ep - RD, device="cuda"); pm = torch.empty(VS, NQ * NH * 2, device="cuda")
                o = torch.empty(NQ, NH, r, device="cuda"); p = torch.tensor([L - NQ], device="cuda")
                f = getattr(lab, v)
                t = bench(lambda: f(Q[:NQ].contiguous(), Ep, cache, RW, d["R0h"], d["cbl"], d["cbr"], p, E.inv_freq, po, pm, o, r, d["scale"], NQ), it=10)
                outs[v] = (t, o.clone())
            same = torch.equal(outs["attn_a"][1], outs["attn_c"][1])
            line += f" | NQ{NQ}: a {outs['attn_a'][0]*1e3:.3f} ms, c(512 thr) {outs['attn_c'][0]*1e3:.3f} ms, bit-equal {same}"
        print(line, flush=True)
print("LAB2_DONE")
