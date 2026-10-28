"""One decode-time block encode (128 tokens, one layer): time of each part (CUDA events, median of 20)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from hgrt.caches import make_cache
from hgrt.stack import BLK, D, G, fg
from sparsepage import summarize
HKV = 4
c = make_cache("stack10", HKV, 8192)
k = (torch.randn(2048, HKV, D, device="cuda") * .8).bfloat16(); v = torch.randn(2048, HKV, D, device="cuda").bfloat16()
c.prefill(k, v)
kx, vx = c.kraw[:BLK].clone(), c.vraw[:BLK].clone()
a = c.n_comp
def t(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); out = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize(); out.append(s.elapsed_time(e))
    return float(np.median(out))
import time
def host(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); h = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - h) / n
print("foveakv GainShape._encode (k+v)", t(lambda: fg.GainShape._encode(c, a - BLK, a, kx, vx)), "ms")
print("sparsepage summarize", t(lambda: summarize(kx)), "ms")
print("full StackCache._encode", t(lambda: c._encode(a - BLK, a, kx, vx)), "ms; host-timed", host(lambda: c._encode(a - BLK, a, kx, vx)))
print("kraw shift clone", t(lambda: c.kraw[:256].copy_(c.kraw[128:384].clone())), "ms")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
    summarize(kx); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cpu_time_total", row_limit=12))
